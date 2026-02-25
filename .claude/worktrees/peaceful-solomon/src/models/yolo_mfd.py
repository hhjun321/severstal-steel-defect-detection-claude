"""
YOLO-MFD: YOLOv8 with Multi-scale Edge Feature Enhancement (Ultralytics)

Uses ultralytics YOLO as the base, with the MEFE (Multi-scale Edge Feature
Enhancement) module injected into the model architecture via a custom model
YAML definition.

The MEFE module is the paper's key contribution:
  - Sobel-based multi-scale edge feature extraction
  - Channel attention for edge-feature fusion
  - Applied at P3/P4 feature levels for micro-defect enhancement

For training, use ultralytics native .train() which provides:
  - Proper CIoU/DFL loss, Task-Aligned Label Assignment
  - Built-in NMS, AMP, cosine LR, early stopping, checkpointing
  - Proven mAP evaluation pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from pathlib import Path


# ============================================================================
# MEFE Module Components (paper's contribution - preserved from original)
# ============================================================================

class SobelEdgeExtractor(nn.Module):
    """Multi-scale Sobel edge feature extraction."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > 1:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)


class MEFE(nn.Module):
    """
    Multi-scale Edge Feature Enhancement (MEFE) module.

    This is a drop-in module compatible with ultralytics model YAML.
    It takes a feature map, extracts Sobel edges at multiple scales,
    and fuses them with the original features via channel attention.

    Args:
        c1: Input channels (from previous layer)
        c2: Output channels (typically == c1 for residual compatibility)
    """

    def __init__(self, c1: int, c2: int = None):
        super().__init__()
        if c2 is None:
            c2 = c1  # ultralytics custom modules may receive only c1 from YAML args
        self.c1 = c1
        self.c2 = c2
        self.edge_extractor = SobelEdgeExtractor()

        # Multi-scale edge processing branches
        branch_ch = c2 // 4
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(1, branch_ch, 1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(1, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(1, branch_ch, 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(1, branch_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.SiLU(inplace=True),
        )

        # Fusion: concat original features + edge features -> output channels
        self.fusion = nn.Sequential(
            nn.Conv2d(c1 + c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # Channel attention gate
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2 // 4, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract Sobel edges
        edge = self.edge_extractor(x)

        # Multi-scale edge processing
        e1 = self.branch_1x1(edge)
        e2 = self.branch_3x3(edge)
        e3 = self.branch_5x5(edge)
        e4 = self.branch_dilated(edge)
        edge_features = torch.cat([e1, e2, e3, e4], dim=1)

        # Fuse with original features
        fused = torch.cat([x, edge_features], dim=1)
        fused = self.fusion(fused)

        # Channel attention
        att = self.attention(fused)
        return fused * att


# ============================================================================
# YOLO-MFD Model Definition (custom YAML for ultralytics)
# ============================================================================

# YOLOv8s architecture with MEFE modules inserted after backbone P3 and P4
# This YAML string is written to a file and loaded by ultralytics
YOLO_MFD_YAML = """
# YOLOv8s-MFD: YOLOv8s with Multi-scale Edge Feature Enhancement
# Channel widths are pre-scaled for YOLOv8s (width=0.50, depth=0.33, max_ch=1024).
# No 'scales' block — all values are final to ensure MEFE gets correct channel counts.
# Depth: repeats already adjusted (3→1, 6→2).

nc: 4

backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]           # 0: P1/2    (64*0.50=32)
  - [-1, 1, Conv, [64, 3, 2]]           # 1: P2/4    (128*0.50=64)
  - [-1, 1, C2f, [64, True]]            # 2          (128*0.50=64, repeats=ceil(3*0.33)=1)
  - [-1, 1, Conv, [128, 3, 2]]          # 3: P3/8    (256*0.50=128)
  - [-1, 2, C2f, [128, True]]           # 4          (256*0.50=128, repeats=ceil(6*0.33)=2)
  - [-1, 1, MEFE, [128, 128]]           # 5: MEFE at P3 level ★ (in=128, out=128)
  - [-1, 1, Conv, [256, 3, 2]]          # 6: P4/16   (512*0.50=256)
  - [-1, 2, C2f, [256, True]]           # 7          (512*0.50=256, repeats=ceil(6*0.33)=2)
  - [-1, 1, MEFE, [256, 256]]           # 8: MEFE at P4 level ★ (in=256, out=256)
  - [-1, 1, Conv, [512, 3, 2]]          # 9: P5/32   (1024*0.50=512)
  - [-1, 1, C2f, [512, True]]           # 10         (1024*0.50=512, repeats=ceil(3*0.33)=1)
  - [-1, 1, SPPF, [512, 5]]            # 11         (1024*0.50=512)

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 12
  - [[-1, 8], 1, Concat, [1]]                    # 13: cat P4-MEFE (512+256=768→C2f→256)
  - [-1, 1, C2f, [256]]                          # 14

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]  # 15
  - [[-1, 5], 1, Concat, [1]]                    # 16: cat P3-MEFE (256+128=384→C2f→128)
  - [-1, 1, C2f, [128]]                          # 17 (P3/8-small)

  - [-1, 1, Conv, [128, 3, 2]]                   # 18
  - [[-1, 14], 1, Concat, [1]]                   # 19: (128+256=384→C2f→256)
  - [-1, 1, C2f, [256]]                          # 20 (P4/16-medium)

  - [-1, 1, Conv, [256, 3, 2]]                   # 21
  - [[-1, 11], 1, Concat, [1]]                   # 22: (256+512=768→C2f→512)
  - [-1, 1, C2f, [512]]                          # 23 (P5/32-large)

  - [[17, 20, 23], 1, Detect, [nc]]              # 24: Detect(P3, P4, P5)
"""


def get_yolo_mfd_yaml_path(output_dir: str = None) -> str:
    """
    Write the YOLO-MFD model YAML to disk and return its path.
    
    Args:
        output_dir: Directory to write the YAML file. If None, uses a temp location.
    
    Returns:
        Absolute path to the YAML file
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "configs")

    yaml_dir = Path(output_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / "yolov8s_mfd.yaml"

    with open(yaml_path, 'w') as f:
        f.write(YOLO_MFD_YAML)

    return str(yaml_path)


def create_yolo_mfd(num_classes: int = 4, pretrained: bool = True):
    """
    Create a YOLO-MFD model using ultralytics.
    
    This builds a YOLOv8s with MEFE modules injected at P3 and P4 backbone levels.
    The MEFE module must be registered with ultralytics before loading the YAML.
    
    Args:
        num_classes: Number of detection classes
        pretrained: If True, start from YOLOv8s pretrained weights (transfer learning)
    
    Returns:
        ultralytics YOLO model instance
    """
    from ultralytics import YOLO
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks_module

    # Register MEFE in ultralytics namespace.
    # ultralytics parse_model() resolves custom modules via globals()[m]
    # in ultralytics/nn/tasks.py — so MEFE must be injected there.
    if not hasattr(modules, 'MEFE'):
        modules.MEFE = MEFE
    # Critical: tasks.py uses globals() lookup for module classes
    if 'MEFE' not in vars(tasks_module):
        tasks_module.MEFE = MEFE
    # Also inject into tasks module's global dict directly
    tasks_module.__dict__['MEFE'] = MEFE

    # Write model YAML
    yaml_path = get_yolo_mfd_yaml_path()

    if pretrained:
        # Load pretrained YOLOv8s, then apply custom architecture
        # Strategy: build from custom YAML (random init for MEFE),
        # but transfer matching weights from pretrained YOLOv8s
        model = YOLO(yaml_path)
        # Load pretrained backbone weights (non-strict: MEFE layers will be random)
        pretrained_model = YOLO("yolov8s.pt")
        _transfer_weights(pretrained_model.model, model.model)
        del pretrained_model
    else:
        model = YOLO(yaml_path)

    return model


def _transfer_weights(src_model: nn.Module, dst_model: nn.Module):
    """
    Transfer matching weights from pretrained YOLOv8s to YOLO-MFD.
    
    Non-matching layers (MEFE) are left with random initialization.
    """
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()

    transferred = 0
    skipped = 0
    for name, param in dst_state.items():
        if name in src_state and src_state[name].shape == param.shape:
            dst_state[name] = src_state[name]
            transferred += 1
        else:
            skipped += 1

    dst_model.load_state_dict(dst_state, strict=False)

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Weight transfer: {transferred} layers transferred, {skipped} layers skipped (MEFE etc.)")


# ============================================================================
# Legacy interface (for backward compatibility with BenchmarkTrainer)
# Not used in ultralytics native training path.
# ============================================================================

class YOLOMFD(nn.Module):
    """
    Legacy wrapper — kept for import compatibility.
    
    For actual training, use create_yolo_mfd() which returns an ultralytics
    YOLO model that trains with .train() natively.
    """

    def __init__(self, num_classes: int = 4, variant: str = 's'):
        super().__init__()
        self.num_classes = num_classes
        raise NotImplementedError(
            "YOLOMFD legacy class is deprecated. "
            "Use create_yolo_mfd() for ultralytics-based training. "
            "See src/training/ultralytics_trainer.py for the training flow."
        )
