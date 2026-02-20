"""
YOLO-MFD: YOLO with Multi-scale Edge Feature Enhancement

Based on the 2025 YOLO-MFD architecture that adds a Multi-scale Edge Feature
Enhancement (MEFE) module to YOLOv8 for improved micro-defect detection on
steel surfaces.

Key modifications over standard YOLOv8:
  - MEFE module: Sobel-based multi-scale edge feature extraction
  - Edge-aware feature fusion in the neck
  - Designed for small/micro defect detection on Severstal-style images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class SobelEdgeExtractor(nn.Module):
    """Multi-scale Sobel edge feature extraction."""

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract edge features from input feature map."""
        # Convert to single channel (mean across channels)
        if x.shape[1] > 1:
            gray = x.mean(dim=1, keepdim=True)
        else:
            gray = x

        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        return edge_magnitude


class MEFEModule(nn.Module):
    """
    Multi-scale Edge Feature Enhancement (MEFE) module.
    
    Extracts edge features at multiple scales and fuses them with
    the original feature map for enhanced micro-defect representation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.edge_extractor = SobelEdgeExtractor()

        # Multi-scale processing branches
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(1, out_channels // 4, 1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(inplace=True),
        )
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(1, out_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(1, out_channels // 4, 5, padding=2, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(inplace=True),
        )
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(1, out_channels // 4, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.SiLU(inplace=True),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Attention gate
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract edge features
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

        # Apply channel attention
        att = self.attention(fused)
        fused = fused * att

        return fused


class ConvBlock(nn.Module):
    """Standard convolution block: Conv + BN + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """CSP bottleneck with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        mid_ch = int(out_ch * expansion)
        self.cv1 = ConvBlock(in_ch, mid_ch, 1)
        self.cv2 = ConvBlock(mid_ch, out_ch, 3)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.shortcut:
            out = out + x
        return out


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLOv8 style)."""

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.mid_ch = out_ch // 2
        self.cv1 = ConvBlock(in_ch, 2 * self.mid_ch, 1)
        self.cv2 = ConvBlock((2 + n) * self.mid_ch, out_ch, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(self.mid_ch, self.mid_ch, shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y = list(y.chunk(2, dim=1))
        for bn in self.bottlenecks:
            y.append(bn(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        mid_ch = in_ch // 2
        self.cv1 = ConvBlock(in_ch, mid_ch, 1)
        self.cv2 = ConvBlock(mid_ch * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DetectionHead(nn.Module):
    """YOLO detection head for multi-class bounding box prediction."""

    def __init__(self, in_channels_list: List[int], num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 4 + 1  # classes + bbox(4) + objectness

        self.heads = nn.ModuleList()
        for in_ch in in_channels_list:
            self.heads.append(nn.Sequential(
                ConvBlock(in_ch, in_ch, 3),
                ConvBlock(in_ch, in_ch, 3),
                nn.Conv2d(in_ch, self.num_outputs, 1),
            ))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        outputs = []
        for feat, head in zip(features, self.heads):
            outputs.append(head(feat))
        return outputs


class YOLOMFD(nn.Module):
    """
    YOLO-MFD: YOLOv8 with Multi-scale Edge Feature Enhancement.
    
    Architecture:
      Backbone: YOLOv8-s style CSP backbone
      Neck: PANet with MEFE module integration
      Head: Decoupled detection head
    """

    # Channel configs for small model variant
    CHANNELS = {
        's': {'backbone': [64, 128, 256, 512], 'neck': [256, 128, 256, 512]},
    }

    def __init__(self, num_classes: int = 4, variant: str = 's'):
        super().__init__()
        self.num_classes = num_classes
        ch = self.CHANNELS[variant]
        bc = ch['backbone']

        # ====== Backbone ======
        self.stem = ConvBlock(3, bc[0], 3, 2)  # /2

        self.stage1 = nn.Sequential(ConvBlock(bc[0], bc[1], 3, 2), C2f(bc[1], bc[1], n=3))  # /4
        self.stage2 = nn.Sequential(ConvBlock(bc[1], bc[2], 3, 2), C2f(bc[2], bc[2], n=6))  # /8  -> P3
        self.stage3 = nn.Sequential(ConvBlock(bc[2], bc[3], 3, 2), C2f(bc[3], bc[3], n=6))  # /16 -> P4
        self.stage4 = nn.Sequential(ConvBlock(bc[3], bc[3], 3, 2), C2f(bc[3], bc[3], n=3), SPPF(bc[3], bc[3]))  # /32 -> P5

        # ====== MEFE Modules ======
        self.mefe_p3 = MEFEModule(bc[2], bc[2])
        self.mefe_p4 = MEFEModule(bc[3], bc[3])

        # ====== Neck (PANet with MEFE) ======
        # Top-down path
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.td_conv1 = C2f(bc[3] + bc[3], bc[2], n=3, shortcut=False)  # P5+P4 -> N4
        self.td_conv2 = C2f(bc[2] + bc[2], bc[1], n=3, shortcut=False)  # N4+P3 -> N3

        # Bottom-up path
        self.bu_down1 = ConvBlock(bc[1], bc[1], 3, 2)
        self.bu_conv1 = C2f(bc[1] + bc[2], bc[2], n=3, shortcut=False)  # N3+N4 -> N4'
        self.bu_down2 = ConvBlock(bc[2], bc[2], 3, 2)
        self.bu_conv2 = C2f(bc[2] + bc[3], bc[3], n=3, shortcut=False)  # N4'+P5 -> N5'

        # ====== Detection Head ======
        self.head = DetectionHead([bc[1], bc[2], bc[3]], num_classes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)     # /8
        p4 = self.stage3(p3)    # /16
        p5 = self.stage4(p4)    # /32

        # Apply MEFE
        p3 = self.mefe_p3(p3)
        p4 = self.mefe_p4(p4)

        # Neck - top-down
        td4 = self.td_conv1(torch.cat([self.upsample(p5), p4], dim=1))
        td3 = self.td_conv2(torch.cat([self.upsample(td4), p3], dim=1))

        # Neck - bottom-up
        bu4 = self.bu_conv1(torch.cat([self.bu_down1(td3), td4], dim=1))
        bu5 = self.bu_conv2(torch.cat([self.bu_down2(bu4), p5], dim=1))

        # Head
        return self.head([td3, bu4, bu5])

    def compute_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO detection loss.
        
        Args:
            predictions: List of [B, num_outputs, H, W] tensors at 3 scales
            targets: List of [N, 5] tensors (class, cx, cy, w, h) per image
        
        Returns:
            Dict with 'total', 'box', 'obj', 'cls' losses
        """
        device = predictions[0].device
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        batch_size = predictions[0].shape[0]
        num_scales = len(predictions)

        for scale_idx, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]

            # Split predictions
            box_pred = pred[..., :4]  # cx, cy, w, h
            obj_pred = pred[..., 4:5]
            cls_pred = pred[..., 5:]

            # Build target tensors
            obj_target = torch.zeros(B, H, W, 1, device=device)
            box_target = torch.zeros(B, H, W, 4, device=device)
            cls_target = torch.zeros(B, H, W, self.num_classes, device=device)

            for b in range(B):
                if b >= len(targets) or targets[b].shape[0] == 0:
                    continue

                for t in targets[b]:
                    cls_id = int(t[0])
                    cx, cy, bw, bh = t[1], t[2], t[3], t[4]

                    # Map to grid cell
                    gi = int(cx * W)
                    gj = int(cy * H)
                    gi = min(max(gi, 0), W - 1)
                    gj = min(max(gj, 0), H - 1)

                    obj_target[b, gj, gi, 0] = 1.0
                    box_target[b, gj, gi] = torch.tensor([cx, cy, bw, bh], device=device)
                    cls_target[b, gj, gi, cls_id] = 1.0

            # Losses
            obj_mask = obj_target.squeeze(-1)  # [B, H, W]
            pos_count = obj_mask.sum().clamp(min=1)

            # Objectness loss (BCE)
            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred.squeeze(-1), obj_mask, reduction='sum'
            ) / pos_count

            # Box loss (CIoU-like: MSE on positive cells)
            if pos_count > 0:
                pos_mask = obj_mask.bool()
                box_loss = F.mse_loss(
                    box_pred[pos_mask], box_target[pos_mask], reduction='sum'
                ) / pos_count
            else:
                box_loss = torch.tensor(0.0, device=device)

            # Classification loss (BCE on positive cells)
            if pos_count > 0:
                pos_mask = obj_mask.bool()
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_pred[pos_mask], cls_target[pos_mask], reduction='sum'
                ) / pos_count
            else:
                cls_loss = torch.tensor(0.0, device=device)

            total_box_loss = total_box_loss + box_loss
            total_obj_loss = total_obj_loss + obj_loss
            total_cls_loss = total_cls_loss + cls_loss

        total_loss = total_box_loss * 5.0 + total_obj_loss * 1.0 + total_cls_loss * 0.5

        return {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'cls': total_cls_loss,
        }

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Run inference and apply NMS.
        
        Returns:
            List of dicts per image: {'boxes': [N,4], 'scores': [N], 'labels': [N]}
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for pred in outputs:
                B, C, H, W = pred.shape
                p = pred[b].permute(1, 2, 0).contiguous()  # [H, W, C]

                obj_scores = torch.sigmoid(p[..., 4])
                cls_scores = torch.sigmoid(p[..., 5:])

                # Grid offsets
                gy, gx = torch.meshgrid(
                    torch.arange(H, device=x.device, dtype=torch.float32),
                    torch.arange(W, device=x.device, dtype=torch.float32),
                    indexing='ij',
                )

                # Decode boxes
                cx = (torch.sigmoid(p[..., 0]) + gx) / W
                cy = (torch.sigmoid(p[..., 1]) + gy) / H
                bw = torch.sigmoid(p[..., 2])
                bh = torch.sigmoid(p[..., 3])

                # Filter by confidence
                combined_scores = obj_scores.unsqueeze(-1) * cls_scores
                max_scores, max_labels = combined_scores.max(dim=-1)
                mask = max_scores > conf_threshold

                if mask.sum() == 0:
                    continue

                # Extract filtered predictions
                filtered_cx = cx[mask]
                filtered_cy = cy[mask]
                filtered_bw = bw[mask]
                filtered_bh = bh[mask]
                filtered_scores = max_scores[mask]
                filtered_labels = max_labels[mask]

                # Convert to x1y1x2y2 (pixel coordinates)
                img_h, img_w = x.shape[2], x.shape[3]
                x1 = (filtered_cx - filtered_bw / 2) * img_w
                y1 = (filtered_cy - filtered_bh / 2) * img_h
                x2 = (filtered_cx + filtered_bw / 2) * img_w
                y2 = (filtered_cy + filtered_bh / 2) * img_h

                boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                all_boxes.append(boxes)
                all_scores.append(filtered_scores)
                all_labels.append(filtered_labels)

            if len(all_boxes) > 0:
                all_boxes = torch.cat(all_boxes, dim=0)
                all_scores = torch.cat(all_scores, dim=0)
                all_labels = torch.cat(all_labels, dim=0)

                # NMS per class
                from torchvision.ops import batched_nms
                keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold)
                results.append({
                    'boxes': all_boxes[keep],
                    'scores': all_scores[keep],
                    'labels': all_labels[keep],
                })
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), device=x.device),
                    'scores': torch.zeros(0, device=x.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=x.device),
                })

        return results
