"""
DeepLabV3+ for Severstal Steel Defect Segmentation

Standard segmentation baseline using DeepLabV3+ with ResNet-101 backbone,
as used in Severstal dataset analysis. Implements Atrous Spatial Pyramid Pooling
(ASPP) and encoder-decoder architecture for multi-class defect segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

try:
    import torchvision.models as models
    from torchvision.models import ResNet101_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class ASPPConv(nn.Module):
    """Atrous convolution with BN and ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        out = self.pool(x)
        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    
    Multi-rate atrous convolutions capture multi-scale context.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates: List[int] = [6, 12, 18]):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(ASPPConv(in_ch, out_ch, rate))
        modules.append(ASPPPooling(in_ch, out_ch))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        results = [conv(x) for conv in self.convs]
        return self.project(torch.cat(results, dim=1))


class Decoder(nn.Module):
    """DeepLabV3+ decoder with low-level feature fusion."""

    def __init__(self, low_level_ch: int, num_classes: int):
        super().__init__()
        # Reduce low-level feature channels
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Fusion convolutions
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1),
        )

    def forward(self, high_level: torch.Tensor, low_level: torch.Tensor) -> torch.Tensor:
        low_level = self.low_level_conv(low_level)
        high_level = F.interpolate(
            high_level, size=low_level.shape[2:],
            mode='bilinear', align_corners=False,
        )
        fused = torch.cat([high_level, low_level], dim=1)
        return self.fuse_conv(fused)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for multi-class steel defect segmentation.
    
    Architecture:
      Encoder: ResNet-101 (pretrained on ImageNet)
      ASPP: Multi-rate atrous convolutions (output_stride=16)
      Decoder: Low-level feature fusion + upsampling
    
    Output: (B, num_classes, H, W) logits
    """

    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet101",
        pretrained: bool = True,
        output_stride: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.output_stride = output_stride

        # ====== Encoder (ResNet backbone) ======
        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for DeepLabV3+")

        if backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if pretrained else None
            resnet = models.resnet101(weights=weights)
        elif backbone == "resnet50":
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify stride for output_stride=16
        if output_stride == 16:
            resnet.layer4[0].conv2.stride = (1, 1)
            resnet.layer4[0].downsample[0].stride = (1, 1)
            # Apply dilation to layer4
            for block in resnet.layer4:
                if hasattr(block.conv2, 'dilation'):
                    block.conv2.dilation = (2, 2)
                    block.conv2.padding = (2, 2)

        self.backbone_layers = nn.ModuleDict({
            'stem': nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
            'layer1': resnet.layer1,  # /4, 256 ch
            'layer2': resnet.layer2,  # /8, 512 ch
            'layer3': resnet.layer3,  # /16, 1024 ch
            'layer4': resnet.layer4,  # /16 (dilated), 2048 ch
        })

        # ====== ASPP ======
        if output_stride == 16:
            aspp_rates = [6, 12, 18]
        elif output_stride == 8:
            aspp_rates = [12, 24, 36]
        else:
            aspp_rates = [6, 12, 18]

        self.aspp = ASPP(2048, 256, aspp_rates)

        # ====== Decoder ======
        self.decoder = Decoder(low_level_ch=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]

        # Encoder
        x = self.backbone_layers['stem'](x)
        low_level = self.backbone_layers['layer1'](x)  # /4
        x = self.backbone_layers['layer2'](low_level)
        x = self.backbone_layers['layer3'](x)
        x = self.backbone_layers['layer4'](x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level)

        # Upsample to original size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute segmentation loss.
        
        Args:
            predictions: (B, num_classes, H, W) logits
            targets: (B, num_classes, H, W) binary masks
        
        Returns:
            Dict with 'total', 'bce', 'dice' losses
        """
        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # Dice loss
        probs = torch.sigmoid(predictions)
        smooth = 1.0
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()

        total = bce_loss * 0.5 + dice_loss * 0.5

        return {
            'total': total,
            'bce': bce_loss,
            'dice': dice_loss,
        }

    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Run inference and return binary masks.
        
        Returns:
            Dict with 'masks': (B, num_classes, H, W) binary masks,
                       'probs': (B, num_classes, H, W) probability maps
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return {
            'masks': masks,
            'probs': probs,
        }
