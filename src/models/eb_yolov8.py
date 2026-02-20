"""
EB-YOLOv8: Enhanced Backbone YOLOv8 with BiFPN

Based on the 2025 EB-YOLOv8 architecture that introduces Bi-directional
Feature Pyramid Network (BiFPN) for compound feature fusion, serving as
a key Severstal benchmark model.

Key modifications over standard YOLOv8:
  - BiFPN neck: Weighted bi-directional feature fusion
  - Enhanced backbone with additional skip connections
  - Optimized for multi-scale defect detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ConvBNSiLU(nn.Module):
    """Conv + BatchNorm + SiLU activation."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2fBlock(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_ch, out_ch, n=1, shortcut=True):
        super().__init__()
        self.mid = out_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, 2 * self.mid, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.mid, out_ch, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                ConvBNSiLU(self.mid, self.mid, 1),
                ConvBNSiLU(self.mid, self.mid, 3),
            ) for _ in range(n)
        )
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv1(x)
        chunks = list(y.chunk(2, dim=1))
        for block in self.blocks:
            out = block(chunks[-1])
            if self.shortcut and out.shape == chunks[-1].shape:
                out = out + chunks[-1]
            chunks.append(out)
        return self.cv2(torch.cat(chunks, dim=1))


class SPPFBlock(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        mid = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, mid, 1)
        self.cv2 = ConvBNSiLU(mid * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class BiFPNLayer(nn.Module):
    """
    Single BiFPN layer with weighted bi-directional feature fusion.
    
    Implements the fast normalized fusion:
      O = sum(w_i * I_i) / (sum(w_i) + eps)
    where w_i are learned weights.
    """

    def __init__(self, channels: List[int], out_channel: int):
        """
        Args:
            channels: Input channel sizes for each feature level [P3, P4, P5]
            out_channel: Unified output channel size
        """
        super().__init__()
        self.num_levels = len(channels)

        # Channel alignment (1x1 conv to unify channels)
        self.align = nn.ModuleList([
            ConvBNSiLU(ch, out_channel, 1) if ch != out_channel else nn.Identity()
            for ch in channels
        ])

        # Top-down fusion weights (learnable)
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32))
            for _ in range(self.num_levels - 1)
        ])

        # Bottom-up fusion weights (learnable)
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i > 0 else 2, dtype=torch.float32))
            for i in range(self.num_levels - 1)
        ])

        # Top-down convolutions
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, groups=out_channel, bias=False),
                nn.Conv2d(out_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.SiLU(inplace=True),
            )
            for _ in range(self.num_levels - 1)
        ])

        # Bottom-up convolutions
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, groups=out_channel, bias=False),
                nn.Conv2d(out_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.SiLU(inplace=True),
            )
            for _ in range(self.num_levels - 1)
        ])

        self.eps = 1e-4

    def _fuse(self, features: List[torch.Tensor], weights: nn.Parameter) -> torch.Tensor:
        """Fast normalized weighted fusion."""
        w = F.relu(weights)
        w = w / (w.sum() + self.eps)

        # Resize all features to match first feature's spatial size
        target_size = features[0].shape[2:]
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='nearest')
            fused = fused + w[i] * feat

        return fused

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: [P3, P4, P5] feature maps (low-res to high-res stride)
        
        Returns:
            Fused features at same levels
        """
        # Align channels
        aligned = [self.align[i](f) for i, f in enumerate(features)]

        # Top-down path: P5 -> P4 -> P3
        td_features = [None] * self.num_levels
        td_features[-1] = aligned[-1]  # P5 unchanged

        for i in range(self.num_levels - 2, -1, -1):
            fused = self._fuse([aligned[i], td_features[i + 1]], self.td_weights[i])
            td_features[i] = self.td_convs[i](fused)

        # Bottom-up path: P3 -> P4 -> P5
        out_features = [None] * self.num_levels
        out_features[0] = td_features[0]  # P3 from top-down

        for i in range(1, self.num_levels):
            if i == 1:
                inputs = [td_features[i], out_features[i - 1]]
            else:
                inputs = [aligned[i], td_features[i], out_features[i - 1]]
            fused = self._fuse(inputs, self.bu_weights[i - 1])
            out_features[i] = self.bu_convs[i - 1](fused)

        return out_features


class BiFPN(nn.Module):
    """
    Stacked BiFPN layers for multi-scale feature fusion.
    """

    def __init__(self, channels: List[int], out_channel: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = channels if i == 0 else [out_channel] * len(channels)
            self.layers.append(BiFPNLayer(in_channels, out_channel))

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        for layer in self.layers:
            features = layer(features)
        return features


class EBYOLOv8Head(nn.Module):
    """Detection head for EB-YOLOv8."""

    def __init__(self, in_ch: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # cls + bbox(4) + obj

        self.cls_branch = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch, 3),
            ConvBNSiLU(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.reg_branch = nn.Sequential(
            ConvBNSiLU(in_ch, in_ch, 3),
            ConvBNSiLU(in_ch, in_ch, 3),
            nn.Conv2d(in_ch, 5, 1),  # 4 bbox + 1 obj
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out = self.cls_branch(x)
        reg_out = self.reg_branch(x)
        return torch.cat([reg_out, cls_out], dim=1)  # [B, 5+C, H, W]


class EBYOLOv8(nn.Module):
    """
    EB-YOLOv8: Enhanced Backbone YOLOv8 with BiFPN.
    
    Architecture:
      Backbone: YOLOv8-s CSP backbone with enhanced skip connections
      Neck: BiFPN (3-layer stacked bi-directional feature pyramid)
      Head: Decoupled cls/reg detection head
    """

    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # Backbone channel config (YOLOv8-s)
        bc = [64, 128, 256, 512]

        # ====== Backbone ======
        self.stem = ConvBNSiLU(3, bc[0], 3, 2)
        self.stage1 = nn.Sequential(ConvBNSiLU(bc[0], bc[1], 3, 2), C2fBlock(bc[1], bc[1], n=3))
        self.stage2 = nn.Sequential(ConvBNSiLU(bc[1], bc[2], 3, 2), C2fBlock(bc[2], bc[2], n=6))
        self.stage3 = nn.Sequential(ConvBNSiLU(bc[2], bc[3], 3, 2), C2fBlock(bc[3], bc[3], n=6))
        self.stage4 = nn.Sequential(
            ConvBNSiLU(bc[3], bc[3], 3, 2),
            C2fBlock(bc[3], bc[3], n=3),
            SPPFBlock(bc[3], bc[3]),
        )

        # ====== Enhanced skip connections ======
        self.skip_p3 = ConvBNSiLU(bc[2], bc[2], 1)
        self.skip_p4 = ConvBNSiLU(bc[3], bc[3], 1)

        # ====== BiFPN Neck ======
        bifpn_channels = [bc[2], bc[3], bc[3]]  # P3, P4, P5
        self.bifpn = BiFPN(bifpn_channels, out_channel=256, num_layers=3)

        # ====== Detection Heads ======
        self.heads = nn.ModuleList([
            EBYOLOv8Head(256, num_classes) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        # Enhanced skip connections
        p3 = self.skip_p3(p3)
        p4 = self.skip_p4(p4)

        # BiFPN
        features = self.bifpn([p3, p4, p5])

        # Detection heads
        outputs = [head(feat) for head, feat in zip(self.heads, features)]
        return outputs

    def compute_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss (same structure as YOLO-MFD).
        """
        device = predictions[0].device
        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        batch_size = predictions[0].shape[0]

        for pred in predictions:
            B, C, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()

            box_pred = pred[..., :4]
            obj_pred = pred[..., 4:5]
            cls_pred = pred[..., 5:]

            obj_target = torch.zeros(B, H, W, 1, device=device)
            box_target = torch.zeros(B, H, W, 4, device=device)
            cls_target = torch.zeros(B, H, W, self.num_classes, device=device)

            for b in range(B):
                if b >= len(targets) or targets[b].shape[0] == 0:
                    continue
                for t in targets[b]:
                    cls_id = int(t[0])
                    cx, cy, bw, bh = t[1], t[2], t[3], t[4]
                    gi = min(max(int(cx * W), 0), W - 1)
                    gj = min(max(int(cy * H), 0), H - 1)
                    obj_target[b, gj, gi, 0] = 1.0
                    box_target[b, gj, gi] = torch.tensor([cx, cy, bw, bh], device=device)
                    cls_target[b, gj, gi, cls_id] = 1.0

            obj_mask = obj_target.squeeze(-1)
            pos_count = obj_mask.sum().clamp(min=1)

            obj_loss = F.binary_cross_entropy_with_logits(
                obj_pred.squeeze(-1), obj_mask, reduction='sum'
            ) / pos_count

            pos_mask = obj_mask.bool()
            if pos_count > 0:
                box_loss = F.mse_loss(
                    box_pred[pos_mask], box_target[pos_mask], reduction='sum'
                ) / pos_count
                cls_loss = F.binary_cross_entropy_with_logits(
                    cls_pred[pos_mask], cls_target[pos_mask], reduction='sum'
                ) / pos_count
            else:
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)

            total_box_loss = total_box_loss + box_loss
            total_obj_loss = total_obj_loss + obj_loss
            total_cls_loss = total_cls_loss + cls_loss

        total = total_box_loss * 5.0 + total_obj_loss * 1.0 + total_cls_loss * 0.5
        return {'total': total, 'box': total_box_loss, 'obj': total_obj_loss, 'cls': total_cls_loss}

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> List[Dict]:
        """Run inference with NMS."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            all_boxes, all_scores, all_labels = [], [], []

            for pred in outputs:
                B, C, H, W = pred.shape
                p = pred[b].permute(1, 2, 0).contiguous()

                obj_scores = torch.sigmoid(p[..., 4])
                cls_scores = torch.sigmoid(p[..., 5:])

                gy, gx = torch.meshgrid(
                    torch.arange(H, device=x.device, dtype=torch.float32),
                    torch.arange(W, device=x.device, dtype=torch.float32),
                    indexing='ij',
                )

                cx = (torch.sigmoid(p[..., 0]) + gx) / W
                cy = (torch.sigmoid(p[..., 1]) + gy) / H
                bw = torch.sigmoid(p[..., 2])
                bh = torch.sigmoid(p[..., 3])

                combined = obj_scores.unsqueeze(-1) * cls_scores
                max_scores, max_labels = combined.max(dim=-1)
                mask = max_scores > conf_threshold

                if mask.sum() == 0:
                    continue

                img_h, img_w = x.shape[2], x.shape[3]
                fcx, fcy, fbw, fbh = cx[mask], cy[mask], bw[mask], bh[mask]
                x1 = (fcx - fbw / 2) * img_w
                y1 = (fcy - fbh / 2) * img_h
                x2 = (fcx + fbw / 2) * img_w
                y2 = (fcy + fbh / 2) * img_h

                all_boxes.append(torch.stack([x1, y1, x2, y2], -1))
                all_scores.append(max_scores[mask])
                all_labels.append(max_labels[mask])

            if all_boxes:
                boxes = torch.cat(all_boxes)
                scores = torch.cat(all_scores)
                labels = torch.cat(all_labels)
                from torchvision.ops import batched_nms
                keep = batched_nms(boxes, scores, labels, iou_threshold)
                results.append({'boxes': boxes[keep], 'scores': scores[keep], 'labels': labels[keep]})
            else:
                results.append({
                    'boxes': torch.zeros((0, 4), device=x.device),
                    'scores': torch.zeros(0, device=x.device),
                    'labels': torch.zeros(0, dtype=torch.long, device=x.device),
                })

        return results
