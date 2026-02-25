"""
Evaluation Metrics for CASDA Benchmark Experiments

Implements:
  - mAP (Mean Average Precision) at IoU 0.5 for detection
  - Per-class AP for detection
  - Dice Score for segmentation
  - Per-class Dice Score
  - Precision-Recall curves
  - FID (Frechet Inception Distance) for synthetic data quality
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import csv


# ============================================================================
# IoU / Overlap Utilities
# ============================================================================

def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes."""
    n = len(boxes1)
    m = len(boxes2)
    iou_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    return iou_matrix


# ============================================================================
# Detection Metrics (mAP, Per-class AP, PR curves)
# ============================================================================

class DetectionEvaluator:
    """
    Evaluates object detection models using mAP and per-class AP.
    
    Accumulates predictions and ground truths across batches,
    then computes metrics on the full dataset.
    """

    def __init__(self, num_classes: int = 4, iou_threshold: float = 0.5,
                 image_size: Tuple[int, int] = (640, 640)):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.image_size = image_size  # (width, height) for GT denormalization
        self.reset()

    def reset(self):
        """Reset accumulated predictions and ground truths."""
        # Per-class accumulators
        self.all_detections = {c: [] for c in range(self.num_classes)}   # [(score, is_tp)]
        self.num_gt_per_class = {c: 0 for c in range(self.num_classes)}

    def update(
        self,
        predictions: List[Dict],
        targets: List[torch.Tensor],
        image_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Accumulate batch predictions and ground truths.
        
        Args:
            predictions: List of dicts per image with 'boxes', 'scores', 'labels'
            targets: List of [N, 5] tensors (class, cx, cy, w, h) in YOLO format
            image_size: Optional (width, height) override; defaults to self.image_size
        """
        img_w, img_h = image_size if image_size is not None else self.image_size

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes'].cpu().numpy() if isinstance(pred['boxes'], torch.Tensor) else np.array(pred['boxes'])
            pred_scores = pred['scores'].cpu().numpy() if isinstance(pred['scores'], torch.Tensor) else np.array(pred['scores'])
            pred_labels = pred['labels'].cpu().numpy() if isinstance(pred['labels'], torch.Tensor) else np.array(pred['labels'])

            # Convert YOLO target (normalized cxcywh) to xyxy pixel coords
            target_np = target.cpu().numpy() if isinstance(target, torch.Tensor) else np.array(target)
            gt_boxes = []
            gt_labels = []
            for t in target_np:
                cls_id = int(t[0])
                cx, cy, bw, bh = t[1], t[2], t[3], t[4]
                # Denormalize using image dimensions
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                gt_boxes.append([x1, y1, x2, y2])
                gt_labels.append(cls_id)
                self.num_gt_per_class[cls_id] += 1

            gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))
            gt_labels = np.array(gt_labels) if gt_labels else np.array([])

            # Match predictions to ground truths per class
            for cls_id in range(self.num_classes):
                cls_pred_mask = pred_labels == cls_id
                cls_gt_mask = gt_labels == cls_id

                cls_pred_boxes = pred_boxes[cls_pred_mask]
                cls_pred_scores = pred_scores[cls_pred_mask]
                cls_gt_boxes = gt_boxes[cls_gt_mask]

                if len(cls_pred_boxes) == 0:
                    continue

                # Sort by confidence (descending)
                sorted_idx = np.argsort(-cls_pred_scores)
                cls_pred_boxes = cls_pred_boxes[sorted_idx]
                cls_pred_scores = cls_pred_scores[sorted_idx]

                matched_gt = set()
                for i in range(len(cls_pred_boxes)):
                    is_tp = False
                    if len(cls_gt_boxes) > 0:
                        ious = np.array([
                            compute_iou(cls_pred_boxes[i], gt_box)
                            for gt_box in cls_gt_boxes
                        ])
                        best_gt = np.argmax(ious)
                        if ious[best_gt] >= self.iou_threshold and best_gt not in matched_gt:
                            is_tp = True
                            matched_gt.add(best_gt)

                    self.all_detections[cls_id].append((cls_pred_scores[i], is_tp))

    def compute_ap(self, class_id: int) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Average Precision for a single class.
        
        Returns:
            ap: Average Precision value
            precisions: Precision values at each recall level
            recalls: Recall values
        """
        detections = self.all_detections[class_id]
        num_gt = self.num_gt_per_class[class_id]

        if num_gt == 0:
            return 0.0, np.array([]), np.array([])

        if len(detections) == 0:
            return 0.0, np.array([0.0]), np.array([0.0])

        # Sort by score descending
        detections.sort(key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        for i, (score, is_tp) in enumerate(detections):
            if is_tp:
                tp[i] = 1
            else:
                fp[i] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        # Interpolate precision (PASCAL VOC 11-point or all-point)
        # All-point interpolation
        mrec = np.concatenate([[0.0], recalls, [1.0]])
        mpre = np.concatenate([[1.0], precisions, [0.0]])

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Compute AP as area under PR curve
        indices = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

        return ap, precisions, recalls

    def compute_metrics(self) -> Dict:
        """
        Compute all detection metrics.
        
        Returns:
            Dict with 'mAP@0.5', 'class_ap', 'precisions', 'recalls'
        """
        class_aps = {}
        class_precisions = {}
        class_recalls = {}

        for cls_id in range(self.num_classes):
            ap, prec, rec = self.compute_ap(cls_id)
            class_aps[f"Class{cls_id + 1}"] = ap
            class_precisions[f"Class{cls_id + 1}"] = prec.tolist() if len(prec) > 0 else []
            class_recalls[f"Class{cls_id + 1}"] = rec.tolist() if len(rec) > 0 else []

        mAP = np.mean(list(class_aps.values())) if class_aps else 0.0

        return {
            'mAP@0.5': float(mAP),
            'class_ap': class_aps,
            'precisions': class_precisions,
            'recalls': class_recalls,
            'num_gt_per_class': {f"Class{k+1}": v for k, v in self.num_gt_per_class.items()},
        }


# ============================================================================
# Segmentation Metrics (Dice Score, IoU)
# ============================================================================

class SegmentationEvaluator:
    """
    Evaluates segmentation models using Dice Score and IoU.
    """

    def __init__(self, num_classes: int = 4, threshold: float = 0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated statistics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.pred_sum = np.zeros(self.num_classes)
        self.gt_sum = np.zeros(self.num_classes)
        self.num_samples = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Accumulate batch predictions and targets.
        
        Args:
            predictions: (B, C, H, W) logits or probabilities
            targets: (B, C, H, W) binary ground truth masks
        """
        # Apply sigmoid if logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)

        pred_binary = (predictions > self.threshold).float()

        # Ensure same spatial size
        if pred_binary.shape[2:] != targets.shape[2:]:
            pred_binary = F.interpolate(
                pred_binary, size=targets.shape[2:],
                mode='nearest',
            )

        pred_np = pred_binary.cpu().numpy()
        target_np = targets.cpu().numpy()

        batch_size = pred_np.shape[0]
        self.num_samples += batch_size

        for c in range(self.num_classes):
            p = pred_np[:, c].reshape(batch_size, -1)
            t = target_np[:, c].reshape(batch_size, -1)

            self.intersection[c] += (p * t).sum()
            self.union[c] += ((p + t) > 0).sum()
            self.pred_sum[c] += p.sum()
            self.gt_sum[c] += t.sum()

    def compute_metrics(self) -> Dict:
        """
        Compute all segmentation metrics.
        
        Returns:
            Dict with 'dice_mean', 'iou_mean', 'class_dice', 'class_iou'
        """
        smooth = 1e-6
        class_dice = {}
        class_iou = {}

        for c in range(self.num_classes):
            dice = (2.0 * self.intersection[c] + smooth) / (self.pred_sum[c] + self.gt_sum[c] + smooth)
            iou = (self.intersection[c] + smooth) / (self.union[c] + smooth)
            class_dice[f"Class{c + 1}"] = float(dice)
            class_iou[f"Class{c + 1}"] = float(iou)

        dice_values = list(class_dice.values())
        iou_values = list(class_iou.values())

        return {
            'dice_mean': float(np.mean(dice_values)),
            'iou_mean': float(np.mean(iou_values)),
            'class_dice': class_dice,
            'class_iou': class_iou,
            'num_samples': self.num_samples,
        }


# ============================================================================
# FID Score Computation
# ============================================================================

class FIDCalculator:
    """
    Frechet Inception Distance (FID) for evaluating synthetic data quality.
    
    Uses InceptionV3 features to compute the distance between
    real and generated image distributions.
    
    Requires: torch, torchvision
    """

    def __init__(self, device: str = 'cuda', dims: int = 2048):
        self.device = device
        self.dims = dims
        self._model = None

    def _get_inception_model(self):
        """Lazy-load InceptionV3."""
        if self._model is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            except (ImportError, TypeError):
                from torchvision.models import inception_v3
                model = inception_v3(pretrained=True)

            # Remove final FC to get 2048-dim features
            model.fc = torch.nn.Identity()
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model

    @torch.no_grad()
    def _extract_features(
        self,
        image_paths: List[str],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract InceptionV3 features from image files."""
        import cv2

        model = self._get_inception_model()
        features = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (299, 299))
                img = img.astype(np.float32) / 255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                batch_images.append(img)

            if not batch_images:
                continue

            batch_tensor = torch.from_numpy(np.stack(batch_images)).permute(0, 3, 1, 2).float().to(self.device)
            feat = model(batch_tensor)
            features.append(feat.cpu().numpy())

        if features:
            return np.concatenate(features, axis=0)
        return np.array([]).reshape(0, self.dims)

    @staticmethod
    def _compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of feature set."""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def _calculate_fid(mu1, sigma1, mu2, sigma2) -> float:
        """
        Compute FID between two Gaussians.
        
        FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        """
        from scipy import linalg

        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    def compute_fid(
        self,
        real_image_paths: List[str],
        generated_image_paths: List[str],
        batch_size: int = 64,
    ) -> float:
        """
        Compute FID between real and generated image sets.
        
        Args:
            real_image_paths: Paths to real images
            generated_image_paths: Paths to generated images
            batch_size: Batch size for feature extraction
        
        Returns:
            FID score (lower is better)
        """
        real_features = self._extract_features(real_image_paths, batch_size)
        gen_features = self._extract_features(generated_image_paths, batch_size)

        if len(real_features) < 2 or len(gen_features) < 2:
            return float('inf')

        mu1, sigma1 = self._compute_statistics(real_features)
        mu2, sigma2 = self._compute_statistics(gen_features)

        return self._calculate_fid(mu1, sigma1, mu2, sigma2)

    def compute_fid_per_class(
        self,
        real_image_paths_by_class: Dict[int, List[str]],
        gen_image_paths_by_class: Dict[int, List[str]],
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Compute FID per defect class."""
        results = {}
        for cls_id in sorted(real_image_paths_by_class.keys()):
            real_paths = real_image_paths_by_class.get(cls_id, [])
            gen_paths = gen_image_paths_by_class.get(cls_id, [])
            if len(real_paths) < 2 or len(gen_paths) < 2:
                results[f"Class{cls_id + 1}_FID"] = float('inf')
            else:
                results[f"Class{cls_id + 1}_FID"] = self.compute_fid(
                    real_paths, gen_paths, batch_size
                )
        return results


# ============================================================================
# Results Reporter
# ============================================================================

class BenchmarkReporter:
    """Formats and saves benchmark experiment results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(
        self,
        model_name: str,
        dataset_group: str,
        metrics: Dict,
        training_history: Optional[Dict] = None,
    ):
        """Add a single experiment result."""
        result = {
            'model': model_name,
            'dataset': dataset_group,
            'metrics': metrics,
        }
        if training_history:
            result['training_history'] = training_history
        self.results.append(result)

    def save_results_json(self):
        """Save all results to JSON."""
        path = self.output_dir / "benchmark_results.json"
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to: {path}")

    def save_comparison_csv(self):
        """
        Save comparison table as CSV matching the experiment.md format:

        | Model | Dataset | mAP@0.5 | Dice | Class1_Score | ... | Class4_Score |

        Class{i}_Score = per-class AP (detection) 또는 per-class Dice (segmentation).
        """
        path = self.output_dir / "benchmark_comparison.csv"

        fieldnames = [
            'Model', 'Dataset', 'mAP@0.5', 'Dice',
            'Class1_Score', 'Class2_Score', 'Class3_Score', 'Class4_Score',
        ]

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in self.results:
                metrics = result['metrics']
                row = {
                    'Model': result['model'],
                    'Dataset': result['dataset'],
                    'mAP@0.5': f"{metrics.get('mAP@0.5', 0.0):.4f}",
                    'Dice': f"{metrics.get('dice_mean', 0.0):.4f}",
                }
                # Per-class 메트릭: detection → class_ap, segmentation → class_dice
                is_segmentation = 'dice_mean' in metrics and 'mAP@0.5' not in metrics
                per_class = (
                    metrics.get('class_dice', {})
                    if is_segmentation
                    else metrics.get('class_ap', {})
                )
                for i in range(1, 5):
                    key = f"Class{i}"
                    row[f"Class{i}_Score"] = f"{per_class.get(key, 0.0):.4f}"

                writer.writerow(row)

        print(f"Comparison table saved to: {path}")

    def save_pr_curves(self, metrics: Dict, model_name: str, dataset_group: str):
        """Save precision-recall curves as JSON data (for later plotting)."""
        pr_data = {
            'model': model_name,
            'dataset': dataset_group,
            'precisions': metrics.get('precisions', {}),
            'recalls': metrics.get('recalls', {}),
        }
        path = self.output_dir / f"pr_curve_{model_name}_{dataset_group}.json"
        with open(path, 'w') as f:
            json.dump(pr_data, f, indent=2)

    def print_summary(self):
        """Print formatted summary table."""
        print("\n" + "=" * 100)
        print("CASDA Benchmark Results Summary")
        print("=" * 100)
        print(f"{'Model':<15} {'Dataset':<20} {'mAP@0.5':>8} {'Dice':>8} "
              f"{'C1 AP':>8} {'C2 AP':>8} {'C3 AP':>8} {'C4 AP':>8}")
        print("-" * 100)

        for result in self.results:
            m = result['metrics']
            cap = m.get('class_ap', {}) or m.get('class_dice', {})
            print(f"{result['model']:<15} {result['dataset']:<20} "
                  f"{m.get('mAP@0.5', 0.0):>8.4f} {m.get('dice_mean', 0.0):>8.4f} "
                  f"{cap.get('Class1', 0.0):>8.4f} {cap.get('Class2', 0.0):>8.4f} "
                  f"{cap.get('Class3', 0.0):>8.4f} {cap.get('Class4', 0.0):>8.4f}")

        print("=" * 100)
