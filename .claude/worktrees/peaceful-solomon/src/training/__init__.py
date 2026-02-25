"""
CASDA Benchmark Training Infrastructure

Provides:
  - Dataset loaders for detection and segmentation (Severstal + CASDA synthetic)
  - Unified trainer for benchmark experiments
  - Evaluation metrics (mAP, Dice, FID, per-class AP)
"""

from src.training.dataset import (
    SeverstalDetectionDataset,
    SeverstalSegmentationDataset,
    CASDASyntheticDataset,
    create_data_loaders,
    split_dataset,
)
from src.training.trainer import BenchmarkTrainer
from src.training.metrics import (
    DetectionEvaluator,
    SegmentationEvaluator,
    FIDCalculator,
    BenchmarkReporter,
)

__all__ = [
    'SeverstalDetectionDataset',
    'SeverstalSegmentationDataset',
    'CASDASyntheticDataset',
    'create_data_loaders',
    'split_dataset',
    'BenchmarkTrainer',
    'DetectionEvaluator',
    'SegmentationEvaluator',
    'FIDCalculator',
    'BenchmarkReporter',
]
