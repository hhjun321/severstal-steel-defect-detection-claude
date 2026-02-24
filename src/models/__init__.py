"""
CASDA Benchmark Models

Provides detection and segmentation models for the benchmark experiment:
  - YOLO-MFD: Multi-scale Edge Feature Enhancement YOLO (ultralytics-based)
  - EB-YOLOv8: Enhanced Backbone YOLOv8 (ultralytics-based)
  - DeepLabV3+: Standard segmentation baseline

Detection models are created via factory functions that return ultralytics YOLO
objects ready for .train()/.val()/.predict(). The old YOLOMFD/EBYOLOv8 nn.Module
classes are deprecated and will raise NotImplementedError if instantiated.
"""

from src.models.yolo_mfd import create_yolo_mfd, MEFE, YOLOMFD
from src.models.eb_yolov8 import create_eb_yolov8, EBYOLOv8
from src.models.deeplabv3plus import DeepLabV3Plus

__all__ = [
    # Factory functions (preferred — return ultralytics YOLO objects)
    'create_yolo_mfd',
    'create_eb_yolov8',
    # Segmentation model (standard nn.Module)
    'DeepLabV3Plus',
    # MEFE module (for custom YAML registration)
    'MEFE',
    # Deprecated stubs (kept for backward compat, raise NotImplementedError)
    'YOLOMFD',
    'EBYOLOv8',
]
