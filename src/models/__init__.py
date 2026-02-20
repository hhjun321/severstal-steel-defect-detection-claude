"""
CASDA Benchmark Models

Provides detection and segmentation models for the benchmark experiment:
  - YOLO-MFD: Multi-scale Edge Feature Enhancement YOLO
  - EB-YOLOv8: Enhanced Backbone YOLOv8 with BiFPN
  - DeepLabV3+: Standard segmentation baseline
"""

from src.models.yolo_mfd import YOLOMFD
from src.models.eb_yolov8 import EBYOLOv8
from src.models.deeplabv3plus import DeepLabV3Plus

__all__ = ['YOLOMFD', 'EBYOLOv8', 'DeepLabV3Plus']
