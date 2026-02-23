"""
Unified Trainer for CASDA Benchmark Experiments

Supports both detection (YOLO-MFD, EB-YOLOv8) and segmentation (DeepLabV3+)
models with common training loop, logging, early stopping, and evaluation.
Includes AMP (Automatic Mixed Precision) for T4 GPU Tensor Core acceleration.
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

from src.training.metrics import DetectionEvaluator, SegmentationEvaluator


logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 15, mode: str = 'max', min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class BenchmarkTrainer:
    """
    Unified trainer for detection and segmentation benchmark models.
    
    Handles:
      - Training loop with AMP (Mixed Precision)
      - Learning rate scheduling (cosine, poly) with warmup
      - Early stopping with patience tracking
      - Best model checkpointing (slim format)
      - TensorBoard logging
      - Evaluation metric computation
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        model_type: str,  # "detection" or "segmentation"
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        output_dir: str,
        device: str = 'cuda',
        resume_from: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.model_name = model_name
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 300)
        self.lr = train_cfg.get('learning_rate', 0.001)
        self.weight_decay = train_cfg.get('weight_decay', 0.0005)
        self.warmup_epochs = train_cfg.get('warmup_epochs', 10)
        self.patience = train_cfg.get('early_stopping_patience', 30)

        # AMP (Automatic Mixed Precision)
        self.use_amp = train_cfg.get('use_amp', True) and device == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("AMP (Mixed Precision) enabled — using FP16 Tensor Cores")

        # Optimizer
        optimizer_name = train_cfg.get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.937,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Learning rate scheduler
        scheduler_name = train_cfg.get('lr_scheduler', 'cosine')
        if scheduler_name == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
            )
        elif scheduler_name == 'poly':
            poly_power = train_cfg.get('poly_power', 0.9)
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: (1 - epoch / self.epochs) ** poly_power,
            )
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

        # Early stopping
        metric_mode = 'max' if model_type == 'detection' else 'max'
        self.early_stopping = EarlyStopping(patience=self.patience, mode=metric_mode)

        # Evaluators
        num_classes = config.get('num_classes', 4)
        if model_type == "detection":
            self.evaluator = DetectionEvaluator(
                num_classes=num_classes,
                iou_threshold=train_cfg.get('iou_threshold', 0.5),
            )
        else:
            self.evaluator = SegmentationEvaluator(
                num_classes=num_classes,
                threshold=config.get('threshold', 0.5),
            )

        # TensorBoard
        self.writer = None
        if HAS_TENSORBOARD:
            log_dir = self.output_dir / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(log_dir))

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],  # mAP or dice
            'learning_rate': [],
            'best_epoch': 0,
            'best_metric': 0.0,
        }

        # Resume from checkpoint if specified
        self.start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            self._load_checkpoint(resume_from)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1

        # Restore AMP scaler state if available
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore history
        saved_history = checkpoint.get('history', {})
        if saved_history:
            self.history = saved_history

        # Advance scheduler to correct position
        for _ in range(self.start_epoch):
            if self.start_epoch > self.warmup_epochs:
                self.scheduler.step()

        logger.info(f"Resumed from epoch {self.start_epoch} "
                     f"(best metric: {self.history.get('best_metric', 0.0):.4f})")

    def _warmup_lr(self, epoch: int, batch_idx: int, num_batches: int):
        """Linear warmup learning rate."""
        if epoch < self.warmup_epochs:
            progress = (epoch * num_batches + batch_idx) / (self.warmup_epochs * num_batches)
            lr = self.lr * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch with AMP."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            self._warmup_lr(epoch, batch_idx, num_batches)

            images = batch['image'].to(self.device)

            # Forward pass with AMP autocast
            with autocast(enabled=self.use_amp):
                if self.model_type == "detection":
                    targets = batch['labels']
                    targets = [t.to(self.device) for t in targets]
                    predictions = self.model(images)
                    loss_dict = self.model.compute_loss(predictions, targets)
                    loss = loss_dict['total']
                else:
                    masks = batch['mask'].to(self.device)
                    predictions = self.model(images)
                    loss_dict = self.model.compute_loss(predictions, masks)
                    loss = loss_dict['total']

            # Backward pass with AMP scaler
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # Logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"  Epoch [{epoch+1}/{self.epochs}] Batch [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, loader: DataLoader = None) -> Tuple[float, Dict]:
        """
        Validate on given loader with AMP inference.
        
        Returns:
            val_loss, metrics_dict
        """
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        self.evaluator.reset()
        total_loss = 0.0
        num_batches = len(loader)

        for batch in loader:
            images = batch['image'].to(self.device)

            with autocast(enabled=self.use_amp):
                if self.model_type == "detection":
                    targets = batch['labels']
                    targets_device = [t.to(self.device) for t in targets]

                    predictions_raw = self.model(images)
                    loss_dict = self.model.compute_loss(predictions_raw, targets_device)
                    total_loss += loss_dict['total'].item()

                    # Get NMS predictions for evaluation
                    conf_thresh = self.config.get('training', {}).get('conf_threshold', 0.25)
                    iou_thresh = self.config.get('training', {}).get('iou_threshold', 0.5)
                    pred_results = self.model.predict(images, conf_thresh, iou_thresh)
                    self.evaluator.update(pred_results, targets)
                else:
                    masks = batch['mask'].to(self.device)
                    predictions = self.model(images)
                    loss_dict = self.model.compute_loss(predictions, masks)
                    total_loss += loss_dict['total'].item()
                    self.evaluator.update(predictions, masks)

        avg_loss = total_loss / max(num_batches, 1)
        metrics = self.evaluator.compute_metrics()
        return avg_loss, metrics

    def _get_primary_metric(self, metrics: Dict) -> float:
        """Get the primary metric for comparison."""
        if self.model_type == "detection":
            return metrics.get('mAP@0.5', 0.0)
        else:
            return metrics.get('dice_mean', 0.0)

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
        }
        # Save AMP scaler state for resume
        if self.use_amp:
            state['scaler_state_dict'] = self.scaler.state_dict()

        # Save latest
        path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        torch.save(state, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
            torch.save(state, best_path)
            logger.info(f"  Saved best model checkpoint (epoch {epoch+1})")

    def train(self) -> Dict:
        """
        Run full training loop with AMP and early stopping tracking.
        
        Returns:
            Final test metrics dict with early stopping metadata.
        """
        metric_name = 'mAP@0.5' if self.model_type == 'detection' else 'Dice'

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {self.model_name} ({self.model_type})")
        logger.info(f"Epochs: {self.epochs}, LR: {self.lr}, Device: {self.device}")
        logger.info(f"AMP: {self.use_amp}, Patience: {self.patience}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}, "
                     f"Val samples: {len(self.val_loader.dataset)}")
        if self.start_epoch > 0:
            logger.info(f"Resuming from epoch {self.start_epoch}")
        logger.info(f"{'='*60}")

        best_metric = self.history.get('best_metric', 0.0)
        start_time = time.time()
        early_stopped = False
        stopped_epoch = self.epochs  # default: ran all epochs

        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate()

            # Scheduler step (after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            # Get primary metric
            primary_metric = self._get_primary_metric(val_metrics)

            # Log
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(primary_metric)
            self.history['learning_rate'].append(current_lr)

            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch [{epoch+1}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val {metric_name}: {primary_metric:.4f} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar(f'Metric/{metric_name}', primary_metric, epoch)
                self.writer.add_scalar('LR', current_lr, epoch)

            # Save checkpoint
            is_best = primary_metric > best_metric
            if is_best:
                best_metric = primary_metric
                self.history['best_epoch'] = epoch
                self.history['best_metric'] = best_metric
            self._save_checkpoint(epoch, val_metrics, is_best)

            # Early stopping
            if self.early_stopping(primary_metric):
                logger.info(f"Early stopping triggered at epoch {epoch+1} "
                            f"(no improvement for {self.patience} epochs)")
                early_stopped = True
                stopped_epoch = epoch + 1
                break

        total_time = time.time() - start_time

        if not early_stopped:
            stopped_epoch = self.epochs

        logger.info(f"\nTraining completed in {total_time:.1f}s")
        logger.info(f"Best {metric_name}: {best_metric:.4f} at epoch {self.history['best_epoch']+1}")
        if early_stopped:
            logger.info(f"Early stopped at epoch {stopped_epoch}/{self.epochs}")
        else:
            logger.info(f"Ran full {self.epochs} epochs (no early stopping)")

        # Load best model and evaluate on test set
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pth"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for test evaluation")

        test_loss, test_metrics = self.validate(self.test_loader)
        test_primary = self._get_primary_metric(test_metrics)
        logger.info(f"\nTest {metric_name}: {test_primary:.4f}")

        # Record early stopping info in history
        self.history['early_stopped'] = early_stopped
        self.history['stopped_epoch'] = stopped_epoch
        self.history['max_epochs'] = self.epochs
        self.history['total_time_seconds'] = round(total_time, 1)
        self.history['use_amp'] = self.use_amp

        # Save training history
        history_path = self.output_dir / f"{self.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Slim down best checkpoint: remove optimizer state (not needed for inference)
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
            slim_state = {
                'epoch': checkpoint['epoch'],
                'model_state_dict': checkpoint['model_state_dict'],
                'metrics': checkpoint['metrics'],
            }
            torch.save(slim_state, best_path)
            logger.info("Saved slim best checkpoint (optimizer state removed)")

        # Remove latest checkpoint (only needed during training for resume)
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pth"
        if latest_path.exists():
            os.remove(latest_path)
            logger.info("Removed latest checkpoint (training complete)")

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        return test_metrics
