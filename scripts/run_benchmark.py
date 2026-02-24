#!/usr/bin/env python3
"""
CASDA Benchmark Experiment Runner

Orchestrates the full benchmark experiment: 3 models x 4 dataset groups = 12 training runs.

Models:
  - YOLO-MFD (Multi-scale Edge Feature Enhancement) — ultralytics native training
  - EB-YOLOv8 (BiFPN-based Enhanced Backbone) — ultralytics native training
  - DeepLabV3+ (Standard Segmentation Baseline) — BenchmarkTrainer

Dataset Groups:
  - Baseline (Raw): Severstal original only
  - Baseline (Trad): Original + traditional geometric augmentations
  - CASDA-Full: Original + all ~2,901 CASDA synthetic images
  - CASDA-Pruning: Original + top CASDA images by suitability

Usage:
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --models yolo_mfd eb_yolov8
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups baseline_raw casda_pruning
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --fid-only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume --output-dir outputs/benchmark_results/20260223_143000
"""

import os
import sys
import argparse
import logging
import time
import json
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deeplabv3plus import DeepLabV3Plus
from src.training.dataset import (
    create_data_loaders,
    get_image_ids_with_defects,
    split_dataset,
)
from src.training.trainer import BenchmarkTrainer
from src.training.ultralytics_trainer import UltralyticsTrainer
from src.training.metrics import (
    DetectionEvaluator,
    SegmentationEvaluator,
    FIDCalculator,
    BenchmarkReporter,
)


# ============================================================================
# Detection Model Set (uses ultralytics)
# ============================================================================
ULTRALYTICS_MODELS = {"yolo_mfd", "eb_yolov8"}


# ============================================================================
# Split ID Resolution (shared by both detection and segmentation paths)
# ============================================================================

def get_split_ids(config: dict) -> tuple:
    """
    Get train/val/test image ID lists from config.
    
    Supports two modes:
      1. Pre-generated split CSV (config['dataset']['split_csv'])
      2. Dynamic split from annotation CSV
    
    Returns:
        (train_ids, val_ids, test_ids)
    """
    ds_config = config['dataset']
    raw_csv = ds_config['annotation_csv']
    annotation_csv = raw_csv if os.path.isabs(raw_csv) else str(PROJECT_ROOT / raw_csv)

    split_csv = ds_config.get('split_csv', None)

    if split_csv is not None and os.path.exists(split_csv):
        split_df = pd.read_csv(split_csv, comment='#')
        train_ids = split_df[split_df['Split'] == 'train']['ImageId'].tolist()
        val_ids = split_df[split_df['Split'] == 'val']['ImageId'].tolist()
        test_ids = split_df[split_df['Split'] == 'test']['ImageId'].tolist()
        logging.info(f"Loaded split from CSV: train={len(train_ids)}, "
                     f"val={len(val_ids)}, test={len(test_ids)}")
    else:
        image_ids, image_classes = get_image_ids_with_defects(annotation_csv)
        train_ids, val_ids, test_ids = split_dataset(
            image_ids, image_classes,
            train_ratio=ds_config['split']['train_ratio'],
            val_ratio=ds_config['split']['val_ratio'],
            test_ratio=ds_config['split']['test_ratio'],
            seed=ds_config['split']['seed'],
        )
        if split_csv is not None:
            logging.warning(f"Split CSV not found: {split_csv} — using dynamic split")

    return train_ids, val_ids, test_ids


# ============================================================================
# Segmentation Model Factory
# ============================================================================

def create_segmentation_model(model_key: str, model_config: dict, num_classes: int = 4):
    """Create a segmentation model."""
    if model_key == "deeplabv3plus":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone=model_config.get('backbone', 'resnet101'),
            pretrained=model_config.get('pretrained', True),
            output_stride=model_config.get('output_stride', 16),
        )
    else:
        raise ValueError(f"Unknown segmentation model: {model_key}")


# ============================================================================
# Single Experiment Run
# ============================================================================

def run_single_experiment(
    model_key: str,
    dataset_group: str,
    config: dict,
    experiment_dir: Path,
    device: str = 'cuda',
    resume: bool = False,
) -> dict:
    """
    Run a single training experiment.
    
    Routes to UltralyticsTrainer for detection models (YOLO-MFD, EB-YOLOv8)
    and BenchmarkTrainer for segmentation models (DeepLabV3+).
    """
    model_config = config['models'][model_key]
    model_name = model_config['name']
    model_type = model_config['type']  # "detection" or "segmentation"
    group_name = config['dataset_groups'][dataset_group]['name']
    group_config = config['dataset_groups'][dataset_group]
    num_classes = config['dataset']['num_classes']

    # Create output directory
    run_dir = experiment_dir / f"{model_key}_{dataset_group}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: check if this run is already completed ----
    meta_path = run_dir / "experiment_meta.json"

    if resume and meta_path.exists():
        # Check for completion marker
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('test_metrics'):
            logging.info(f"\n{'#'*70}")
            logging.info(f"# SKIP (completed): {model_name} + {group_name}")
            logging.info(f"# Loading existing results from: {meta_path}")
            logging.info(f"{'#'*70}")
            return meta['test_metrics']

    logging.info(f"\n{'#'*70}")
    logging.info(f"# Experiment: {model_name} + {group_name}")
    logging.info(f"# Type: {model_type}")
    logging.info(f"{'#'*70}")

    # Get split IDs
    train_ids, val_ids, test_ids = get_split_ids(config)

    # Save split info once per experiment directory
    split_path = experiment_dir / "dataset_split.json"
    if not split_path.exists():
        split_info = {
            'num_train': len(train_ids),
            'num_val': len(val_ids),
            'num_test': len(test_ids),
            'split_config': config['dataset']['split'],
        }
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)

    # ====================================================================
    # Detection models: use UltralyticsTrainer (native ultralytics .train())
    # ====================================================================
    if model_type == "detection" and model_key in ULTRALYTICS_MODELS:
        trainer = UltralyticsTrainer(
            model_key=model_key,
            model_config=model_config,
            dataset_config=config['dataset'],
            group_config=group_config,
            dataset_group=dataset_group,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            output_dir=str(run_dir),
            device=device,
            resume=resume,
        )

        test_metrics = trainer.train()
        history = getattr(trainer, 'history', {})

    # ====================================================================
    # Segmentation models: use BenchmarkTrainer (existing training loop)
    # ====================================================================
    else:
        model = create_segmentation_model(model_key, model_config, num_classes)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Create data loaders (segmentation uses existing dataset.py pipeline)
        input_size = tuple(model_config.get('input_size', [256, 512]))
        batch_size = model_config['training'].get('batch_size', 8)
        num_workers = config['experiment'].get('num_workers', 4)

        train_loader, val_loader, test_loader, split_info_ds = create_data_loaders(
            config=config,
            dataset_group=dataset_group,
            model_type=model_type,
            input_size=input_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        logging.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                     f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

        # Resume checkpoint for segmentation
        resume_checkpoint = None
        if resume:
            checkpoint_dir = run_dir / "checkpoints"
            latest_path = checkpoint_dir / f"{model_key}_{dataset_group}_latest.pth"
            if latest_path.exists():
                resume_checkpoint = str(latest_path)
                logging.info(f"Resuming from: {resume_checkpoint}")

        training_config = {**model_config['training'], 'num_classes': num_classes}
        seg_trainer = BenchmarkTrainer(
            model=model,
            model_name=f"{model_key}_{dataset_group}",
            model_type=model_type,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config,
            output_dir=str(run_dir),
            device=device,
            resume_from=resume_checkpoint,
        )

        test_metrics = seg_trainer.train()
        history = seg_trainer.history

    # Save experiment metadata
    meta = {
        'model': model_name,
        'model_key': model_key,
        'model_type': model_type,
        'dataset_group': group_name,
        'dataset_group_key': dataset_group,
        'test_metrics': test_metrics,
        'best_epoch': history.get('best_epoch', 0),
        'best_metric': history.get('best_metric', 0.0),
        'total_epochs_trained': len(history.get('train_loss', [])) or len(history.get('val_metric', [])),
        'early_stopped': history.get('early_stopped', False),
        'stopped_epoch': history.get('stopped_epoch', 0),
        'max_epochs': history.get('max_epochs', 0),
        'total_time_seconds': history.get('total_time_seconds', 0.0),
        'use_amp': history.get('use_amp', False),
        'training_pipeline': 'ultralytics' if model_key in ULTRALYTICS_MODELS else 'benchmark_trainer',
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "experiment_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return test_metrics


# ============================================================================
# FID Evaluation
# ============================================================================

def run_fid_evaluation(config: dict, experiment_dir: Path, device: str = 'cuda') -> dict:
    """Compute FID scores between real and CASDA-generated images."""
    logging.info(f"\n{'#'*70}")
    logging.info(f"# FID Score Evaluation")
    logging.info(f"{'#'*70}")

    fid_calc = FIDCalculator(device=device)
    ds_config = config['dataset']
    casda_config = ds_config.get('casda', {})

    raw_image_dir = ds_config['image_dir']
    image_dir = Path(raw_image_dir) if os.path.isabs(raw_image_dir) else PROJECT_ROOT / raw_image_dir
    real_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    real_images = [str(p) for p in real_images]

    if not real_images:
        logging.warning("No real images found for FID computation")
        return {'fid_overall': float('inf')}

    raw_casda_dir = casda_config.get('full_dir', 'data/augmented/casda_full')
    casda_full_dir = Path(raw_casda_dir) if os.path.isabs(raw_casda_dir) else PROJECT_ROOT / raw_casda_dir
    casda_images = []
    if casda_full_dir.exists():
        for subdir in [casda_full_dir, casda_full_dir / "images"]:
            if subdir.exists():
                casda_images.extend(sorted(subdir.glob("*.png")))
                casda_images.extend(sorted(subdir.glob("*.jpg")))
    casda_images = [str(p) for p in casda_images]

    results = {}
    if casda_images:
        logging.info(f"Computing FID: {len(real_images)} real vs {len(casda_images)} synthetic")
        overall_fid = fid_calc.compute_fid(
            real_images[:1000],
            casda_images[:1000],
            batch_size=config.get('evaluation', {}).get('fid', {}).get('batch_size', 64),
        )
        results['fid_overall'] = overall_fid
        logging.info(f"FID Score (overall): {overall_fid:.2f}")
    else:
        logging.warning("No CASDA images found for FID computation")
        results['fid_overall'] = float('inf')

    fid_path = experiment_dir / "fid_results.json"
    with open(fid_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"FID results saved to: {fid_path}")

    return results


# ============================================================================
# Hypothesis Testing
# ============================================================================

def run_hypothesis_tests(reporter: BenchmarkReporter, config: dict) -> dict:
    """Evaluate the 5 hypotheses defined in experiment.md."""
    logging.info(f"\n{'#'*70}")
    logging.info(f"# Hypothesis Testing")
    logging.info(f"{'#'*70}")

    hypotheses_results = {}

    for h_config in config.get('reporting', {}).get('hypothesis_tests', []):
        h_name = h_config['name']
        h_desc = h_config['description']
        metric = h_config.get('metric', 'mAP@0.5')
        compare = h_config.get('compare', [])

        logging.info(f"\n{h_name}: {h_desc}")

        if h_name == "H5":
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'see FID results'}
            continue

        if len(compare) < 2:
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'insufficient data'}
            continue

        group_a, group_b = compare[0], compare[1]

        a_values = []
        b_values = []

        for result in reporter.results:
            ds = result['dataset']
            metrics = result['metrics']

            if h_name == "H4":
                focus_classes = h_config.get('focus_classes', [3, 4])
                class_ap = metrics.get('class_ap', {})
                val = np.mean([class_ap.get(f"Class{c}", 0.0) for c in focus_classes])
            else:
                val = metrics.get(metric, metrics.get('mAP@0.5', 0.0))

            if ds == config['dataset_groups'].get(group_a, {}).get('name', group_a):
                a_values.append(val)
            elif ds == config['dataset_groups'].get(group_b, {}).get('name', group_b):
                b_values.append(val)

        if a_values and b_values:
            mean_a = np.mean(a_values)
            mean_b = np.mean(b_values)
            improvement = mean_a - mean_b

            hypotheses_results[h_name] = {
                'description': h_desc,
                f'{group_a}_mean': float(mean_a),
                f'{group_b}_mean': float(mean_b),
                'improvement': float(improvement),
                'supported': improvement > 0,
            }
            status = "SUPPORTED" if improvement > 0 else "NOT SUPPORTED"
            logging.info(f"  {group_a}: {mean_a:.4f} vs {group_b}: {mean_b:.4f} "
                         f"-> {status} (delta={improvement:+.4f})")
        else:
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'no data'}

    return hypotheses_results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CASDA Benchmark Experiment Runner (3 models x 4 datasets = 12 runs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark (all 12 experiments)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml

  # Colab: specify data paths and output directory explicitly
  python scripts/run_benchmark.py \\
      --config /content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml \\
      --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --casda-dir /content/drive/MyDrive/data/Severstal/data/augmented_v4_dataset \\
      --split-csv /content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/benchmark_results \\
      --models yolo_mfd --groups baseline_raw --epochs 10

  # Run specific models only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --models yolo_mfd deeplabv3plus

  # Run specific dataset groups only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups baseline_raw casda_pruning

  # Resume: add CASDA runs to existing experiment (skips completed runs)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume --output-dir outputs/benchmark_results/20260223_143000
        """,
    )
    parser.add_argument('--config', type=str, default='configs/benchmark_experiment.yaml',
                        help='Path to experiment config YAML')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to image directory (overrides config dataset.image_dir)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Path to annotation CSV (overrides config dataset.annotation_csv)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Subset of models to run (e.g., yolo_mfd eb_yolov8)')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='Subset of dataset groups (e.g., baseline_raw casda_pruning)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu). Auto-detected if not specified.')
    parser.add_argument('--fid-only', action='store_true',
                        help='Only compute FID scores, skip model training')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from existing experiment directory (specified by --output-dir). '
                             'Skips already-completed runs (those with experiment_meta.json), '
                             'resumes interrupted runs from last.pt/latest.pth.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs for all models (for quick testing)')
    parser.add_argument('--casda-dir', type=str, default=None,
                        help='Parent dir containing casda_full/ and casda_pruning/ '
                             '(overrides config dataset.casda paths)')
    parser.add_argument('--split-csv', type=str, default=None,
                        help='사전 생성된 분할 CSV 파일 경로 '
                             '(scripts/create_dataset_split.py로 생성). '
                             '지정 시 동적 분할 대신 이 파일의 분할을 사용')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override data paths if specified via CLI
    if args.data_dir:
        config['dataset']['image_dir'] = args.data_dir
        print(f"[INFO] image_dir overridden to: {args.data_dir}")
    if args.csv:
        config['dataset']['annotation_csv'] = args.csv
        print(f"[INFO] annotation_csv overridden to: {args.csv}")
    if args.casda_dir:
        casda_base = args.casda_dir
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['full_dir'] = os.path.join(casda_base, 'casda_full')
        config['dataset']['casda']['pruning_dir'] = os.path.join(casda_base, 'casda_pruning')
        print(f"[INFO] casda paths overridden: {casda_base}/casda_full, {casda_base}/casda_pruning")

    # Override split CSV if specified
    if args.split_csv:
        split_csv_path = os.path.abspath(args.split_csv)
        config['dataset']['split_csv'] = split_csv_path
        print(f"[INFO] split_csv overridden to: {split_csv_path}")

    # Override epochs if specified (for quick testing)
    if args.epochs is not None:
        for model_key in config.get('models', {}):
            config['models'][model_key]['training']['epochs'] = args.epochs
        print(f"[INFO] Epochs overridden to {args.epochs} for all models")

    # Setup
    seed = args.seed or config['experiment'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device or config['experiment'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, falling back to CPU")
        device = 'cpu'

    # Determine experiment directory
    output_dir = Path(args.output_dir or config['experiment'].get('output_dir', 'outputs/benchmark_results'))

    if args.resume:
        experiment_dir = output_dir
        if not experiment_dir.exists():
            print(f"Error: Resume directory not found: {experiment_dir}")
            sys.exit(1)
        print(f"[INFO] Resuming experiment from: {experiment_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = output_dir / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = experiment_dir / "benchmark.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info(f"CASDA Benchmark Experiment")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output: {experiment_dir}")
    logging.info(f"Device: {device}")
    logging.info(f"Seed: {seed}")

    # Save config copy
    with open(experiment_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Determine models and groups to run
    model_keys = args.models or list(config['models'].keys())
    group_keys = args.groups or list(config['dataset_groups'].keys())

    logging.info(f"Models: {model_keys}")
    logging.info(f"Dataset groups: {group_keys}")
    logging.info(f"Total experiments: {len(model_keys) * len(group_keys)}")

    # Log training pipeline info
    for mk in model_keys:
        pipeline = "ultralytics" if mk in ULTRALYTICS_MODELS else "BenchmarkTrainer"
        logging.info(f"  {mk}: {pipeline}")

    # Initialize reporter
    reporter = BenchmarkReporter(str(experiment_dir))

    # ====== FID Evaluation ======
    casda_groups = {'casda_full', 'casda_pruning'}
    has_casda = any(g in casda_groups for g in group_keys)

    if args.fid_only or (has_casda and config.get('evaluation', {}).get('fid', {}).get('compute', True)):
        fid_results = run_fid_evaluation(config, experiment_dir, device)
    elif not has_casda:
        logging.info("Skipping FID evaluation (no CASDA groups selected)")

    if args.fid_only:
        logging.info("FID-only mode complete.")
        return

    # ====== Run Experiments ======
    total_runs = len(model_keys) * len(group_keys)
    run_idx = 0
    start_time = time.time()

    for model_key in model_keys:
        for group_key in group_keys:
            run_idx += 1
            logging.info(f"\n{'='*70}")
            logging.info(f"Run {run_idx}/{total_runs}: {model_key} + {group_key}")
            logging.info(f"{'='*70}")

            try:
                test_metrics = run_single_experiment(
                    model_key=model_key,
                    dataset_group=group_key,
                    config=config,
                    experiment_dir=experiment_dir,
                    device=device,
                    resume=args.resume,
                )

                model_name = config['models'][model_key]['name']
                group_name = config['dataset_groups'][group_key]['name']
                reporter.add_result(model_name, group_name, test_metrics)

            except Exception as e:
                logging.error(f"Experiment failed: {model_key} + {group_key}: {e}")
                import traceback
                logging.error(traceback.format_exc())
                continue

    total_time = time.time() - start_time
    logging.info(f"\nAll experiments completed in {total_time:.1f}s ({total_time/3600:.1f}h)")

    # ====== Results ======
    reporter.print_summary()
    reporter.save_results_json()
    reporter.save_comparison_csv()

    # Save PR curves for detection models
    for result in reporter.results:
        if 'precisions' in result['metrics']:
            reporter.save_pr_curves(
                result['metrics'],
                result['model'].replace(' ', '_'),
                result['dataset'].replace(' ', '_'),
            )

    # ====== Hypothesis Testing ======
    h_results = run_hypothesis_tests(reporter, config)
    h_path = experiment_dir / "hypothesis_results.json"
    with open(h_path, 'w') as f:
        json.dump(h_results, f, indent=2, default=str)

    logging.info(f"\nAll results saved to: {experiment_dir}")
    logging.info("Benchmark experiment complete!")


if __name__ == "__main__":
    main()
