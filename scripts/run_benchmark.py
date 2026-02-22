#!/usr/bin/env python3
"""
CASDA Benchmark Experiment Runner

Orchestrates the full benchmark experiment: 3 models x 4 dataset groups = 12 training runs.

Models:
  - YOLO-MFD (Multi-scale Edge Feature Enhancement)
  - EB-YOLOv8 (BiFPN-based Enhanced Backbone)
  - DeepLabV3+ (Standard Segmentation Baseline)

Dataset Groups:
  - Baseline (Raw): Severstal original only
  - Baseline (Trad): Original + traditional geometric augmentations
  - CASDA-Full: Original + 5,000 CASDA synthetic images
  - CASDA-Pruning: Original + top 2,000 CASDA images by suitability

Usage:
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --models yolo_mfd eb_yolov8
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups baseline_raw casda_pruning
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --fid-only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume outputs/benchmark_results/20260223_143000
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
from pathlib import Path
from datetime import datetime
# Note: use built-in dict (Python 3.9+) instead of typing.Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.yolo_mfd import YOLOMFD
from src.models.eb_yolov8 import EBYOLOv8
from src.models.deeplabv3plus import DeepLabV3Plus
from src.training.dataset import create_data_loaders
from src.training.trainer import BenchmarkTrainer
from src.training.metrics import (
    DetectionEvaluator,
    SegmentationEvaluator,
    FIDCalculator,
    BenchmarkReporter,
)


# ============================================================================
# Model Factory
# ============================================================================

def create_model(model_key: str, model_config: dict, num_classes: int = 4):
    """
    Create a benchmark model from config.
    
    Args:
        model_key: Model identifier (yolo_mfd, eb_yolov8, deeplabv3plus)
        model_config: Model-specific configuration
        num_classes: Number of defect classes
    
    Returns:
        Model instance
    """
    if model_key == "yolo_mfd":
        return YOLOMFD(num_classes=num_classes, variant='s')
    elif model_key == "eb_yolov8":
        return EBYOLOv8(num_classes=num_classes)
    elif model_key == "deeplabv3plus":
        return DeepLabV3Plus(
            num_classes=num_classes,
            backbone=model_config.get('backbone', 'resnet101'),
            pretrained=model_config.get('pretrained', True),
            output_stride=model_config.get('output_stride', 16),
        )
    else:
        raise ValueError(f"Unknown model: {model_key}")


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
    
    Args:
        model_key: Model identifier
        dataset_group: Dataset group key
        config: Full experiment config
        experiment_dir: Output directory for this run
        device: Training device
        resume: If True, skip training when completed checkpoint exists
    
    Returns:
        Test metrics dict
    """
    model_config = config['models'][model_key]
    model_name = model_config['name']
    model_type = model_config['type']  # "detection" or "segmentation"
    group_name = config['dataset_groups'][dataset_group]['name']
    num_classes = config['dataset']['num_classes']

    # Create output directory
    run_dir = experiment_dir / f"{model_key}_{dataset_group}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: check if this run is already completed ----
    checkpoint_dir = run_dir / "checkpoints"
    best_path = checkpoint_dir / f"{model_key}_{dataset_group}_best.pth"
    meta_path = run_dir / "experiment_meta.json"

    if resume and best_path.exists() and meta_path.exists():
        logging.info(f"\n{'#'*70}")
        logging.info(f"# SKIP (completed): {model_name} + {group_name}")
        logging.info(f"# Loading existing results from: {meta_path}")
        logging.info(f"{'#'*70}")

        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get('test_metrics', {})

    # ---- Resume: check if training was interrupted (latest.pth exists, no best.pth or incomplete) ----
    latest_path = checkpoint_dir / f"{model_key}_{dataset_group}_latest.pth"
    resume_checkpoint = None
    if resume and latest_path.exists() and not best_path.exists():
        logging.info(f"Found interrupted checkpoint: {latest_path}")
        resume_checkpoint = str(latest_path)

    logging.info(f"\n{'#'*70}")
    logging.info(f"# Experiment: {model_name} + {group_name}")
    logging.info(f"# Type: {model_type}")
    if resume_checkpoint:
        logging.info(f"# Resuming from: {resume_checkpoint}")
    logging.info(f"{'#'*70}")

    # Create model
    model = create_model(model_key, model_config, num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Create data loaders
    input_size = tuple(model_config.get('input_size', [640, 640]))
    batch_size = model_config['training'].get('batch_size', 16)
    num_workers = config['experiment'].get('num_workers', 4)

    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        dataset_group=dataset_group,
        model_type=model_type,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logging.info(f"Data loaded - Train: {len(train_loader.dataset)}, "
                 f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Create trainer
    # Merge model-level num_classes into training config
    training_config = {**model_config['training'], 'num_classes': num_classes}

    trainer = BenchmarkTrainer(
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

    # Train and evaluate
    test_metrics = trainer.train()

    # Save experiment metadata
    meta = {
        'model': model_name,
        'model_key': model_key,
        'model_type': model_type,
        'dataset_group': group_name,
        'dataset_group_key': dataset_group,
        'num_params': total_params,
        'test_metrics': test_metrics,
        'training_history': trainer.history,
        'timestamp': datetime.now().isoformat(),
    }
    with open(run_dir / "experiment_meta.json", 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    return test_metrics


# ============================================================================
# FID Evaluation
# ============================================================================

def run_fid_evaluation(config: dict, experiment_dir: Path, device: str = 'cuda') -> dict:
    """
    Compute FID scores between real and CASDA-generated images.
    
    Returns:
        dict with overall and per-class FID scores
    """
    logging.info(f"\n{'#'*70}")
    logging.info(f"# FID Score Evaluation")
    logging.info(f"{'#'*70}")

    fid_calc = FIDCalculator(device=device)
    ds_config = config['dataset']
    casda_config = ds_config.get('casda', {})

    # Collect real image paths
    image_dir = Path(ds_config['image_dir'])
    real_images = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    real_images = [str(p) for p in real_images]

    if not real_images:
        logging.warning("No real images found for FID computation")
        return {'fid_overall': float('inf')}

    # Collect CASDA-Full images
    casda_full_dir = Path(casda_config.get('full_dir', 'data/augmented/casda_full'))
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
            real_images[:1000],  # Cap for speed
            casda_images[:1000],
            batch_size=config.get('evaluation', {}).get('fid', {}).get('batch_size', 64),
        )
        results['fid_overall'] = overall_fid
        logging.info(f"FID Score (overall): {overall_fid:.2f}")
    else:
        logging.warning("No CASDA images found for FID computation")
        results['fid_overall'] = float('inf')

    # Save FID results
    fid_path = experiment_dir / "fid_results.json"
    with open(fid_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"FID results saved to: {fid_path}")

    return results


# ============================================================================
# Hypothesis Testing
# ============================================================================

def run_hypothesis_tests(reporter: BenchmarkReporter, config: dict) -> dict:
    """
    Evaluate the 5 hypotheses defined in experiment.md.
    """
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
            # FID-based, handled separately
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'see FID results'}
            continue

        if len(compare) < 2:
            hypotheses_results[h_name] = {'description': h_desc, 'status': 'insufficient data'}
            continue

        group_a, group_b = compare[0], compare[1]
        across_models = h_config.get('across_models', False)

        # Collect metric values
        a_values = []
        b_values = []

        for result in reporter.results:
            ds = result['dataset']
            metrics = result['metrics']

            if h_name == "H4":
                # Focus on specific classes
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

  # Run specific models only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --models yolo_mfd deeplabv3plus

  # Run specific dataset groups only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --groups baseline_raw casda_pruning

  # Run FID evaluation only
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --fid-only

  # Use CPU instead of GPU
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --device cpu

  # Resume: add CASDA runs to existing baseline experiment (skips completed runs)
  python scripts/run_benchmark.py --config configs/benchmark_experiment.yaml --resume outputs/benchmark_results/20260223_143000
        """,
    )
    parser.add_argument('--config', type=str, default='configs/benchmark_experiment.yaml',
                        help='Path to experiment config YAML')
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
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from existing experiment directory. '
                             'Skips already-completed runs (those with best.pth). '
                             'Example: --resume outputs/benchmark_results/20260223_143000')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs for all models (for quick testing)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

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
    if args.resume:
        experiment_dir = Path(args.resume)
        if not experiment_dir.exists():
            print(f"Error: Resume directory not found: {experiment_dir}")
            sys.exit(1)
        print(f"[INFO] Resuming experiment from: {experiment_dir}")
    else:
        output_dir = Path(args.output_dir or config['experiment'].get('output_dir', 'outputs/benchmark_results'))
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

    # Initialize reporter
    reporter = BenchmarkReporter(str(experiment_dir))

    # ====== FID Evaluation ======
    # FID는 CASDA 합성 이미지가 있을 때만 의미 있음
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
                    resume=bool(args.resume),
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
