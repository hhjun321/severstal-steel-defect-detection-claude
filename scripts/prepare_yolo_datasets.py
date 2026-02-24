#!/usr/bin/env python3
"""
YOLO 포맷 데이터셋 사전 변환 스크립트.

Severstal CSV + 이미지를 ultralytics YOLO 디렉토리 구조로 변환하여 저장한다.
변환된 데이터셋은 run_benchmark.py의 --yolo-dir 옵션으로 참조하여
매 학습마다 반복되는 변환(수 분)을 건너뛸 수 있다.

출력 구조:
    {output-dir}/
      baseline_raw/
        images/{train,val,test}/
        labels/{train,val,test}/
        dataset.yaml
      baseline_trad/
        ...
      casda_full/
        ...
      casda_pruning/
        ...

의존성: pandas, numpy, cv2, pyyaml

사용법 (로컬):
    python scripts/prepare_yolo_datasets.py \\
        --config configs/benchmark_experiment.yaml \\
        --output-dir /path/to/yolo_datasets

사용법 (Colab):
    python /content/severstal-steel-defect-detection/scripts/prepare_yolo_datasets.py \\
        --config /content/severstal-steel-defect-detection/configs/benchmark_experiment.yaml \\
        --data-dir /content/drive/MyDrive/data/Severstal/train_images \\
        --csv /content/drive/MyDrive/data/Severstal/train.csv \\
        --casda-dir /content/drive/MyDrive/data/Severstal/data/augmented_v4_dataset \\
        --split-csv /content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv \\
        --output-dir /content/yolo_datasets

    이후 벤치마크 실행 시:
    python scripts/run_benchmark.py \\
        --config ... --yolo-dir /content/yolo_datasets ...
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataset import get_image_ids_with_defects, split_dataset
from src.training.dataset_yolo import prepare_yolo_dataset, validate_yolo_dataset

# Group alias system (same as run_benchmark.py)
GROUP_ALIASES = {
    "baseline": "baseline_raw",
    "raw":      "baseline_raw",
    "trad":     "baseline_trad",
    "traditional": "baseline_trad",
    "full":     "casda_full",
    "pruning":  "casda_pruning",
    "all":      "__ALL__",
}

# Only groups that use YOLO format (detection models)
YOLO_GROUPS = {"baseline_raw", "baseline_trad", "casda_full", "casda_pruning"}


def resolve_groups(group_args, available_groups):
    """Resolve CLI group names/aliases to canonical config keys."""
    if group_args is None:
        return list(available_groups)

    resolved = []
    for g in group_args:
        key = g.lower()
        if key in GROUP_ALIASES:
            if GROUP_ALIASES[key] == "__ALL__":
                return list(available_groups)
            canonical = GROUP_ALIASES[key]
        elif key in available_groups:
            canonical = key
        else:
            print(f"[ERROR] Unknown dataset group: '{g}'")
            print(f"  Available: {', '.join(available_groups)}")
            print(f"  Aliases: {', '.join(f'{k}→{v}' for k, v in GROUP_ALIASES.items() if v != '__ALL__')}")
            sys.exit(1)

        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def get_split_ids(config):
    """Get train/val/test image ID lists from config."""
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


def prepare_single_group(
    group_key: str,
    config: dict,
    output_dir: Path,
    train_ids: list,
    val_ids: list,
    test_ids: list,
    force: bool = False,
):
    """
    Prepare YOLO dataset for a single dataset group.

    Args:
        group_key: Dataset group key (e.g. 'baseline_raw')
        config: Full experiment config dict
        output_dir: Parent output directory (group subdir will be created)
        train_ids, val_ids, test_ids: Image ID lists
        force: If True, overwrite existing dataset even if valid
    """
    ds_config = config['dataset']
    group_config = config['dataset_groups'][group_key]
    group_dir = output_dir / group_key

    # Check if already valid (skip if not forced)
    if not force:
        existing = validate_yolo_dataset(str(output_dir), dataset_group=group_key)
        if existing:
            logging.info(f"[SKIP] {group_key}: already exists and valid at {group_dir}")
            return existing

    logging.info(f"[CONVERT] {group_key}: converting to YOLO format...")

    # Resolve paths
    raw_img = ds_config['image_dir']
    raw_csv = ds_config['annotation_csv']
    image_dir = raw_img if os.path.isabs(raw_img) else str(PROJECT_ROOT / raw_img)
    annotation_csv = raw_csv if os.path.isabs(raw_csv) else str(PROJECT_ROOT / raw_csv)

    # CASDA settings
    casda_data = group_config.get('casda_data', None)
    casda_dir = None
    casda_mode = None
    casda_config_dict = None

    if casda_data is not None:
        casda_cfg = ds_config.get('casda', {})
        casda_config_dict = casda_cfg
        if casda_data == "full":
            raw_dir = casda_cfg.get('full_dir', 'data/augmented/casda_full')
            casda_dir = raw_dir if os.path.isabs(raw_dir) else str(PROJECT_ROOT / raw_dir)
            casda_mode = "full"
        elif casda_data == "pruning":
            raw_dir = casda_cfg.get('pruning_dir', 'data/augmented/casda_pruning')
            casda_dir = raw_dir if os.path.isabs(raw_dir) else str(PROJECT_ROOT / raw_dir)
            casda_mode = "pruning"

    num_classes = ds_config.get('num_classes', 4)
    class_names = ds_config.get('class_names', [f"Class{i+1}" for i in range(num_classes)])

    start = time.time()
    yaml_path = prepare_yolo_dataset(
        image_dir=image_dir,
        annotation_csv=annotation_csv,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        output_dir=str(group_dir),
        dataset_group=group_key,
        casda_dir=casda_dir,
        casda_mode=casda_mode,
        casda_config=casda_config_dict,
        num_classes=num_classes,
        class_names=class_names,
    )
    elapsed = time.time() - start
    logging.info(f"[DONE] {group_key}: {elapsed:.1f}s → {yaml_path}")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 포맷 데이터셋 사전 변환 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
출력 구조:
  {output-dir}/
    baseline_raw/     images/ labels/ dataset.yaml
    casda_full/       images/ labels/ dataset.yaml
    ...

이후 벤치마크 실행 시:
  python scripts/run_benchmark.py --yolo-dir {output-dir} ...

사용 예시:
  # 모든 그룹 변환
  python scripts/prepare_yolo_datasets.py --config configs/benchmark_experiment.yaml \\
      --output-dir /content/yolo_datasets --groups all

  # baseline만 변환
  python scripts/prepare_yolo_datasets.py --config configs/benchmark_experiment.yaml \\
      --output-dir /content/yolo_datasets --groups baseline

  # 강제 재변환 (기존 데이터 무시)
  python scripts/prepare_yolo_datasets.py --config configs/benchmark_experiment.yaml \\
      --output-dir /content/yolo_datasets --groups baseline --force
        """,
    )
    parser.add_argument('--config', type=str, default='configs/benchmark_experiment.yaml',
                        help='실험 설정 YAML 경로')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='YOLO 데이터셋 출력 디렉토리 (--yolo-dir로 참조할 경로)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='이미지 디렉토리 (config dataset.image_dir 오버라이드)')
    parser.add_argument('--csv', type=str, default=None,
                        help='어노테이션 CSV (config dataset.annotation_csv 오버라이드)')
    parser.add_argument('--casda-dir', type=str, default=None,
                        help='CASDA 데이터 상위 디렉토리 (casda_full/, casda_pruning/ 포함)')
    parser.add_argument('--split-csv', type=str, default=None,
                        help='사전 생성된 분할 CSV 경로')
    parser.add_argument('--groups', nargs='+', default=None,
                        help='변환할 데이터셋 그룹. '
                             'baseline, trad, full, pruning, all 사용 가능. '
                             '미지정 시 전체 그룹 변환')
    parser.add_argument('--force', action='store_true',
                        help='이미 변환된 데이터셋이 있어도 강제 재변환')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (config 오버라이드)')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override paths from CLI
    if args.data_dir:
        config['dataset']['image_dir'] = args.data_dir
        logging.info(f"image_dir → {args.data_dir}")
    if args.csv:
        config['dataset']['annotation_csv'] = args.csv
        logging.info(f"annotation_csv → {args.csv}")
    if args.casda_dir:
        casda_base = args.casda_dir
        if 'casda' not in config['dataset']:
            config['dataset']['casda'] = {}
        config['dataset']['casda']['full_dir'] = os.path.join(casda_base, 'casda_full')
        config['dataset']['casda']['pruning_dir'] = os.path.join(casda_base, 'casda_pruning')
        logging.info(f"casda paths → {casda_base}/casda_full, {casda_base}/casda_pruning")
    if args.split_csv:
        config['dataset']['split_csv'] = os.path.abspath(args.split_csv)
        logging.info(f"split_csv → {config['dataset']['split_csv']}")
    if args.seed is not None:
        config['dataset']['split']['seed'] = args.seed

    # Resolve groups
    available_groups = [
        g for g in config.get('dataset_groups', {}).keys()
        if g in YOLO_GROUPS
    ]
    group_keys = resolve_groups(args.groups, available_groups)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"=" * 60)
    logging.info(f"YOLO Dataset Preparation")
    logging.info(f"  Config: {config_path}")
    logging.info(f"  Output: {output_dir}")
    logging.info(f"  Groups: {group_keys}")
    logging.info(f"  Force:  {args.force}")
    logging.info(f"=" * 60)

    # Load split IDs (once, shared across all groups)
    train_ids, val_ids, test_ids = get_split_ids(config)
    logging.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Convert each group
    total_start = time.time()
    results = {}

    for group_key in group_keys:
        try:
            yaml_path = prepare_single_group(
                group_key=group_key,
                config=config,
                output_dir=output_dir,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                force=args.force,
            )
            results[group_key] = yaml_path
        except Exception as e:
            logging.error(f"[FAIL] {group_key}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            results[group_key] = None

    total_elapsed = time.time() - total_start

    # Summary
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Conversion complete ({total_elapsed:.1f}s)")
    logging.info(f"{'=' * 60}")
    for group_key, yaml_path in results.items():
        status = "OK" if yaml_path else "FAIL"
        logging.info(f"  [{status}] {group_key}: {yaml_path or 'error'}")

    logging.info(f"\n벤치마크 실행 시 사용법:")
    logging.info(f"  python scripts/run_benchmark.py --yolo-dir {output_dir} ...")

    # Return exit code
    failed = sum(1 for v in results.values() if v is None)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
