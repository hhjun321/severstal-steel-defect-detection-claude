#!/usr/bin/env python3
"""
독립 데이터셋 분할 CSV 생성 스크립트.

Severstal train.csv에서 결함 이미지를 추출하고,
계층적(stratified) 분할을 수행하여 CSV 파일로 저장한다.
이 CSV를 벤치마크 실험에서 참조하면 모든 모델/그룹이
동일한 train/val/test 분할을 사용하게 된다.

출력 CSV 형식:
    ImageId,Split,PrimaryClass
    0002cc93b.jpg,train,1
    0007a71bf.jpg,val,3
    ...

의존성: pandas, scikit-learn (cv2, torch 불필요)

사용법:
    python scripts/create_dataset_split.py \\
        --csv train.csv \\
        --output splits/split_70_15_15_seed42.csv \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \\
        --seed 42

Colab 사용법:
    python /content/severstal-steel-defect-detection/scripts/create_dataset_split.py \\
        --csv /content/drive/MyDrive/data/Severstal/train.csv \\
        --output /content/drive/MyDrive/data/Severstal/casda/splits/split_70_15_15_seed42.csv
"""

import argparse
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


# ============================================================================
# 분할 로직 (dataset.py와 동일, 무거운 의존성 없이 독립 구현)
# ============================================================================

def get_image_ids_with_defects(annotation_csv: str) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    annotation CSV에서 결함이 있는 이미지 ID와 클래스 정보를 추출.
    
    dataset.py의 get_image_ids_with_defects()와 동일한 로직.
    
    Returns:
        image_ids: 정렬된 고유 이미지 ID 리스트
        image_classes: {image_id: [class_id, ...]} (class_id는 1-indexed)
    """
    df = pd.read_csv(annotation_csv)
    if 'ImageId_ClassId' in df.columns:
        df[['ImageId', 'ClassId']] = df['ImageId_ClassId'].str.rsplit('_', n=1, expand=True)
        df['ClassId'] = df['ClassId'].astype(int)

    # 실제 결함이 있는 행만 (EncodedPixels가 NaN이 아닌 것)
    defect_df = df[df['EncodedPixels'].notna()]
    image_classes: Dict[str, List[int]] = {}
    for _, row in defect_df.iterrows():
        img_id = row['ImageId']
        cls_id = int(row['ClassId'])
        if img_id not in image_classes:
            image_classes[img_id] = []
        image_classes[img_id].append(cls_id)

    image_ids = sorted(image_classes.keys())
    return image_ids, image_classes


def split_dataset(
    image_ids: List[str],
    image_classes: Dict[str, List[int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    주요 결함 클래스 기반 계층적 분할.
    
    dataset.py의 split_dataset()와 동일한 로직.
    동일한 image_ids/ratios/seed → 동일한 분할 결과 보장.
    
    Returns:
        train_ids, val_ids, test_ids
    """
    # 각 이미지의 주요(첫 번째) 클래스로 계층화
    primary_classes = [image_classes[iid][0] for iid in image_ids]

    train_ids, temp_ids, train_cls, temp_cls = train_test_split(
        image_ids, primary_classes,
        test_size=1 - train_ratio,
        stratify=primary_classes,
        random_state=seed,
    )
    val_size = val_ratio / (val_ratio + test_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=1 - val_size,
        stratify=temp_cls,
        random_state=seed,
    )
    return train_ids, val_ids, test_ids


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Severstal 데이터셋 분할 CSV 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 70/15/15 분할
  python scripts/create_dataset_split.py --csv train.csv --output splits/split_default.csv

  # 80/10/10 분할 (다른 시드)
  python scripts/create_dataset_split.py --csv train.csv --output splits/split_80_10_10.csv \\
      --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --seed 123
        """,
    )
    parser.add_argument('--csv', required=True,
                        help='Severstal train.csv 경로 (annotation CSV)')
    parser.add_argument('--output', required=True,
                        help='출력 분할 CSV 파일 경로')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='학습 데이터 비율 (기본: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='검증 데이터 비율 (기본: 0.15)')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='테스트 데이터 비율 (기본: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (기본: 42)')
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    """비율 합이 1.0인지 검증."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"분할 비율의 합이 1.0이 아닙니다: "
            f"{train_ratio} + {val_ratio} + {test_ratio} = {total}"
        )
    for name, ratio in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
        if ratio <= 0 or ratio >= 1:
            raise ValueError(f"{name}_ratio는 0과 1 사이여야 합니다: {ratio}")


def format_class_distribution(ids: list, image_classes: dict) -> str:
    """클래스 분포를 포맷팅된 문자열로 반환."""
    counter = Counter(image_classes[iid][0] for iid in ids)
    parts = [f"Class{k}: {counter.get(k, 0)}" for k in sorted(counter.keys())]
    return ", ".join(parts)


def main():
    args = parse_args()

    # 1. 비율 검증
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    logger.info(f"분할 비율: train={args.train_ratio}, val={args.val_ratio}, "
                f"test={args.test_ratio}, seed={args.seed}")

    # 2. 결함 이미지 추출
    if not os.path.exists(args.csv):
        logger.error(f"annotation CSV를 찾을 수 없습니다: {args.csv}")
        sys.exit(1)

    image_ids, image_classes = get_image_ids_with_defects(args.csv)
    logger.info(f"결함 이미지 수: {len(image_ids)}장")

    if len(image_ids) == 0:
        logger.error("결함 이미지가 없습니다. CSV 파일을 확인하세요.")
        sys.exit(1)

    # 3. 계층적 분할 수행
    train_ids, val_ids, test_ids = split_dataset(
        image_ids, image_classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    logger.info(f"분할 결과: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # 4. CSV 구성
    rows = []
    for iid in sorted(train_ids):
        rows.append({
            'ImageId': iid,
            'Split': 'train',
            'PrimaryClass': image_classes[iid][0],
        })
    for iid in sorted(val_ids):
        rows.append({
            'ImageId': iid,
            'Split': 'val',
            'PrimaryClass': image_classes[iid][0],
        })
    for iid in sorted(test_ids):
        rows.append({
            'ImageId': iid,
            'Split': 'test',
            'PrimaryClass': image_classes[iid][0],
        })

    df = pd.DataFrame(rows)

    # 5. 메타데이터 헤더를 CSV 파일 상단에 주석으로 추가
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        # 메타데이터 주석 (pandas에서 읽을 때 comment='#' 사용)
        f.write("# Severstal Dataset Split\n")
        f.write(f"# Created: {datetime.now().isoformat()}\n")
        f.write(f"# Source CSV: {os.path.basename(args.csv)}\n")
        f.write(f"# Ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}\n")
        f.write(f"# Seed: {args.seed}\n")
        f.write(f"# Total: {len(df)}, Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}\n")
        f.write("#\n")
        # CSV 데이터
        df.to_csv(f, index=False, lineterminator='\n')

    logger.info(f"분할 CSV 저장 완료: {args.output}")

    # 6. 통계 출력
    print("\n" + "=" * 60)
    print("데이터셋 분할 요약")
    print("=" * 60)
    print(f"  Annotation CSV : {args.csv}")
    print(f"  출력 파일       : {args.output}")
    print(f"  비율            : {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print(f"  시드            : {args.seed}")
    print(f"  총 결함 이미지  : {len(image_ids)}장")
    print("-" * 60)

    for split_name, split_ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        dist_str = format_class_distribution(split_ids, image_classes)
        pct = len(split_ids) / len(image_ids) * 100
        print(f"  {split_name:5s} : {len(split_ids):5d} ({pct:5.1f}%) - {dist_str}")

    print("=" * 60)

    # 7. 검증: 중복 없음 / 전체 포함 확인
    all_split_ids = set(train_ids) | set(val_ids) | set(test_ids)
    assert len(all_split_ids) == len(image_ids), \
        f"분할 후 총 이미지 수 불일치: {len(all_split_ids)} != {len(image_ids)}"
    assert len(set(train_ids) & set(val_ids)) == 0, "train과 val에 중복 이미지 존재"
    assert len(set(train_ids) & set(test_ids)) == 0, "train과 test에 중복 이미지 존재"
    assert len(set(val_ids) & set(test_ids)) == 0, "val과 test에 중복 이미지 존재"

    logger.info("검증 완료: 중복 없음, 전체 이미지 포함 확인")


if __name__ == '__main__':
    main()
