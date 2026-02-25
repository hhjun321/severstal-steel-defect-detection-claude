"""
v5 ROI Extraction Script (256×256 Tile-Based)
==============================================

Severstal 이미지(1600×256px)에서 결함을 중심으로 하는 256×256 타일을 추출합니다.

v4 대비 개선 사항:
  - ROI 크기: 13–63px 소형 패치 → 256×256 타일 (최대 2× 업스케일)
  - 결함 + 주변 배경 컨텍스트가 충분히 포함됨
  - 추출된 타일은 prepare_controlnet_data.py 와 완전 호환되는
    metadata CSV 포맷으로 저장됨

Severstal 이미지 규격:
  - 크기: 1600×256 (가로×세로)
  - 가로 방향으로 결함이 산재해 있음
  - 세로는 이미 256px이므로 타일의 세로는 항상 전체 높이 사용

타일 추출 전략:
  - 결함 bbox의 수평 중심(centroid_x)을 기준으로 256px 창 설정
  - 이미지 경계 초과 시 창을 시프트하여 256px 유지
  - 세로는 항상 [0, 256] (전체 높이)

Usage:
    # 전체 데이터셋 처리
    python scripts/extract_rois.py \\
        --train_csv /content/drive/MyDrive/data/Severstal/train.csv \\
        --image_dir /content/drive/MyDrive/data/Severstal/train_images \\
        --output_dir /content/drive/MyDrive/data/Severstal/roi_patches_v5.1

    # 테스트 (100장만)
    python scripts/extract_rois.py \\
        --train_csv train.csv \\
        --image_dir train_images \\
        --output_dir data/processed/roi_patches_v5.1 \\
        --max_images 100
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.rle_utils import rle_decode

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
IMAGE_H = 256       # Severstal 이미지 높이 (고정)
IMAGE_W = 1600      # Severstal 이미지 너비
TILE_SIZE = 256     # 추출할 타일 크기 (정사각형)


# ──────────────────────────────────────────────────────────────
# Region finding
# ──────────────────────────────────────────────────────────────

def find_regions(mask: np.ndarray):
    """
    이진 마스크에서 연결 요소(Connected Component)를 찾아 각 region의
    bbox, 픽셀 좌표, centroid를 반환합니다.

    Args:
        mask: 이진 마스크 (H, W), uint8 or bool

    Returns:
        List of dicts, 각 dict:
          - 'pixels': (row_indices, col_indices)
          - 'bbox': (x1, y1, x2, y2)  # image 좌표계
          - 'centroid': (cx, cy)
          - 'area': int
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, label_img = cv2.connectedComponents(mask_u8)

    regions = []
    for label in range(1, num_labels):  # 0 = background
        rows, cols = np.where(label_img == label)
        if len(rows) == 0:
            continue

        x1, y1 = int(cols.min()), int(rows.min())
        x2, y2 = int(cols.max()), int(rows.max())
        cx = float(cols.mean())
        cy = float(rows.mean())

        regions.append({
            'pixels': (rows, cols),
            'bbox': (x1, y1, x2, y2),
            'centroid': (cx, cy),
            'area': len(rows),
        })

    # 면적 내림차순 정렬
    regions.sort(key=lambda r: r['area'], reverse=True)
    return regions


# ──────────────────────────────────────────────────────────────
# Shape metrics
# ──────────────────────────────────────────────────────────────

def compute_shape_metrics(region: dict, tile_mask: np.ndarray):
    """
    결함 형태 지표를 계산합니다.

    Args:
        region: find_regions()가 반환한 region dict
        tile_mask: 타일 범위 내의 이진 마스크 (256×256)

    Returns:
        dict: linearity, solidity, extent, aspect_ratio, defect_subtype
    """
    x1, y1, x2, y2 = region['bbox']
    bbox_w = max(x2 - x1, 1)
    bbox_h = max(y2 - y1, 1)
    area = region['area']

    # Aspect ratio (longer side / shorter side)
    aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)

    # Extent = area / bbox_area
    extent = area / (bbox_w * bbox_h)

    # Linearity: 길쭉한 정도. aspect_ratio가 높고 extent가 낮을수록 선형 결함.
    linearity = min(1.0, (aspect_ratio - 1.0) / 9.0 + (1.0 - extent) * 0.5)
    linearity = float(np.clip(linearity, 0.0, 1.0))

    # Solidity: area / convex_hull_area
    solidity = 0.5
    try:
        region_mask = np.zeros(tile_mask.shape, dtype=np.uint8)
        rows, cols = region['pixels']
        region_mask[rows, cols] = 255
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
            if hull_area > 0:
                solidity = float(area / hull_area)
    except Exception:
        pass

    # Defect subtype
    if linearity > 0.65 and aspect_ratio > 3.0:
        defect_subtype = 'linear'
    elif solidity > 0.8 and extent > 0.6:
        defect_subtype = 'compact_blob'
    elif area < 200:
        defect_subtype = 'diffuse'
    else:
        defect_subtype = 'general'

    return {
        'linearity': round(linearity, 6),
        'solidity': round(solidity, 6),
        'extent': round(float(extent), 6),
        'aspect_ratio': round(float(aspect_ratio), 6),
        'defect_subtype': defect_subtype,
    }


# ──────────────────────────────────────────────────────────────
# Tile window computation
# ──────────────────────────────────────────────────────────────

def compute_tile_window(defect_bbox, image_w: int = IMAGE_W,
                        tile_size: int = TILE_SIZE):
    """
    결함 bbox를 기준으로 tile window를 계산합니다.

    결함 bbox의 수평 중심을 기준으로 tile_size 너비의 창을 만든 후,
    이미지 경계를 초과하면 창을 시프트하여 항상 tile_size 너비를 유지합니다.

    세로는 항상 전체 높이 [0, image_h] 를 사용합니다.

    Args:
        defect_bbox: (x1, y1, x2, y2) — image 좌표계
        image_w: 원본 이미지 너비 (기본 1600)
        tile_size: 타일 크기 (기본 256)

    Returns:
        (tx1, tx2): 타일의 수평 범위 [tx1, tx2)
    """
    x1, _, x2, _ = defect_bbox
    cx = (x1 + x2) // 2

    tx1 = cx - tile_size // 2
    tx2 = tx1 + tile_size

    # 이미지 경계 처리
    if tx1 < 0:
        tx1 = 0
        tx2 = tile_size
    if tx2 > image_w:
        tx2 = image_w
        tx1 = image_w - tile_size

    return int(tx1), int(tx2)


# ──────────────────────────────────────────────────────────────
# Main extraction
# ──────────────────────────────────────────────────────────────

def extract_tiles_from_image(image_id: str, image_path: Path,
                             class_masks: dict,
                             output_dir: Path,
                             min_area: int = 50):
    """
    단일 이미지에서 모든 결함 클래스의 타일을 추출합니다.

    Args:
        image_id: 이미지 ID (예: '0002cc93b.jpg')
        image_path: 이미지 파일 경로
        class_masks: {class_id: binary_mask (H,W)} dict
        output_dir: 타일 저장 루트 디렉토리
        min_area: 최소 결함 픽셀 수 (미만이면 무시)

    Returns:
        List of metadata dicts (row per tile)
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"  [WARN] Cannot read image: {image_path}")
        return []

    h, w = img_bgr.shape[:2]
    results = []

    tile_img_dir = output_dir / 'images'
    tile_mask_dir = output_dir / 'masks'
    tile_img_dir.mkdir(parents=True, exist_ok=True)
    tile_mask_dir.mkdir(parents=True, exist_ok=True)

    for class_id, full_mask in class_masks.items():
        regions = find_regions(full_mask)

        for region_id, region in enumerate(regions):
            area = region['area']
            if area < min_area:
                continue

            defect_bbox = region['bbox']   # (x1, y1, x2, y2) in image coords
            centroid = region['centroid']  # (cx, cy)

            # 타일 창 계산
            tx1, tx2 = compute_tile_window(defect_bbox, image_w=w,
                                           tile_size=TILE_SIZE)

            # 타일 이미지 / 마스크 크롭 (전체 높이 × 256px)
            tile_img = img_bgr[:, tx1:tx2]
            tile_mask_full = full_mask[:, tx1:tx2]

            # 타일 내 이 region만의 마스크 (다른 region 제외)
            region_tile_mask = np.zeros_like(tile_mask_full)
            rows, cols = region['pixels']
            tile_cols = cols - tx1
            in_tile = (tile_cols >= 0) & (tile_cols < TILE_SIZE)
            if in_tile.sum() == 0:
                continue
            region_tile_mask[rows[in_tile], tile_cols[in_tile]] = 1

            # 형태 지표 계산 (tile 좌표계 기준)
            shape = compute_shape_metrics(
                {
                    'pixels': (rows[in_tile], tile_cols[in_tile]),
                    'bbox': (
                        int(tile_cols[in_tile].min()),
                        int(rows[in_tile].min()),
                        int(tile_cols[in_tile].max()),
                        int(rows[in_tile].max()),
                    ),
                    'area': int(in_tile.sum()),
                },
                region_tile_mask,
            )

            # 파일명 생성 및 저장
            stem = f"{image_id}_class{class_id}_region{region_id}"
            tile_img_path = tile_img_dir / f"{stem}.png"
            tile_mask_path = tile_mask_dir / f"{stem}.png"

            cv2.imwrite(str(tile_img_path), tile_img)
            cv2.imwrite(str(tile_mask_path),
                        (region_tile_mask * 255).astype(np.uint8))

            # Tile bbox in image coords (roi_bbox)
            roi_bbox = (tx1, 0, tx2, IMAGE_H)

            # suitability_score: area와 coverage 기반 간단 추정
            tile_area = TILE_SIZE * IMAGE_H
            coverage = area / tile_area
            area_score = min(1.0, area / 2000.0)
            suitability = float(np.clip(
                0.4 + 0.4 * area_score + 0.2 * (1 - coverage * 5), 0.4, 0.9
            ))

            results.append({
                'image_id': image_id,
                'class_id': class_id,
                'region_id': region_id,
                'roi_bbox': str(roi_bbox),
                'defect_bbox': str(defect_bbox),
                'centroid': str(centroid),
                'area': int(in_tile.sum()),
                'linearity': shape['linearity'],
                'solidity': shape['solidity'],
                'extent': shape['extent'],
                'aspect_ratio': shape['aspect_ratio'],
                'defect_subtype': shape['defect_subtype'],
                'background_type': 'steel_surface',
                'suitability_score': round(suitability, 4),
                'matching_score': 0.7,
                'continuity_score': 0.5,
                'stability_score': 0.5,
                'recommendation': 'acceptable',
                'prompt': '',
                'roi_image_path': str(tile_img_path),
                'roi_mask_path': str(tile_mask_path),
            })

    return results


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='v5: Severstal 이미지에서 256×256 결함 타일(ROI) 추출'
    )
    parser.add_argument('--train_csv', type=str, default='train.csv',
                        help='RLE 어노테이션이 있는 train.csv 경로')
    parser.add_argument('--image_dir', type=str, default='train_images',
                        help='원본 학습 이미지 디렉토리')
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/roi_patches',
                        help='ROI 타일 출력 디렉토리')
    parser.add_argument('--min_area', type=int, default=50,
                        help='최소 결함 픽셀 수 (미만이면 무시, 기본 50)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='처리할 최대 이미지 수 (테스트용)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='(예약됨, 현재 미사용)')
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    # ── 입력 검증 ──
    if not train_csv.exists():
        print(f"Error: train.csv not found: {train_csv}")
        return
    if not image_dir.exists():
        print(f"Error: image_dir not found: {image_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("v5 ROI Extraction (256×256 Tile-Based)")
    print("=" * 80)
    print(f"train.csv : {train_csv}")
    print(f"image_dir : {image_dir}")
    print(f"output_dir: {output_dir}")
    print(f"min_area  : {args.min_area}")
    print(f"max_images: {args.max_images or 'all'}")
    print("=" * 80)

    # ── train.csv 로드 ──
    print("\n[1/4] train.csv 로드...")
    train_df = pd.read_csv(train_csv)

    # 컬럼명 정규화
    col_map = {}
    for col in train_df.columns:
        if col.lower() in ('imageid', 'image_id'):
            col_map[col] = 'ImageId'
        elif col.lower() in ('classid', 'class_id'):
            col_map[col] = 'ClassId'
        elif col.lower() in ('encodedpixels', 'encoded_pixels'):
            col_map[col] = 'EncodedPixels'
    if col_map:
        train_df = train_df.rename(columns=col_map)

    # 결함이 있는 이미지만 (EncodedPixels가 NaN이 아닌 것)
    defect_df = train_df.dropna(subset=['EncodedPixels'])
    image_ids = sorted(defect_df['ImageId'].unique())

    if args.max_images:
        image_ids = image_ids[:args.max_images]

    print(f"결함 이미지 수: {len(image_ids):,} "
          f"(전체 {len(defect_df['ImageId'].unique()):,})")

    # ── ROI 타일 추출 ──
    print("\n[2/4] ROI 타일 추출 중...")
    all_metadata = []
    missing_images = 0

    for image_id in tqdm(image_ids, desc="이미지 처리"):
        image_path = image_dir / image_id
        if not image_path.exists():
            missing_images += 1
            continue

        # 이 이미지의 모든 클래스 마스크 디코딩
        class_masks = {}
        img_rows = defect_df[defect_df['ImageId'] == image_id]
        for _, row in img_rows.iterrows():
            class_id = int(row['ClassId'])
            rle = row['EncodedPixels']
            if pd.isna(rle) or rle == '':
                continue
            mask = rle_decode(rle, shape=(IMAGE_H, IMAGE_W))
            if mask.sum() > 0:
                class_masks[class_id] = mask

        if not class_masks:
            continue

        tiles = extract_tiles_from_image(
            image_id=image_id,
            image_path=image_path,
            class_masks=class_masks,
            output_dir=output_dir,
            min_area=args.min_area,
        )
        all_metadata.extend(tiles)

    print(f"\n추출된 ROI 수: {len(all_metadata):,}")
    if missing_images:
        print(f"  [WARN] 이미지 파일 없음: {missing_images}장")

    # ── 메타데이터 저장 ──
    print("\n[3/4] 메타데이터 저장...")
    metadata_df = pd.DataFrame(all_metadata)
    metadata_path = output_dir / 'roi_metadata.csv'
    metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')
    print(f"저장 완료: {metadata_path}")

    # ── 통계 출력 및 저장 ──
    print("\n[4/4] 통계 생성...")

    stats_lines = [
        "v5 ROI Extraction Statistics (256×256 Tile-Based)",
        "=" * 60,
        f"처리 이미지 수: {len(image_ids):,}",
        f"추출 ROI 수  : {len(all_metadata):,}",
    ]

    if len(all_metadata) > 0:
        class_counts = metadata_df['class_id'].value_counts().sort_index()
        stats_lines.append("\n클래스별 ROI 수:")
        for cls, cnt in class_counts.items():
            stats_lines.append(f"  Class {cls}: {cnt:,}")

        subtype_counts = metadata_df['defect_subtype'].value_counts()
        stats_lines.append("\n결함 유형별 ROI 수:")
        for st, cnt in subtype_counts.items():
            stats_lines.append(f"  {st}: {cnt:,}")

        stats_lines.append(f"\n결함 면적 통계 (px):")
        stats_lines.append(f"  min   : {metadata_df['area'].min()}")
        stats_lines.append(f"  max   : {metadata_df['area'].max()}")
        stats_lines.append(f"  mean  : {metadata_df['area'].mean():.1f}")
        stats_lines.append(f"  median: {metadata_df['area'].median():.1f}")

        stats_lines.append(f"\nsuitability_score 통계:")
        stats_lines.append(
            f"  min : {metadata_df['suitability_score'].min():.4f}"
        )
        stats_lines.append(
            f"  max : {metadata_df['suitability_score'].max():.4f}"
        )
        stats_lines.append(
            f"  mean: {metadata_df['suitability_score'].mean():.4f}"
        )

    stats_text = "\n".join(stats_lines)
    print("\n" + stats_text)

    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text + "\n")
    print(f"\n통계 저장: {stats_path}")

    print("\n" + "=" * 80)
    print("[완료] v5 ROI 추출 완료!")
    print("=" * 80)
    print(f"\n다음 단계:")
    print(f"  python scripts/prepare_controlnet_data.py \\")
    print(f"      --roi_metadata {metadata_path} \\")
    print(f"      --output_dir <controlnet_dataset_v5.1> \\")
    print(f"      --hint_mode canny")
    print("=" * 80)


if __name__ == '__main__':
    main()
