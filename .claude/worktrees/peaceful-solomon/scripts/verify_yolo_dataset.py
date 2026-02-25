#!/usr/bin/env python3
"""
YOLO 데이터셋 검증 스크립트.

변환된 YOLO 데이터셋(images/, labels/, dataset.yaml)이 정상적으로
결함 bbox를 포함하는지 3단계로 검증한다:

  1. 통계 분석: 이미지/라벨 수, 클래스 분포, 좌표 범위 검증
  2. CSV Cross-check: 원본 train.csv RLE → bbox 재계산 후 .txt 라벨과 비교
  3. 시각화: 샘플 이미지에 YOLO bbox(초록) + CSV bbox(빨강) overlay → PNG 저장

사용법 (Colab):
    python /content/severstal-steel-defect-detection/scripts/verify_yolo_dataset.py \\
        --yolo-dir /content/yolo_datasets \\
        --group baseline_raw \\
        --csv /content/drive/MyDrive/data/Severstal/train.csv \\
        --output-dir /content/drive/MyDrive/data/Severstal/casda/yolo_verify \\
        --num-samples 20 \\
        --split train
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.rle_utils import rle_decode

logger = logging.getLogger(__name__)

# 클래스별 시각화 색상 (BGR)
CLASS_COLORS = {
    0: (255, 100, 100),   # Class1 — 파랑
    1: (100, 200, 100),   # Class2 — 초록
    2: (0, 220, 255),     # Class3 — 노랑
    3: (80, 80, 255),     # Class4 — 빨강
}
CLASS_NAMES = {0: "Class1", 1: "Class2", 2: "Class3", 3: "Class4"}


# ============================================================================
# 1. 통계 분석
# ============================================================================

def parse_label_file(label_path: Path):
    """
    YOLO 라벨 파일 파싱.

    Returns:
        list of (class_id, cx, cy, w, h) — 모두 float
        빈 파일이면 빈 리스트
    """
    bboxes = []
    if not label_path.exists():
        return bboxes
    text = label_path.read_text().strip()
    if not text:
        return bboxes
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        bboxes.append((cls_id, cx, cy, w, h))
    return bboxes


def print_dataset_stats(yolo_base: Path, split: str):
    """전체 split에 대한 통계 출력. 파싱된 bbox 리스트도 반환."""
    img_dir = yolo_base / "images" / split
    lbl_dir = yolo_base / "labels" / split

    if not img_dir.exists():
        print(f"  [ERROR] images/{split} 디렉토리가 없습니다: {img_dir}")
        return {}, 0

    img_files = sorted([f for f in img_dir.iterdir() if f.suffix in (".jpg", ".png", ".jpeg")])
    num_images = len(img_files)

    class_counts = defaultdict(int)
    empty_count = 0
    total_bboxes = 0
    coord_violations = 0
    class_violations = 0
    all_labels = {}  # stem -> list of (cls, cx, cy, w, h)

    for img_f in img_files:
        stem = img_f.stem
        lbl_path = lbl_dir / (stem + ".txt")
        bboxes = parse_label_file(lbl_path)
        all_labels[stem] = bboxes

        if len(bboxes) == 0:
            empty_count += 1
            continue

        for cls_id, cx, cy, w, h in bboxes:
            total_bboxes += 1
            class_counts[cls_id] += 1
            # 좌표 범위 검증
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                coord_violations += 1
            # 클래스 범위 검증
            if cls_id < 0 or cls_id > 3:
                class_violations += 1

    num_labels = sum(1 for f in lbl_dir.iterdir() if f.suffix == ".txt") if lbl_dir.exists() else 0

    print(f"\n{'=' * 60}")
    print(f"  YOLO Dataset Statistics: {yolo_base.name}/{split}")
    print(f"{'=' * 60}")
    print(f"  Images: {num_images},  Label files: {num_labels}")
    print(f"  Empty labels (no defects): {empty_count} ({empty_count/max(num_images,1)*100:.1f}%)")
    print(f"  Total bboxes: {total_bboxes}")
    print(f"  Class distribution:")
    for cls_id in sorted(class_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f"Unknown{cls_id}")
        print(f"    Class{cls_id} ({name}): {class_counts[cls_id]} bboxes")
    print(f"  Coord range violations (outside 0~1): {coord_violations} / {total_bboxes}")
    print(f"  Class ID violations (outside 0~3):    {class_violations} / {total_bboxes}")

    return all_labels, num_images


# ============================================================================
# 2. CSV Cross-check
# ============================================================================

def rle_to_bboxes(rle_string: str, shape=(256, 1600)):
    """RLE → contours → normalized bbox list. dataset_yolo.py의 _rle_to_bboxes와 동일 로직."""
    mask = rle_decode(rle_string, shape)
    if mask.sum() == 0:
        return []
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bboxes = []
    h, w = shape
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 16:
            continue
        cx = (x + bw / 2.0) / w
        cy = (y + bh / 2.0) / h
        nw = bw / w
        nh = bh / h
        bboxes.append((cx, cy, nw, nh))
    return bboxes


def cross_check_with_csv(csv_path: str, all_labels: dict, split_img_dir: Path):
    """
    원본 CSV RLE에서 bbox를 재계산하여 YOLO 라벨과 비교.

    Returns:
        csv_bboxes_map: {stem: [(cls_id, cx, cy, w, h), ...]} 재계산 결과
    """
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:
        df[["ImageId", "ClassId"]] = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
        df["ClassId"] = df["ClassId"].astype(int)

    defect_rows = df[df["EncodedPixels"].notna()]

    # CSV에서 bbox 재계산
    csv_annotations = defaultdict(list)  # stem -> [(cls, cx, cy, w, h)]
    for _, row in defect_rows.iterrows():
        img_id = row["ImageId"]
        cls_id = int(row["ClassId"]) - 1  # 0-indexed
        rle = row["EncodedPixels"]
        bboxes = rle_to_bboxes(rle)
        stem = Path(img_id).stem
        for bbox in bboxes:
            csv_annotations[stem].append((cls_id, *bbox))

    # split에 있는 이미지만 비교
    img_stems = set()
    for f in split_img_dir.iterdir():
        if f.suffix in (".jpg", ".png", ".jpeg"):
            img_stems.add(f.stem)

    match_count = 0
    count_mismatch = 0
    coord_mismatch = 0
    total_checked = 0
    mismatch_details = []

    for stem in sorted(img_stems):
        # CASDA 합성 이미지는 cross-check 대상이 아님
        if stem.startswith("casda_"):
            continue

        total_checked += 1
        yolo_bboxes = all_labels.get(stem, [])
        csv_bboxes = csv_annotations.get(stem, [])

        yolo_count = len(yolo_bboxes)
        csv_count = len(csv_bboxes)

        if yolo_count != csv_count:
            count_mismatch += 1
            if len(mismatch_details) < 10:
                mismatch_details.append(
                    f"    {stem}: YOLO={yolo_count} bboxes, CSV={csv_count} bboxes"
                )
            continue

        # bbox 좌표 비교 (순서 무관 매칭, tolerance=0.01 ≈ ±16px@1600w)
        matched = _match_bboxes(yolo_bboxes, csv_bboxes, tol=0.01)
        if matched:
            match_count += 1
        else:
            coord_mismatch += 1
            if len(mismatch_details) < 10:
                mismatch_details.append(
                    f"    {stem}: bbox count OK ({yolo_count}) but coords differ"
                )

    print(f"\n{'=' * 60}")
    print(f"  CSV Cross-check ({total_checked} images)")
    print(f"{'=' * 60}")
    exact_match = match_count
    print(f"  Exact match:         {exact_match} / {total_checked} "
          f"({exact_match/max(total_checked,1)*100:.1f}%)")
    print(f"  Bbox count mismatch: {count_mismatch}")
    print(f"  Coord mismatch:      {coord_mismatch}")
    if mismatch_details:
        print(f"  Mismatch details (up to 10):")
        for d in mismatch_details:
            print(d)
    else:
        print(f"  No mismatches found!")

    return csv_annotations


def _match_bboxes(yolo_bboxes, csv_bboxes, tol=0.01):
    """
    두 bbox 리스트가 tolerance 내에서 1:1 매칭되는지 확인.
    순서 무관, greedy 매칭.
    """
    if len(yolo_bboxes) != len(csv_bboxes):
        return False
    if len(yolo_bboxes) == 0:
        return True

    csv_remaining = list(csv_bboxes)
    for y_cls, y_cx, y_cy, y_w, y_h in yolo_bboxes:
        found = False
        for i, (c_cls, c_cx, c_cy, c_w, c_h) in enumerate(csv_remaining):
            if (y_cls == c_cls
                    and abs(y_cx - c_cx) < tol
                    and abs(y_cy - c_cy) < tol
                    and abs(y_w - c_w) < tol
                    and abs(y_h - c_h) < tol):
                csv_remaining.pop(i)
                found = True
                break
        if not found:
            return False
    return True


# ============================================================================
# 3. 시각화
# ============================================================================

def visualize_samples(
    yolo_base: Path,
    split: str,
    all_labels: dict,
    csv_annotations: dict,
    output_dir: Path,
    num_samples: int = 20,
):
    """
    샘플 이미지에 YOLO bbox(초록 실선) + CSV bbox(빨강 점선) overlay → PNG 저장.
    결함이 있는 이미지를 우선 샘플링.
    """
    img_dir = yolo_base / "images" / split
    save_dir = output_dir / split
    save_dir.mkdir(parents=True, exist_ok=True)

    # 결함 있는 이미지 우선, 그 다음 빈 이미지
    with_defects = [s for s, bbs in all_labels.items() if len(bbs) > 0]
    without_defects = [s for s, bbs in all_labels.items() if len(bbs) == 0]

    # 결함 이미지에서 클래스 다양성 확보를 위해 셔플
    rng = np.random.RandomState(42)
    rng.shuffle(with_defects)

    # 결함 이미지 우선, 부족하면 빈 이미지에서 채움
    samples = with_defects[:num_samples]
    if len(samples) < num_samples:
        samples += without_defects[:num_samples - len(samples)]

    saved = 0
    for stem in samples:
        # 이미지 찾기
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            # symlink가 깨졌을 수 있음 — resolve 시도
            resolved = img_path.resolve()
            img = cv2.imread(str(resolved))
        if img is None:
            continue

        h, w = img.shape[:2]

        # YOLO bbox 그리기 (실선, 두꺼움)
        yolo_bboxes = all_labels.get(stem, [])
        for cls_id, cx, cy, bw, bh in yolo_bboxes:
            color = CLASS_COLORS.get(cls_id, (200, 200, 200))
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"YOLO {CLASS_NAMES.get(cls_id, str(cls_id))}"
            cv2.putText(img, label_text, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # CSV bbox 그리기 (점선 효과 — 얇은 선 + 다른 색조)
        csv_bboxes = csv_annotations.get(stem, [])
        for cls_id, cx, cy, bw, bh in csv_bboxes:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            # 점선 효과: 짧은 세그먼트로 그리기
            _draw_dashed_rect(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1, dash_len=8)
            label_text = f"CSV {CLASS_NAMES.get(cls_id, str(cls_id))}"
            cv2.putText(img, label_text, (x1, min(y2 + 14, h - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 정보 텍스트
        info = f"{stem}  |  YOLO: {len(yolo_bboxes)} bbox  |  CSV: {len(csv_bboxes)} bbox"
        cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        out_path = save_dir / f"{stem}_verify.png"
        cv2.imwrite(str(out_path), img)
        saved += 1

    print(f"\n  Saved {saved} verification images to {save_dir}/")
    return saved


def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_len=8):
    """OpenCV에 점선 사각형 그리기 (4변 각각)."""
    x1, y1 = pt1
    x2, y2 = pt2
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        dist = max(abs(ex - sx), abs(ey - sy))
        if dist == 0:
            continue
        num_dashes = max(dist // (dash_len * 2), 1)
        for i in range(num_dashes):
            t0 = i / num_dashes
            t1 = min((i + 0.5) / num_dashes, 1.0)
            p0 = (int(sx + (ex - sx) * t0), int(sy + (ey - sy) * t0))
            p1 = (int(sx + (ex - sx) * t1), int(sy + (ey - sy) * t1))
            cv2.line(img, p0, p1, color, thickness)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLO 데이터셋 검증: 통계 + CSV cross-check + 시각화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python scripts/verify_yolo_dataset.py \\
      --yolo-dir /content/yolo_datasets \\
      --group baseline_raw \\
      --csv /content/drive/MyDrive/data/Severstal/train.csv \\
      --output-dir /content/drive/MyDrive/data/Severstal/casda/yolo_verify \\
      --num-samples 20 --split train

출력:
  1) 콘솔: 통계 + cross-check 결과
  2) PNG: {output-dir}/{split}/ 에 시각화 이미지 저장
        """,
    )
    parser.add_argument("--yolo-dir", type=str, required=True,
                        help="YOLO 데이터셋 상위 디렉토리")
    parser.add_argument("--group", type=str, default="baseline_raw",
                        help="데이터셋 그룹명 (default: baseline_raw)")
    parser.add_argument("--csv", type=str, default=None,
                        help="원본 train.csv 경로 (없으면 cross-check 생략)")
    parser.add_argument("--output-dir", type=str, default="outputs/yolo_verify",
                        help="시각화 PNG 저장 위치 (default: outputs/yolo_verify)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="시각화할 샘플 수 (default: 20)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"],
                        help="검증 대상 split (default: train)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Resolve paths
    yolo_base = Path(args.yolo_dir) / args.group
    if not yolo_base.exists():
        print(f"[ERROR] YOLO dataset directory not found: {yolo_base}")
        print(f"  Available groups in {args.yolo_dir}:")
        parent = Path(args.yolo_dir)
        if parent.exists():
            for d in sorted(parent.iterdir()):
                if d.is_dir():
                    print(f"    {d.name}/")
        sys.exit(1)

    split_img_dir = yolo_base / "images" / args.split
    if not split_img_dir.exists():
        print(f"[ERROR] images/{args.split}/ not found in {yolo_base}")
        sys.exit(1)

    # ── Step 1: 통계 분석 ──
    all_labels, num_images = print_dataset_stats(yolo_base, args.split)
    if num_images == 0:
        print("[ERROR] No images found. Exiting.")
        sys.exit(1)

    # ── Step 2: CSV Cross-check ──
    csv_annotations = {}
    if args.csv and os.path.exists(args.csv):
        csv_annotations = cross_check_with_csv(args.csv, all_labels, split_img_dir)
    elif args.csv:
        print(f"\n[WARN] CSV not found: {args.csv} — cross-check skipped")
    else:
        print(f"\n[INFO] --csv not specified — cross-check skipped")

    # ── Step 3: 시각화 ──
    output_dir = Path(args.output_dir)
    saved = visualize_samples(
        yolo_base=yolo_base,
        split=args.split,
        all_labels=all_labels,
        csv_annotations=csv_annotations,
        output_dir=output_dir,
        num_samples=args.num_samples,
    )

    print(f"\n{'=' * 60}")
    print(f"  Verification complete.")
    print(f"  Stats: {num_images} images checked")
    print(f"  Visualization: {saved} images saved to {output_dir / args.split}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
