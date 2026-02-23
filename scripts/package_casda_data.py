#!/usr/bin/env python3
"""
Package ControlNet v4 output into CASDA benchmark format.

Converts:
  - augmented_images_v4/generated/*.png  (generated images)
  - augmented_images_v4/generation_summary.json  (metadata + quality scores)
  - controlnet_dataset_v4/hints/*_hint.png  (hint images → Red channel = mask)

Into:
  - casda_full/images/*.png
  - casda_full/masks/*.png
  - casda_full/metadata.json
  - casda_pruning/images/*.png   (filtered by suitability threshold + top_k)
  - casda_pruning/masks/*.png
  - casda_pruning/metadata.json

Usage (Colab example):
  python scripts/package_casda_data.py \
      --generated-dir /content/drive/MyDrive/data/Severstal/augmented_images_v4/generated \
      --summary-json /content/drive/MyDrive/data/Severstal/augmented_images_v4/generation_summary.json \
      --hint-dir /content/drive/MyDrive/data/Severstal/controlnet_dataset_v4/hints \
      --output-dir /content/drive/MyDrive/data/Severstal/data/augmented \
      --suitability-threshold 0.7 \
      --pruning-top-k 2000
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def parse_class_id_from_filename(filename: str) -> int:
    """
    Extract 0-indexed class_id from filename.
    
    Filename patterns:
      - 6b634bbe9.jpg_class1_region2_gen0.png  → class_id=0
      - 008d0f87b.jpg_class3_region2_gen0.png  → class_id=2
    """
    match = re.search(r"_class(\d+)_", filename)
    if match:
        return int(match.group(1)) - 1  # 1-indexed → 0-indexed
    raise ValueError(f"Cannot parse class_id from filename: {filename}")


def extract_mask_from_hint(hint_path: str, threshold: int = 127) -> Optional[np.ndarray]:
    """
    Extract binary defect mask from hint image's Red channel.
    
    Hint images are 3-channel PNGs where the Red channel encodes
    the defect region with 4-level intensity indicator enhancement.
    Thresholding the Red channel gives a binary mask.
    
    Returns:
        Grayscale binary mask (0 or 255), or None if hint not found.
    """
    hint = cv2.imread(hint_path, cv2.IMREAD_COLOR)
    if hint is None:
        return None
    
    # OpenCV loads as BGR, so Red = channel 2
    red_channel = hint[:, :, 2]
    
    # Threshold to binary
    _, binary_mask = cv2.threshold(red_channel, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask


def build_quality_map(summary: dict) -> dict:
    """
    Build filename → quality_score mapping from generation_summary.json.
    
    The quality section contains sample_scores with per-image quality metrics.
    """
    quality_map = {}
    quality_section = summary.get("quality", {})
    sample_scores = quality_section.get("sample_scores", [])
    
    for entry in sample_scores:
        fname = entry.get("filename", "")
        score = entry.get("quality_score", 0.0)
        quality_map[fname] = score
    
    return quality_map


def build_sample_name_to_result(summary: dict) -> dict:
    """
    Build sample_name → result entry mapping from results[].
    Needed to get hint_path for each sample.
    """
    mapping = {}
    for result in summary.get("results", []):
        sample_name = result.get("sample_name", "")
        mapping[sample_name] = result
    return mapping


def filename_to_sample_name(filename: str) -> str:
    """
    Convert generated filename to sample_name.
    
    Example:
      6b634bbe9.jpg_class1_region2_gen0.png → 6b634bbe9.jpg_class1_region2
    """
    # Remove _genN.png suffix
    match = re.match(r"(.+)_gen\d+\.png$", filename)
    if match:
        return match.group(1)
    # Fallback: strip extension
    return Path(filename).stem


def package_data(
    generated_dir: Path,
    summary_json: Path,
    hint_dir: Path,
    output_dir: Path,
    suitability_threshold: float = 0.7,
    pruning_top_k: int = 2000,
    mask_threshold: int = 127,
) -> None:
    """Main packaging logic."""
    
    # Load generation summary
    print(f"Loading summary: {summary_json}")
    with open(summary_json) as f:
        summary = json.load(f)
    
    total_samples = summary.get("total_samples", 0)
    print(f"Total samples in summary: {total_samples}")
    
    # Build lookup maps
    quality_map = build_quality_map(summary)
    sample_name_map = build_sample_name_to_result(summary)
    
    print(f"Quality scores available for {len(quality_map)} images")
    print(f"Result entries for {len(sample_name_map)} samples")
    
    # Discover generated images
    generated_images = sorted(generated_dir.glob("*.png"))
    print(f"Found {len(generated_images)} generated images in {generated_dir}")
    
    if not generated_images:
        print("ERROR: No generated images found. Check --generated-dir path.")
        sys.exit(1)
    
    # Create output directories
    full_dir = output_dir / "casda_full"
    full_img_dir = full_dir / "images"
    full_mask_dir = full_dir / "masks"
    pruning_dir = output_dir / "casda_pruning"
    pruning_img_dir = pruning_dir / "images"
    pruning_mask_dir = pruning_dir / "masks"
    
    for d in [full_img_dir, full_mask_dir, pruning_img_dir, pruning_mask_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Process each generated image
    all_metadata = []
    skipped_no_hint = 0
    skipped_no_mask = 0
    skipped_class_parse = 0
    
    for img_path in generated_images:
        filename = img_path.name
        sample_name = filename_to_sample_name(filename)
        
        # Parse class_id
        try:
            class_id = parse_class_id_from_filename(filename)
        except ValueError as e:
            print(f"  SKIP (class parse): {e}")
            skipped_class_parse += 1
            continue
        
        # Get quality score
        suitability_score = quality_map.get(filename, 0.0)
        
        # Find hint image for mask extraction
        # Try from results[] first (has exact hint_path), then construct from sample_name
        result_entry = sample_name_map.get(sample_name, {})
        hint_path_str = result_entry.get("hint_path", "")
        
        # The hint_path in JSON is an absolute Colab path, but we use --hint-dir
        # Construct the expected hint filename from sample_name
        hint_filename = f"{sample_name}_hint.png"
        local_hint_path = hint_dir / hint_filename
        
        # Extract mask from hint
        mask = extract_mask_from_hint(str(local_hint_path), threshold=mask_threshold)
        
        if mask is None:
            # Try alternative: maybe hint file uses a slightly different name
            # Check if the hint_path from JSON gives a filename we can use
            if hint_path_str:
                alt_hint_name = Path(hint_path_str).name
                alt_hint_path = hint_dir / alt_hint_name
                mask = extract_mask_from_hint(str(alt_hint_path), threshold=mask_threshold)
            
            if mask is None:
                skipped_no_hint += 1
                # Still include the image but without mask
                # (CASDASyntheticDataset falls back to whole-image bbox for detection)
                mask_rel_path = None
            else:
                skipped_no_hint -= 0  # found with alternate
        
        # Copy image to casda_full/images/
        dest_img = full_img_dir / filename
        shutil.copy2(str(img_path), str(dest_img))
        
        # Save mask if available
        mask_rel_path_str = None
        if mask is not None:
            mask_filename = filename.replace(".png", "_mask.png")
            mask_dest = full_mask_dir / mask_filename
            cv2.imwrite(str(mask_dest), mask)
            mask_rel_path_str = f"masks/{mask_filename}"
            # Verify mask has defect pixels
            defect_pixels = np.count_nonzero(mask)
            if defect_pixels == 0:
                skipped_no_mask += 1
        
        entry = {
            "image_path": f"images/{filename}",
            "class_id": class_id,
            "suitability_score": suitability_score,
        }
        if mask_rel_path_str:
            entry["mask_path"] = mask_rel_path_str
        
        all_metadata.append(entry)
    
    # Save casda_full metadata
    full_meta_path = full_dir / "metadata.json"
    with open(full_meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CASDA-Full packaging complete:")
    print(f"  Total images: {len(all_metadata)}")
    print(f"  With masks: {sum(1 for m in all_metadata if 'mask_path' in m)}")
    print(f"  Without masks: {sum(1 for m in all_metadata if 'mask_path' not in m)}")
    print(f"  Skipped (class parse error): {skipped_class_parse}")
    print(f"  Skipped (no hint found): {skipped_no_hint}")
    print(f"  Masks with zero defect pixels: {skipped_no_mask}")
    print(f"  Output: {full_dir}")
    
    # Class distribution
    class_counts = {}
    for entry in all_metadata:
        cid = entry["class_id"]
        class_counts[cid] = class_counts.get(cid, 0) + 1
    print(f"  Class distribution: {dict(sorted(class_counts.items()))}")
    
    # Quality score stats
    scores = [e["suitability_score"] for e in all_metadata]
    if scores:
        print(f"  Quality scores: min={min(scores):.4f}, max={max(scores):.4f}, "
              f"mean={sum(scores)/len(scores):.4f}")
    
    # Build casda_pruning: filter by threshold, sort descending, take top_k
    pruned = [
        e for e in all_metadata
        if e.get("suitability_score", 0) >= suitability_threshold
    ]
    pruned.sort(key=lambda x: x["suitability_score"], reverse=True)
    pruned = pruned[:pruning_top_k]
    
    # Copy pruned images and masks
    for entry in pruned:
        # Copy image
        src_img = full_img_dir / Path(entry["image_path"]).name
        dst_img = pruning_img_dir / Path(entry["image_path"]).name
        shutil.copy2(str(src_img), str(dst_img))
        
        # Copy mask if available
        if "mask_path" in entry:
            src_mask = full_dir / entry["mask_path"]
            dst_mask = pruning_mask_dir / Path(entry["mask_path"]).name
            if src_mask.exists():
                shutil.copy2(str(src_mask), str(dst_mask))
    
    # Save casda_pruning metadata
    pruning_meta_path = pruning_dir / "metadata.json"
    with open(pruning_meta_path, "w") as f:
        json.dump(pruned, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"CASDA-Pruning packaging complete:")
    print(f"  Threshold: >= {suitability_threshold}")
    print(f"  Top-K: {pruning_top_k}")
    print(f"  After threshold filter: {len([e for e in all_metadata if e.get('suitability_score', 0) >= suitability_threshold])}")
    print(f"  Final count (after top_k): {len(pruned)}")
    
    if pruned:
        pruned_scores = [e["suitability_score"] for e in pruned]
        print(f"  Score range: [{min(pruned_scores):.4f}, {max(pruned_scores):.4f}]")
        
        pruned_class_counts = {}
        for entry in pruned:
            cid = entry["class_id"]
            pruned_class_counts[cid] = pruned_class_counts.get(cid, 0) + 1
        print(f"  Class distribution: {dict(sorted(pruned_class_counts.items()))}")
    
    print(f"  Output: {pruning_dir}")
    
    # Save packaging report
    report = {
        "source": {
            "generated_dir": str(generated_dir),
            "summary_json": str(summary_json),
            "hint_dir": str(hint_dir),
        },
        "casda_full": {
            "total_images": len(all_metadata),
            "with_masks": sum(1 for m in all_metadata if "mask_path" in m),
            "class_distribution": dict(sorted(class_counts.items())),
            "quality_score_stats": {
                "min": round(min(scores), 4) if scores else None,
                "max": round(max(scores), 4) if scores else None,
                "mean": round(sum(scores) / len(scores), 4) if scores else None,
            },
        },
        "casda_pruning": {
            "suitability_threshold": suitability_threshold,
            "pruning_top_k": pruning_top_k,
            "total_images": len(pruned),
            "class_distribution": dict(sorted(
                {e["class_id"]: 0 for e in pruned}.items()
            )),  # will be overwritten below
        },
        "skipped": {
            "class_parse_error": skipped_class_parse,
            "no_hint_found": skipped_no_hint,
            "zero_defect_mask": skipped_no_mask,
        },
    }
    # Fix pruning class distribution
    pruned_cd = {}
    for entry in pruned:
        cid = entry["class_id"]
        pruned_cd[cid] = pruned_cd.get(cid, 0) + 1
    report["casda_pruning"]["class_distribution"] = dict(sorted(pruned_cd.items()))
    
    report_path = output_dir / "packaging_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nPackaging report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Package ControlNet v4 output into CASDA benchmark format"
    )
    parser.add_argument(
        "--generated-dir",
        type=str,
        required=True,
        help="Path to directory with generated images (e.g., augmented_images_v4/generated)",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        required=True,
        help="Path to generation_summary.json",
    )
    parser.add_argument(
        "--hint-dir",
        type=str,
        required=True,
        help="Path to hint images directory (controlnet_dataset_v4/hints)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory (will create casda_full/ and casda_pruning/ inside)",
    )
    parser.add_argument(
        "--suitability-threshold",
        type=float,
        default=0.7,
        help="Minimum quality score for pruning (default: 0.7)",
    )
    parser.add_argument(
        "--pruning-top-k",
        type=int,
        default=2000,
        help="Max number of images for pruning set (default: 2000)",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=127,
        help="Red channel threshold for binary mask extraction (default: 127)",
    )
    
    args = parser.parse_args()
    
    package_data(
        generated_dir=Path(args.generated_dir),
        summary_json=Path(args.summary_json),
        hint_dir=Path(args.hint_dir),
        output_dir=Path(args.output_dir),
        suitability_threshold=args.suitability_threshold,
        pruning_top_k=args.pruning_top_k,
        mask_threshold=args.mask_threshold,
    )


if __name__ == "__main__":
    main()
