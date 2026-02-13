"""
ControlNet Model Validation - Multi-Phase Batch Runner
=======================================================

재학습된 ControlNet 모델의 생성 품질을 4단계로 검증합니다.

Phase 1: 기본 생성 검증 (학습 데이터로 모델 작동 확인)
Phase 2: 파라미터 탐색 (guidance_scale, inference_steps 조합 비교)
Phase 3: 미학습 데이터 일반화 검증
Phase 4: 결과 종합 시각화 및 리포트

Usage (Colab):
    # 전체 Phase 실행 (권장: --model_path로 best_model 사용)
    python scripts/run_validation_phases.py \
        --model_path /content/drive/MyDrive/data/Severstal/controlnet_training/best_model \
        --jsonl_path /content/drive/MyDrive/data/Severstal/controlnet_dataset/train.jsonl \
        --roi_metadata_path /content/drive/MyDrive/data/Severstal/roi_patches/roi_metadata.csv \
        --training_log_path /content/drive/MyDrive/data/Severstal/controlnet_training/training_log.json \
        --output_base /content/drive/MyDrive/data/Severstal/test_results

    # 특정 Phase만 실행
    python scripts/run_validation_phases.py \
        --model_path .../best_model \
        --phases 1 2

    # pipeline 사용 (저장 최적화 이전 모델)
    python scripts/run_validation_phases.py \
        --pipeline_path /content/drive/MyDrive/data/Severstal/controlnet_training/pipeline \
        --phases 1

Note:
    저장 최적화(--skip_save_pipeline, --save_fp16) 적용 후에는 pipeline/
    디렉토리가 생성되지 않으므로 --model_path를 사용해야 합니다.
    --model_path는 test_controlnet.py에서 pipeline_reference.json을 참조하여
    base SD model을 자동으로 로드합니다.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1: 기본 생성 검증
# =============================================================================

def run_phase1(args):
    """Phase 1: 학습 데이터로 기본 생성 검증"""
    logger.info("=" * 60)
    logger.info("  Phase 1: Basic Generation Validation")
    logger.info("=" * 60)

    output_dir = Path(args.output_base) / "phase1_basic"

    cmd = [
        sys.executable, str(args.script_dir / "test_controlnet.py"),
    ]

    # model source (model_path 우선, pipeline_path는 fallback)
    cmd += _model_source_args(args)

    cmd += [
        "--jsonl_path", args.jsonl_path,
        "--output_dir", str(output_dir),
        "--num_inference_steps", "30",
        "--guidance_scale", "7.5",
        "--seed", "42",
    ]

    if args.image_root:
        cmd += ["--image_root", args.image_root]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Phase 1 failed:\n{result.stderr}")
        return False
    logger.info(result.stdout)

    # 결과 확인
    summary_path = output_dir / "generation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info(
            f"Phase 1 complete: {summary['generated']}/{summary['total_samples']} "
            f"samples generated"
        )
        return summary["generated"] > 0
    else:
        logger.warning("Phase 1: No summary file generated")
        return False


# =============================================================================
# Phase 2: 파라미터 탐색
# =============================================================================

PHASE2_GUIDANCE_SCALES = [3.0, 5.0, 7.5, 10.0]
PHASE2_INFERENCE_STEPS = [20, 30, 50]
PHASE2_SEEDS = [42, 123, 456, 789]


def run_phase2(args):
    """Phase 2: 다양한 파라미터 조합으로 생성 품질 비교"""
    logger.info("=" * 60)
    logger.info("  Phase 2: Parameter Exploration")
    logger.info("=" * 60)

    # 대표 hint 1개 선택 (첫 번째 학습 샘플)
    representative_hint = _get_first_hint(args.jsonl_path)
    if representative_hint is None:
        logger.error("Phase 2: Cannot find representative hint from JSONL")
        return False

    phase2_results = []

    # guidance_scale 탐색
    logger.info("--- Guidance Scale Exploration ---")
    for gs in PHASE2_GUIDANCE_SCALES:
        output_dir = Path(args.output_base) / f"phase2_gs{gs}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", "30",
            "--guidance_scale", str(gs),
            "--num_images_per_sample", "4",
            "--seed", "42",
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        logger.info(f"  guidance_scale={gs}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  gs={gs} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "guidance_scale", "value": gs,
                "output_dir": str(output_dir),
            })

    # inference_steps 탐색
    logger.info("--- Inference Steps Exploration ---")
    for steps in PHASE2_INFERENCE_STEPS:
        output_dir = Path(args.output_base) / f"phase2_steps{steps}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", str(steps),
            "--guidance_scale", "7.5",
            "--num_images_per_sample", "4",
            "--seed", "42",
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        logger.info(f"  num_inference_steps={steps}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  steps={steps} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "num_inference_steps", "value": steps,
                "output_dir": str(output_dir),
            })

    # seed 다양성 확인
    logger.info("--- Seed Diversity Check ---")
    for seed in PHASE2_SEEDS:
        output_dir = Path(args.output_base) / f"phase2_seed{seed}"

        cmd = [
            sys.executable, str(args.script_dir / "test_controlnet.py"),
        ]
        cmd += _model_source_args(args)

        cmd += [
            "--hint_image", representative_hint["hint_path"],
            "--prompt", representative_hint["prompt"],
            "--output_dir", str(output_dir),
            "--num_inference_steps", "30",
            "--guidance_scale", "7.5",
            "--num_images_per_sample", "1",
            "--seed", str(seed),
        ]

        if args.image_root:
            cmd += ["--image_root", args.image_root]

        logger.info(f"  seed={seed}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"  seed={seed} failed: {result.stderr[:200]}")
        else:
            phase2_results.append({
                "param": "seed", "value": seed,
                "output_dir": str(output_dir),
            })

    # Phase 2 결과 요약 저장
    summary_path = Path(args.output_base) / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "guidance_scales_tested": PHASE2_GUIDANCE_SCALES,
            "inference_steps_tested": PHASE2_INFERENCE_STEPS,
            "seeds_tested": PHASE2_SEEDS,
            "results": phase2_results,
        }, f, indent=2)

    logger.info(f"Phase 2 complete: {len(phase2_results)} parameter combinations tested")
    logger.info(f"Summary saved: {summary_path}")
    return len(phase2_results) > 0


# =============================================================================
# Phase 3: 미학습 데이터 일반화 검증
# =============================================================================

# 학습에 사용된 이미지 ID
TRAINED_IMAGE_IDS = {"0002cc93b.jpg", "0007a71bf.jpg", "000a4bcdd.jpg"}

# 클래스별 선별 개수
UNSEEN_SAMPLES_PER_CLASS = 3


def run_phase3(args):
    """Phase 3: 미학습 데이터로 일반화 능력 평가"""
    logger.info("=" * 60)
    logger.info("  Phase 3: Unseen Data Generalization")
    logger.info("=" * 60)

    if not args.roi_metadata_path:
        logger.error("Phase 3 requires --roi_metadata_path")
        return False

    # 1. 미학습 ROI 선별
    unseen_rois = _select_unseen_rois(args.roi_metadata_path)
    if not unseen_rois:
        logger.error("No unseen ROIs found")
        return False

    logger.info(f"Selected {len(unseen_rois)} unseen ROIs for testing")

    # 2. 미학습 ROI로부터 hint 생성 및 inference
    output_dir = Path(args.output_base) / "phase3_unseen"
    output_dir.mkdir(parents=True, exist_ok=True)
    hints_dir = output_dir / "generated_hints"
    hints_dir.mkdir(exist_ok=True)
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(exist_ok=True)
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    # hint 생성 및 inference를 위한 임시 JSONL 생성
    unseen_jsonl_path = output_dir / "unseen_test.jsonl"
    hint_generation_success = _generate_unseen_hints(
        unseen_rois, hints_dir, unseen_jsonl_path, args
    )

    if not hint_generation_success:
        logger.warning("Phase 3: Hint generation skipped or failed. "
                       "Attempting inference with ROI images directly.")
        # ROI 이미지를 hint 대신 사용하는 fallback jsonl 생성
        _create_fallback_jsonl(unseen_rois, unseen_jsonl_path)

    # 3. test_controlnet.py로 inference 실행
    cmd = [
        sys.executable, str(args.script_dir / "test_controlnet.py"),
    ]
    cmd += _model_source_args(args)

    cmd += [
        "--jsonl_path", str(unseen_jsonl_path),
        "--output_dir", str(output_dir),
        "--num_inference_steps", "30",
        "--guidance_scale", "7.5",
        "--seed", "42",
    ]

    if args.image_root:
        cmd += ["--image_root", args.image_root]

    logger.info(f"Running inference on unseen data...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Phase 3 inference failed:\n{result.stderr}")
        return False
    logger.info(result.stdout)

    # 결과 요약
    summary_path = output_dir / "generation_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info(
            f"Phase 3 complete: {summary['generated']}/{summary['total_samples']} "
            f"unseen samples generated"
        )
        return summary["generated"] > 0

    return False


def _select_unseen_rois(roi_metadata_path: str) -> List[Dict]:
    """roi_metadata.csv에서 미학습 ROI를 클래스별로 선별합니다.

    선별 기준:
    - 학습 미사용 이미지의 ROI
    - 클래스별 suitability_score 상위 UNSEEN_SAMPLES_PER_CLASS개
    - recommendation이 'suitable' 또는 'acceptable'인 것
    """
    rois_by_class = {}

    with open(roi_metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            if image_id in TRAINED_IMAGE_IDS:
                continue

            class_id = int(row["class_id"])
            try:
                score = float(row["suitability_score"])
            except (ValueError, KeyError):
                score = 0.0

            recommendation = row.get("recommendation", "")
            if recommendation not in ("suitable", "acceptable"):
                continue

            if class_id not in rois_by_class:
                rois_by_class[class_id] = []
            rois_by_class[class_id].append({
                "image_id": image_id,
                "class_id": class_id,
                "region_id": int(row["region_id"]),
                "suitability_score": score,
                "defect_subtype": row.get("defect_subtype", "general"),
                "background_type": row.get("background_type", "complex_pattern"),
                "stability_score": float(row.get("stability_score", 0.5)),
                "linearity": float(row.get("linearity", 0.5)),
                "solidity": float(row.get("solidity", 0.5)),
                "extent": float(row.get("extent", 0.5)),
                "aspect_ratio": float(row.get("aspect_ratio", 1.0)),
                "roi_image_path": row.get("roi_image_path", ""),
                "roi_mask_path": row.get("roi_mask_path", ""),
                "prompt": row.get("prompt", ""),
            })

    # 클래스별 상위 suitability_score 선별
    selected = []
    for class_id in sorted(rois_by_class.keys()):
        class_rois = sorted(
            rois_by_class[class_id],
            key=lambda x: x["suitability_score"],
            reverse=True,
        )
        n = min(UNSEEN_SAMPLES_PER_CLASS, len(class_rois))
        selected.extend(class_rois[:n])
        logger.info(
            f"  Class {class_id}: selected {n} ROIs "
            f"(total available: {len(class_rois)}, "
            f"top score: {class_rois[0]['suitability_score']:.3f})"
        )

    return selected


def _generate_unseen_hints(
    unseen_rois: List[Dict],
    hints_dir: Path,
    output_jsonl_path: Path,
    args,
) -> bool:
    """미학습 ROI로부터 hint 이미지를 생성하고 JSONL을 작성합니다."""
    try:
        import cv2
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.preprocessing.hint_generator import HintImageGenerator
    except ImportError as e:
        logger.warning(f"Cannot import hint generator dependencies: {e}")
        return False

    generator = HintImageGenerator()
    jsonl_entries = []

    for roi in unseen_rois:
        roi_image_path = roi["roi_image_path"]
        roi_mask_path = roi["roi_mask_path"]

        # 경로 resolve
        try:
            search_dirs = [args.image_root, str(Path(args.jsonl_path).parent)]
            img_path = _resolve_roi_path(roi_image_path, search_dirs)
            mask_path = _resolve_roi_path(roi_mask_path, search_dirs)
        except FileNotFoundError as e:
            logger.warning(f"  Skip {roi['image_id']} region {roi['region_id']}: {e}")
            continue

        # 이미지 로드
        roi_image = cv2.imread(str(img_path))
        if roi_image is None:
            logger.warning(f"  Cannot load image: {img_path}")
            continue
        roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

        roi_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if roi_mask is None:
            logger.warning(f"  Cannot load mask: {mask_path}")
            continue
        roi_mask = (roi_mask > 127).astype(np.uint8)

        # hint 생성
        defect_metrics = {
            "linearity": roi["linearity"],
            "solidity": roi["solidity"],
            "extent": roi["extent"],
            "aspect_ratio": roi["aspect_ratio"],
        }

        hint_image = generator.generate_hint_image(
            roi_image=roi_image,
            roi_mask=roi_mask,
            defect_metrics=defect_metrics,
            background_type=roi["background_type"],
            stability_score=roi["stability_score"],
        )

        # hint 저장
        hint_filename = (
            f"{roi['image_id']}_class{roi['class_id']}"
            f"_region{roi['region_id']}_hint.png"
        )
        hint_path = hints_dir / hint_filename
        generator.save_hint_image(hint_image, hint_path)

        # 프롬프트 생성
        prompt = roi.get("prompt", "")
        if not prompt:
            prompt = (
                f"a {roi['defect_subtype']} surface defect on "
                f"{roi['background_type']} metal surface, "
                f"steel defect class {roi['class_id']}"
            )

        jsonl_entries.append({
            "hint": str(hint_path),
            "target": str(img_path),
            "source": str(img_path),
            "prompt": prompt,
            "negative_prompt": (
                "blurry, low quality, artifacts, noise, distorted, "
                "warped, unrealistic, oversaturated"
            ),
        })

    if not jsonl_entries:
        return False

    # JSONL 저장
    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Generated {len(jsonl_entries)} hint images -> {output_jsonl_path}")
    return True


def _create_fallback_jsonl(unseen_rois: List[Dict], output_jsonl_path: Path):
    """hint 생성이 불가능할 때, ROI 이미지 정보만으로 JSONL을 작성합니다.

    이 경우 Colab에서 수동으로 hint 생성 후 재실행해야 합니다.
    """
    entries = []
    for roi in unseen_rois:
        prompt = roi.get("prompt", "")
        if not prompt:
            prompt = (
                f"a {roi['defect_subtype']} surface defect on "
                f"{roi['background_type']} metal surface, "
                f"steel defect class {roi['class_id']}"
            )

        entries.append({
            "hint": roi["roi_image_path"],  # hint 대신 원본 사용 (임시)
            "target": roi["roi_image_path"],
            "source": roi["roi_image_path"],
            "prompt": prompt,
            "negative_prompt": (
                "blurry, low quality, artifacts, noise, distorted, "
                "warped, unrealistic, oversaturated"
            ),
            "_note": "fallback: using ROI image as hint (hint generation failed)",
        })

    with open(output_jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"Fallback JSONL created with {len(entries)} entries -> {output_jsonl_path}"
    )


def _resolve_roi_path(path_str: str, search_dirs: list) -> Path:
    """ROI 이미지/마스크 경로를 resolve합니다."""
    path = Path(path_str)

    if path.is_absolute() and path.exists():
        return path

    filename = path.name

    for base_dir in search_dirs:
        if base_dir is None:
            continue
        base = Path(base_dir)

        candidate = base / path_str
        if candidate.exists():
            return candidate

        candidate = base / filename
        if candidate.exists():
            return candidate

        for subdir in ["images", "masks", "roi_patches/images", "roi_patches/masks"]:
            candidate = base / subdir / filename
            if candidate.exists():
                return candidate

    if path.exists():
        return path

    raise FileNotFoundError(f"Cannot find ROI file: '{path_str}'")


# =============================================================================
# Phase 4: 시각화 및 리포트
# =============================================================================

def run_phase4(args):
    """Phase 4: 학습 Loss 시각화 및 검증 결과 종합 리포트"""
    logger.info("=" * 60)
    logger.info("  Phase 4: Visualization & Report")
    logger.info("=" * 60)

    output_dir = Path(args.output_base) / "phase4_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for Phase 4")
        return False

    success = True

    # 4-1. Loss 곡선 시각화
    if args.training_log_path and Path(args.training_log_path).exists():
        _visualize_training_loss(args.training_log_path, output_dir)
    else:
        logger.warning("Training log not found, skipping loss visualization")

    # 4-2. Phase 1~3 결과 종합
    _generate_summary_report(args.output_base, output_dir)

    # 4-3. Phase 2 파라미터별 비교 그리드
    _create_phase2_comparison_grid(args.output_base, output_dir)

    logger.info(f"Phase 4 complete: reports saved to {output_dir}")
    return success


def _visualize_training_loss(log_path: str, output_dir: Path):
    """training_log.json 기반 Loss 곡선과 분포를 시각화합니다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(log_path) as f:
        log_data = json.load(f)

    steps = [entry["step"] for entry in log_data]
    losses = [entry["loss"] for entry in log_data]
    lrs = [entry["lr"] for entry in log_data]
    epochs = [entry["epoch"] for entry in log_data]

    # --- Figure 1: Loss 추이 + 이동 평균 ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Loss 추이
    ax1 = axes[0]
    ax1.plot(steps, losses, "b-", alpha=0.4, linewidth=0.8, label="Loss (raw)")

    # 10-step 이동 평균
    window = 10
    if len(losses) >= window:
        moving_avg = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            moving_avg.append(np.mean(losses[start:i + 1]))
        ax1.plot(steps, moving_avg, "r-", linewidth=2.0,
                 label=f"Moving Average ({window}-step)")

    # Warmup 구간 표시
    warmup_end_step = 50
    ax1.axvline(x=warmup_end_step, color="green", linestyle="--",
                alpha=0.7, label=f"Warmup End (step {warmup_end_step})")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # LR 추이 (secondary axis)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, lrs, "g--", alpha=0.4, linewidth=0.8, label="Learning Rate")
    ax1_twin.set_ylabel("Learning Rate", color="green")
    ax1_twin.tick_params(axis="y", labelcolor="green")

    # Loss 분포 히스토그램
    ax2 = axes[1]
    ax2.hist(losses, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=np.mean(losses), color="red", linestyle="--",
                label=f"Mean: {np.mean(losses):.4f}")
    ax2.axvline(x=np.median(losses), color="orange", linestyle="--",
                label=f"Median: {np.median(losses):.4f}")
    ax2.set_xlabel("Loss")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Loss Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_dir / 'training_loss_curve.png'}")

    # --- Figure 2: Epoch별 평균 Loss ---
    epoch_losses = {}
    for entry in log_data:
        ep = entry["epoch"]
        if ep not in epoch_losses:
            epoch_losses[ep] = []
        epoch_losses[ep].append(entry["loss"])

    epoch_nums = sorted(epoch_losses.keys())
    epoch_avg = [np.mean(epoch_losses[ep]) for ep in epoch_nums]

    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.bar(epoch_nums, epoch_avg, color="steelblue", alpha=0.7, width=0.8)
    ax.plot(epoch_nums, epoch_avg, "r-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Loss")
    ax.set_title("Average Loss per Epoch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2.savefig(output_dir / "epoch_avg_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"  Saved: {output_dir / 'epoch_avg_loss.png'}")

    # 통계 요약
    stats = {
        "total_steps": len(log_data),
        "total_epochs": max(epochs) + 1 if epochs else 0,
        "loss_mean": float(np.mean(losses)),
        "loss_median": float(np.median(losses)),
        "loss_std": float(np.std(losses)),
        "loss_min": float(np.min(losses)),
        "loss_min_step": steps[int(np.argmin(losses))],
        "loss_max": float(np.max(losses)),
        "loss_max_step": steps[int(np.argmax(losses))],
        "final_loss": losses[-1] if losses else None,
        "nan_count": sum(1 for l in losses if np.isnan(l)),
    }

    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  Saved: {stats_path}")

    # 콘솔 출력
    logger.info(f"  Training Statistics:")
    logger.info(f"    Total Steps: {stats['total_steps'] * 10} "
                f"(logged every 10 steps)")
    logger.info(f"    Loss Mean:   {stats['loss_mean']:.4f}")
    logger.info(f"    Loss Min:    {stats['loss_min']:.4f} "
                f"(step {stats['loss_min_step']})")
    logger.info(f"    Loss Max:    {stats['loss_max']:.4f} "
                f"(step {stats['loss_max_step']})")
    logger.info(f"    Final Loss:  {stats['final_loss']:.4f}")
    logger.info(f"    NaN Count:   {stats['nan_count']}")


def _generate_summary_report(output_base: str, report_dir: Path):
    """Phase 1~3 결과를 종합하는 텍스트 리포트를 생성합니다."""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  ControlNet Model Validation Report")
    report_lines.append("=" * 70)
    report_lines.append("")

    base = Path(output_base)

    # Phase 1 결과
    p1_summary = base / "phase1_basic" / "generation_summary.json"
    if p1_summary.exists():
        with open(p1_summary) as f:
            p1 = json.load(f)
        report_lines.append("Phase 1: Basic Generation")
        report_lines.append("-" * 40)
        report_lines.append(f"  Samples: {p1['generated']}/{p1['total_samples']}")
        report_lines.append(f"  Model: {p1.get('model_path', 'N/A')}")
        report_lines.append(f"  Steps: {p1.get('num_inference_steps', 'N/A')}")
        report_lines.append(f"  Guidance: {p1.get('guidance_scale', 'N/A')}")
        report_lines.append(f"  Status: {'PASS' if p1['generated'] > 0 else 'FAIL'}")
        report_lines.append("")
    else:
        report_lines.append("Phase 1: Not executed")
        report_lines.append("")

    # Phase 2 결과
    p2_summary = base / "phase2_summary.json"
    if p2_summary.exists():
        with open(p2_summary) as f:
            p2 = json.load(f)
        report_lines.append("Phase 2: Parameter Exploration")
        report_lines.append("-" * 40)
        report_lines.append(f"  Guidance scales: {p2.get('guidance_scales_tested', [])}")
        report_lines.append(f"  Inference steps: {p2.get('inference_steps_tested', [])}")
        report_lines.append(f"  Seeds: {p2.get('seeds_tested', [])}")
        report_lines.append(f"  Combinations tested: {len(p2.get('results', []))}")
        report_lines.append("")
    else:
        report_lines.append("Phase 2: Not executed")
        report_lines.append("")

    # Phase 3 결과
    p3_summary = base / "phase3_unseen" / "generation_summary.json"
    if p3_summary.exists():
        with open(p3_summary) as f:
            p3 = json.load(f)
        report_lines.append("Phase 3: Unseen Data Generalization")
        report_lines.append("-" * 40)
        report_lines.append(f"  Unseen samples: {p3['generated']}/{p3['total_samples']}")
        report_lines.append(f"  Status: {'PASS' if p3['generated'] > 0 else 'FAIL'}")
        report_lines.append("")
    else:
        report_lines.append("Phase 3: Not executed")
        report_lines.append("")

    # 저장
    report_text = "\n".join(report_lines)
    report_path = report_dir / "validation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"  Summary report saved: {report_path}")
    logger.info("\n" + report_text)


def _create_phase2_comparison_grid(output_base: str, report_dir: Path):
    """Phase 2의 파라미터별 생성 결과를 하나의 비교 그리드로 결합합니다."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("PIL not available, skipping Phase 2 comparison grid")
        return

    base = Path(output_base)
    images_with_labels = []

    # guidance_scale 비교
    for gs in PHASE2_GUIDANCE_SCALES:
        gs_dir = base / f"phase2_gs{gs}"
        if not gs_dir.exists():
            continue
        # 첫 번째 생성 이미지 찾기
        for img_file in sorted(gs_dir.glob("**/*_generated_*.png")):
            images_with_labels.append((img_file, f"gs={gs}"))
            break
        for img_file in sorted(gs_dir.glob("**/*_gen0.png")):
            images_with_labels.append((img_file, f"gs={gs}"))
            break

    if not images_with_labels:
        logger.info("  No Phase 2 images found for comparison grid")
        return

    # 그리드 생성
    imgs = []
    labels = []
    for img_path, label in images_with_labels:
        try:
            img = Image.open(img_path).convert("RGB")
            imgs.append(img)
            labels.append(label)
        except Exception:
            pass

    if not imgs:
        return

    # 동일 크기로 리사이즈
    target_h = min(img.height for img in imgs)
    target_w = min(img.width for img in imgs)
    resized = [img.resize((target_w, target_h), Image.LANCZOS) for img in imgs]

    label_h = 30
    n = len(resized)
    grid_w = target_w * n
    grid_h = target_h + label_h

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(resized, labels)):
        x_offset = i * target_w
        grid.paste(img, (x_offset, label_h))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (target_w - text_w) // 2
        draw.text((text_x, 5), label, fill=(0, 0, 0), font=font)

    grid_path = report_dir / "phase2_parameter_comparison.png"
    grid.save(grid_path)
    logger.info(f"  Phase 2 comparison grid saved: {grid_path}")


# =============================================================================
# Utility
# =============================================================================

def _model_source_args(args) -> List[str]:
    """model_path 또는 pipeline_path를 test_controlnet.py CLI 인자로 변환합니다.

    우선순위: model_path > pipeline_path
    저장 최적화(--skip_save_pipeline) 적용 후에는 pipeline이 없으므로
    model_path가 기본이 됩니다.
    """
    if args.model_path:
        return ["--model_path", args.model_path]
    elif args.pipeline_path:
        return ["--pipeline_path", args.pipeline_path]
    else:
        raise ValueError("Either --model_path or --pipeline_path must be specified")


def _get_first_hint(jsonl_path: str) -> Optional[Dict]:
    """JSONL에서 첫 번째 샘플의 hint 경로와 프롬프트를 반환합니다."""
    jsonl = Path(jsonl_path)
    if not jsonl.exists():
        return None

    with open(jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            hint_path = sample.get("hint", "")

            # hint 경로 resolve 시도
            resolved = None
            data_dir = jsonl.parent
            search_dirs = [str(data_dir), str(data_dir.parent)]

            path = Path(hint_path)
            if path.is_absolute() and path.exists():
                resolved = str(path)
            else:
                for base in search_dirs:
                    candidate = Path(base) / hint_path
                    if candidate.exists():
                        resolved = str(candidate)
                        break
                    candidate = Path(base) / path.name
                    if candidate.exists():
                        resolved = str(candidate)
                        break
                    for subdir in ["hints", "images"]:
                        candidate = Path(base) / subdir / path.name
                        if candidate.exists():
                            resolved = str(candidate)
                            break

            if resolved is None:
                resolved = hint_path  # Colab에서 resolve 될 수 있으므로 원본 유지

            return {
                "hint_path": resolved,
                "prompt": sample.get("prompt", "a surface defect on metal surface"),
            }

    return None


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="ControlNet Model Validation - Multi-Phase Runner"
    )

    # Model
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Path to trained ControlNet model directory (best_model/ or final_model/). "
             "Recommended: 저장 최적화 후에는 이 옵션을 사용합니다.",
    )
    parser.add_argument(
        "--pipeline_path", type=str, default=None,
        help="Path to saved full pipeline directory (저장 최적화 이전 모델용)",
    )

    # Data
    parser.add_argument(
        "--jsonl_path", type=str, required=True,
        help="Path to train.jsonl",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root directory for image resolution",
    )
    parser.add_argument(
        "--roi_metadata_path", type=str, default=None,
        help="Path to roi_metadata.csv (required for Phase 3)",
    )
    parser.add_argument(
        "--training_log_path", type=str, default=None,
        help="Path to training_log.json (required for Phase 4 loss visualization)",
    )

    # Output
    parser.add_argument(
        "--output_base", type=str, default="test_results",
        help="Base output directory for all phases",
    )

    # Phase selection
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[1, 2, 3, 4],
        help="Phases to run (default: 1 2 3 4)",
    )

    args = parser.parse_args()

    # Validation: model_path 또는 pipeline_path 중 하나는 필수
    if not args.model_path and not args.pipeline_path:
        parser.error("Either --model_path or --pipeline_path must be specified. "
                     "--model_path is recommended for optimized models.")

    # model_path와 pipeline_path 동시 지정 시 model_path 우선 (경고 출력)
    if args.model_path and args.pipeline_path:
        logger.warning(
            "Both --model_path and --pipeline_path specified. "
            "Using --model_path (recommended)."
        )

    # Script directory
    args.script_dir = Path(__file__).parent

    return args


def main():
    args = parse_args()

    model_display = args.model_path or args.pipeline_path
    model_type = "model_path" if args.model_path else "pipeline_path"

    logger.info("=" * 70)
    logger.info("  ControlNet Model Validation - Multi-Phase Runner")
    logger.info("=" * 70)
    logger.info(f"  Phases to run: {args.phases}")
    logger.info(f"  Output base: {args.output_base}")
    logger.info(f"  Model ({model_type}): {model_display}")
    logger.info("")

    Path(args.output_base).mkdir(parents=True, exist_ok=True)

    results = {}

    if 1 in args.phases:
        results["phase1"] = run_phase1(args)

    if 2 in args.phases:
        results["phase2"] = run_phase2(args)

    if 3 in args.phases:
        results["phase3"] = run_phase3(args)

    if 4 in args.phases:
        results["phase4"] = run_phase4(args)

    # 최종 결과
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Validation Complete")
    logger.info("=" * 70)
    for phase, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {phase}: {status}")
    logger.info("=" * 70)

    # 전체 결과 저장
    final_summary = Path(args.output_base) / "validation_results.json"
    with open(final_summary, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {final_summary}")


if __name__ == "__main__":
    main()
