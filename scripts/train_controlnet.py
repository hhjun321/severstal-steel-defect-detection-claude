"""
ControlNet Training Script for Steel Defect Augmentation
=========================================================

Hugging Face diffusers 라이브러리를 사용하여 ControlNet을 학습합니다.
[마스크/힌트 → 결함 이미지] 매핑을 학습하는 파이프라인입니다.

Architecture:
    - Base Model: Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
    - ControlNet: lllyasviel/sd-controlnet-canny (초기 가중치)
    - Conditioning: Multi-channel hint image (R=defect shape, G=structure, B=texture)
    - Text Prompt: Defect + background description

Usage:
    # 기본 학습
    python scripts/train_controlnet.py \\
        --data_dir data/processed/controlnet_dataset \\
        --output_dir outputs/controlnet_training

    # 학습 재개
    python scripts/train_controlnet.py \\
        --data_dir data/processed/controlnet_dataset \\
        --output_dir outputs/controlnet_training \\
        --resume_from_checkpoint latest

    # Mixed precision (fp16) + Gradient checkpointing (메모리 절약)
    python scripts/train_controlnet.py \\
        --data_dir data/processed/controlnet_dataset \\
        --mixed_precision fp16 \\
        --gradient_checkpointing \\
        --gradient_accumulation_steps 4
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hugging Face imports
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from transformers import CLIPTextModel, CLIPTokenizer

# Logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class SteelDefectControlNetDataset(Dataset):
    """
    ControlNet 학습용 Steel Defect 데이터셋.

    train.jsonl에서 로드하며, 각 샘플은:
    - source/target: 결함이 포함된 ROI 이미지 (학습 대상)
    - hint: 3채널 conditioning 이미지 (R=mask, G=structure, B=texture)
    - prompt: 결함 + 배경 텍스트 설명
    - negative_prompt: 네거티브 프롬프트
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str = None,
        resolution: int = 512,
        tokenizer=None,
    ):
        """
        Args:
            jsonl_path: train.jsonl 경로
            image_root: 이미지 파일의 루트 디렉토리 (source/target 경로의 base)
            resolution: 학습 이미지 해상도
            tokenizer: CLIP tokenizer (텍스트 인코딩용)
        """
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.image_root = Path(image_root) if image_root else None
        self.jsonl_path = Path(jsonl_path)
        self.data_dir = self.jsonl_path.parent  # controlnet_dataset 디렉토리

        # Load training entries
        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    self.samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # [-1, 1] 범위
        ])

        self.conditioning_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),  # [0, 1] 범위
        ])

    def _resolve_image_path(self, path_str: str) -> Path:
        """
        이미지 경로를 해석합니다.
        Colab 경로(/content/drive/...)도 로컬 경로로 변환 시도합니다.

        경로 해석 우선순위:
        1. 절대 경로로 존재하면 그대로 사용
        2. image_root + 파일명으로 시도
        3. data_dir 기준 상대 경로로 시도
        4. 프로젝트 루트 기준 상대 경로로 시도
        """
        path = Path(path_str)

        # 1. 절대 경로로 존재하면 사용
        if path.is_absolute() and path.exists():
            return path

        # 파일명만 추출 (Colab 경로 등에서)
        filename = path.name

        # 2. image_root가 지정된 경우
        if self.image_root:
            candidate = self.image_root / filename
            if candidate.exists():
                return candidate

        # 3. data_dir 기준 (controlnet_dataset 디렉토리)
        candidate = self.data_dir / path_str
        if candidate.exists():
            return candidate

        # 4. data_dir의 상위 디렉토리 기준
        candidate = self.data_dir.parent / path_str
        if candidate.exists():
            return candidate

        # 5. 프로젝트 루트 기준
        project_root = Path(__file__).parent.parent
        candidate = project_root / path_str
        if candidate.exists():
            return candidate

        # 6. 상대 경로 그대로
        if path.exists():
            return path

        raise FileNotFoundError(
            f"Cannot resolve image path: '{path_str}'\n"
            f"  Tried: absolute, image_root='{self.image_root}', "
            f"data_dir='{self.data_dir}', project_root='{project_root}'"
        )

    def _resolve_hint_path(self, path_str: str) -> Path:
        """hint 이미지 경로를 해석합니다."""
        path = Path(path_str)

        # 절대 경로로 존재하면 사용
        if path.is_absolute() and path.exists():
            return path

        # 프로젝트 루트 기준 (hint 경로는 보통 상대 경로)
        project_root = Path(__file__).parent.parent
        candidate = project_root / path_str
        if candidate.exists():
            return candidate

        # data_dir 기준
        candidate = self.data_dir / Path(path_str).name
        if candidate.exists():
            return candidate

        # hints 서브디렉토리
        candidate = self.data_dir / "hints" / Path(path_str).name
        if candidate.exists():
            return candidate

        if path.exists():
            return path

        raise FileNotFoundError(f"Cannot resolve hint path: '{path_str}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load target image (결함 포함 ROI)
        target_path = self._resolve_image_path(sample["target"])
        target_image = Image.open(target_path).convert("RGB")

        # Load hint/conditioning image (3-channel)
        hint_path = self._resolve_hint_path(sample["hint"])
        conditioning_image = Image.open(hint_path).convert("RGB")

        # Apply transforms
        target_tensor = self.image_transforms(target_image)
        conditioning_tensor = self.conditioning_transforms(conditioning_image)

        # Tokenize prompt
        prompt = sample.get("prompt", "")
        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids[0]
        else:
            input_ids = torch.tensor([0])  # placeholder

        return {
            "pixel_values": target_tensor,
            "conditioning_pixel_values": conditioning_tensor,
            "input_ids": input_ids,
            "prompt": prompt,
        }


def collate_fn(examples):
    """DataLoader용 collate function."""
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    conditioning_pixel_values = torch.stack([ex["conditioning_pixel_values"] for ex in examples])
    input_ids = torch.stack([ex["input_ids"] for ex in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "input_ids": input_ids,
    }


# =============================================================================
# Validation / Inference
# =============================================================================

def log_validation(
    controlnet, vae, text_encoder, tokenizer, unet, noise_scheduler,
    args, step, device, validation_prompts=None, validation_images=None,
):
    """
    학습 중 검증 이미지를 생성합니다.
    """
    logger.info("Running validation...")

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if validation_prompts is None:
        validation_prompts = [
            "a linear scratch on vertical striped metal surface, steel defect class 3",
            "a compact blob defect on smooth metal surface, steel defect class 1",
        ]

    output_dir = Path(args.output_dir) / "validation" / f"step_{step}"
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device).manual_seed(42)

    for i, prompt in enumerate(validation_prompts):
        # validation_images가 있으면 사용, 없으면 빈 이미지
        if validation_images and i < len(validation_images):
            control_image = validation_images[i]
        else:
            # 기본 검증용 빈 hint 이미지
            control_image = Image.new("RGB", (args.resolution, args.resolution), (0, 0, 0))

        with torch.autocast(str(device)):
            image = pipeline(
                prompt=prompt,
                image=control_image,
                num_inference_steps=20,
                generator=generator,
            ).images[0]

        image.save(output_dir / f"val_{i:03d}.png")
        logger.info(f"  Saved validation image: val_{i:03d}.png")

    del pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    logger.info(f"Validation images saved to {output_dir}")


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """메인 학습 루프."""

    device = torch.device(args.device)
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # =========================================================================
    # 1. 모델 로드
    # =========================================================================
    logger.info("Loading models...")

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # Text encoder (frozen)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    # VAE (frozen)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    # UNet (frozen)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    # ControlNet (trainable)
    if args.controlnet_model_name_or_path:
        logger.info(f"Loading ControlNet from: {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model_name_or_path,
            torch_dtype=weight_dtype,
        )
    else:
        logger.info("Initializing ControlNet from UNet...")
        controlnet = ControlNetModel.from_unet(unet)

    controlnet.to(device)
    controlnet.train()

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # =========================================================================
    # 2. 데이터셋 로드
    # =========================================================================
    logger.info("Loading dataset...")

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / "train.jsonl"

    if not jsonl_path.exists():
        logger.error(f"train.jsonl not found at {jsonl_path}")
        logger.error("Run 'python scripts/prepare_controlnet_data.py' first.")
        return

    train_dataset = SteelDefectControlNetDataset(
        jsonl_path=str(jsonl_path),
        image_root=args.image_root,
        resolution=args.resolution,
        tokenizer=tokenizer,
    )

    if len(train_dataset) == 0:
        logger.error("No valid training samples found!")
        return

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    logger.info(f"Dataset: {len(train_dataset)} samples")
    logger.info(f"Batch size: {args.train_batch_size}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    logger.info(f"Effective batch size: {effective_batch}")

    # =========================================================================
    # 3. Optimizer & Scheduler
    # =========================================================================
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 총 학습 스텝 계산
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # =========================================================================
    # 4. 학습 상태 복원
    # =========================================================================
    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        ckpt_dir = Path(args.output_dir)
        if args.resume_from_checkpoint == "latest":
            # 가장 최근 체크포인트 자동 탐색
            dirs = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda d: int(d.name.split("-")[1]),
            )
            if dirs:
                args.resume_from_checkpoint = str(dirs[-1])
                logger.info(f"Resuming from latest checkpoint: {args.resume_from_checkpoint}")
            else:
                logger.warning("No checkpoints found, training from scratch")
                args.resume_from_checkpoint = None

        if args.resume_from_checkpoint:
            ckpt_path = Path(args.resume_from_checkpoint)
            if (ckpt_path / "controlnet").exists():
                controlnet = ControlNetModel.from_pretrained(ckpt_path / "controlnet")
                controlnet.to(device)
                controlnet.train()
                logger.info(f"Loaded ControlNet from {ckpt_path / 'controlnet'}")

            # optimizer, scheduler 상태 복원
            opt_path = ckpt_path / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
                logger.info("Loaded optimizer state")

            sched_path = ckpt_path / "lr_scheduler.pt"
            if sched_path.exists():
                lr_scheduler.load_state_dict(torch.load(sched_path, map_location=device))
                logger.info("Loaded lr_scheduler state")

            # global_step 복원
            step_file = ckpt_path / "global_step.txt"
            if step_file.exists():
                global_step = int(step_file.read_text().strip())
                resume_step = global_step % num_update_steps_per_epoch
                first_epoch = global_step // num_update_steps_per_epoch
                logger.info(
                    f"Resuming from step {global_step} "
                    f"(epoch {first_epoch}, step_in_epoch {resume_step})"
                )

    # =========================================================================
    # 5. 학습 루프
    # =========================================================================
    logger.info("=" * 80)
    logger.info("  ControlNet Training for Steel Defect Augmentation")
    logger.info("=" * 80)
    logger.info(f"  Device: {device}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    logger.info(f"  Num examples: {len(train_dataset)}")
    logger.info(f"  Num epochs: {args.num_train_epochs}")
    logger.info(f"  Batch size: {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps: {max_train_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  LR scheduler: {args.lr_scheduler}")
    logger.info(f"  Warmup steps: {args.lr_warmup_steps}")
    logger.info("=" * 80)

    progress_bar = tqdm(
        range(global_step, max_train_steps),
        desc="Training",
        disable=False,
    )

    # Loss tracking
    running_loss = 0.0
    best_loss = float("inf")
    loss_log = []

    for epoch in range(first_epoch, args.num_train_epochs):
        controlnet.train()
        epoch_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            # Skip steps for resumed training
            if epoch == first_epoch and step < resume_step:
                continue

            # === Forward Pass ===

            # 1. Target 이미지를 VAE latent로 인코딩
            latents = vae.encode(
                batch["pixel_values"].to(device, dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # 2. 노이즈 추가 (Forward diffusion)
            noise = torch.randn_like(latents)
            batch_size = latents.shape[0]

            # 랜덤 타임스텝 선택
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()

            # 노이즈가 추가된 latent
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3. 텍스트 인코딩
            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(device)
            )[0].to(dtype=weight_dtype)

            # 4. ControlNet conditioning
            controlnet_image = batch["conditioning_pixel_values"].to(
                device, dtype=weight_dtype
            )

            # ControlNet forward
            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # 5. UNet forward (ControlNet residual 주입)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype)
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            # === Loss 계산 ===
            # Prediction target 결정
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss = loss / args.gradient_accumulation_steps

            # === Backward ===
            loss.backward()

            # Gradient accumulation
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        controlnet.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                running_loss += loss.item() * args.gradient_accumulation_steps

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "epoch": f"{epoch}/{args.num_train_epochs}",
                })

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = running_loss / args.logging_steps
                    logger.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    loss_log.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    })
                    running_loss = 0.0

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(
                        controlnet, optimizer, lr_scheduler,
                        global_step, args, loss_log
                    )

                # Validation
                if args.validation_steps and global_step % args.validation_steps == 0:
                    log_validation(
                        controlnet=controlnet,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        noise_scheduler=noise_scheduler,
                        args=args,
                        step=global_step,
                        device=device,
                    )
                    controlnet.train()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            num_batches += 1

            if global_step >= max_train_steps:
                break

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{args.num_train_epochs}: avg_loss={avg_epoch_loss:.4f}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_path = Path(args.output_dir) / "best_model"
            controlnet.save_pretrained(save_path)
            logger.info(f"New best model saved (loss={best_loss:.4f}) -> {save_path}")

    # =========================================================================
    # 6. 최종 저장
    # =========================================================================
    logger.info("Training complete!")

    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    controlnet.save_pretrained(final_path)
    logger.info(f"Final model saved to {final_path}")

    # Save training log
    log_path = Path(args.output_dir) / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(loss_log, f, indent=2)
    logger.info(f"Training log saved to {log_path}")

    # Save full pipeline for inference
    save_full_pipeline(controlnet, args, device, weight_dtype)

    logger.info("=" * 80)
    logger.info("  Training Complete!")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 80)


def save_checkpoint(controlnet, optimizer, lr_scheduler, global_step, args, loss_log):
    """체크포인트를 저장합니다."""
    ckpt_dir = Path(args.output_dir) / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ControlNet 가중치
    controlnet.save_pretrained(ckpt_dir / "controlnet")

    # Optimizer & scheduler 상태
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")

    # Global step
    (ckpt_dir / "global_step.txt").write_text(str(global_step))

    # Training log
    with open(ckpt_dir / "training_log.json", "w") as f:
        json.dump(loss_log, f, indent=2)

    logger.info(f"Checkpoint saved: {ckpt_dir}")

    # 오래된 체크포인트 정리 (최근 N개만 유지)
    if args.checkpoints_total_limit:
        ckpt_parent = Path(args.output_dir)
        checkpoints = sorted(
            [d for d in ckpt_parent.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[1]),
        )
        if len(checkpoints) > args.checkpoints_total_limit:
            for old_ckpt in checkpoints[:-args.checkpoints_total_limit]:
                import shutil
                shutil.rmtree(old_ckpt)
                logger.info(f"Removed old checkpoint: {old_ckpt}")


def save_full_pipeline(controlnet, args, device, weight_dtype):
    """추론용 전체 파이프라인을 저장합니다."""
    try:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet,
            torch_dtype=weight_dtype,
            safety_checker=None,
        )

        pipeline_path = Path(args.output_dir) / "pipeline"
        pipeline.save_pretrained(pipeline_path)
        logger.info(f"Full pipeline saved to {pipeline_path}")
    except Exception as e:
        logger.warning(f"Could not save full pipeline: {e}")
        logger.info("You can still use the ControlNet weights separately.")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ControlNet for Steel Defect Augmentation (diffusers)"
    )

    # === Data ===
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing train.jsonl and hint images",
    )
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root directory for source/target images "
             "(if different from data_dir). "
             "Useful when images are in a separate location.",
    )

    # === Model ===
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained Stable Diffusion model (HF hub or local path)",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path", type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="Pretrained ControlNet for initialization "
             "(set to '' to initialize from UNet)",
    )

    # === Training ===
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],
    )

    # === Optimizer ===
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # === LR Scheduler ===
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant_with_warmup",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    # === Logging & Saving ===
    parser.add_argument("--output_dir", type=str, default="outputs/controlnet_training")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--validation_steps", type=int, default=0,
                        help="Run validation every N steps (0=disabled)")

    # === Resume ===
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Path to checkpoint directory or 'latest'",
    )

    # === Device ===
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate
    if not args.controlnet_model_name_or_path:
        args.controlnet_model_name_or_path = None

    return args


def main():
    args = parse_args()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save training config
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Training config saved to {config_path}")

    # Run training
    train(args)


if __name__ == "__main__":
    main()
