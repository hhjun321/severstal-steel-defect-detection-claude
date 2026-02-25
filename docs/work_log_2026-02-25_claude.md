# 작업 일지 — 2026-02-25

## 목적

`improvement_plan.md`(프로젝트 루트)에 정리된 CASDA 벤치마크 코드 버그 수정.
`next_step_mod.md` 기반 실험 설계를 실제로 실행하기 위한 사전 코드 정비 작업.

---

## 수정 파일 요약

| 파일 | 우선순위 | 수정 내용 |
|------|----------|-----------|
| `scripts/package_casda_data.py` | Critical | `--roi-metadata` 옵션 추가, ROI 점수 전파 구현 |
| `src/training/dataset_yolo.py` | Major | Pruning fallback 로직 수정 |
| `src/training/dataset.py` | Major | Pruning fallback 로직 수정 + docstring 수정 |
| `src/training/metrics.py` | Medium | Segmentation per-class 메트릭 표시 수정 |

---

## 상세 수정 내역

### 1. `scripts/package_casda_data.py`

**문제**
ControlNet 생성 이미지 패키징 시 `quality_score`가 전부 0.0으로 기록됨.
이는 `generation_summary.json`에 품질 점수가 없었기 때문이며, 실제 적합도 점수는
`data/processed/roi_patches/roi_metadata.csv`에 `(image_id, class_id, region_id)` 키로 존재함.

**수정 내용**
- `parse_roi_key_from_filename()` 함수 추가
  - 파일명 패턴 `454d794dc.jpg_class4_region0_gen0.png` → `('454d794dc.jpg', 4, 0)` 파싱
- `build_roi_suitability_map()` 함수 추가
  - `roi_metadata.csv` 읽어서 `{(image_id, class_id, region_id): suitability_score}` 딕셔너리 반환
- `package_data()` 함수에 `roi_metadata: Optional[Path] = None` 파라미터 추가
- suitability_score 결정 우선순위 구현:
  1. `--roi-metadata` 지정 시 ROI 맵에서 조회 (최우선)
  2. `--quality-json` / `generation_summary.json`의 quality 섹션
  3. 모두 없으면 `--default-score` (기본값 1.0)
- `main()`에 `--roi-metadata` argparse 인자 추가

**효과**
재패키징 시 `packaging_report.json`의 `quality_score`가 실제 ROI 적합도 점수(0.54~0.69)로 기록됨.

---

### 2. `src/training/dataset_yolo.py` — `_add_casda_to_training()`

**문제**
Pruning 모드에서 `suitability_score = 0.0`이면 `0.0 >= 0.63(threshold)` → False → 전체 제거.
`casda_pruning.total_images = 0`이 되어 실험 불가.

**수정 전 (버그)**
```python
all_samples = [s for s in all_samples if s.get('suitability_score', 1.0) >= threshold]
all_samples.sort(key=lambda x: x.get('suitability_score', 1.0), reverse=True)
all_samples = all_samples[:top_k]
```

**수정 후 (fallback 추가)**
```python
filtered = [s for s in all_samples if s.get('suitability_score', 0.0) >= threshold]
if len(filtered) >= top_k:
    filtered.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
    all_samples = filtered[:top_k]
else:
    logger.warning(f"Pruning: only {len(filtered)} samples pass threshold ...")
    all_samples.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
    all_samples = all_samples[:top_k]
```

**효과**
점수가 없거나 threshold 미달이어도 score 기준 상위 top_k(기본 2,000매)를 선택.

---

### 3. `src/training/dataset.py` — `CASDASyntheticDataset._load_metadata()`

**문제**
dataset_yolo.py와 동일한 pruning 버그 존재. 또한 docstring에 "5,000 CASDA synthetic images"로
잘못 기재 (실제 생성 수: 2,901개).

**수정 내용**
- Pruning fallback 로직: dataset_yolo.py와 동일한 패턴 적용
- L8 docstring: `"all 5,000 CASDA synthetic images"` → `"all ~2,901 synthetic images (ControlNet v4)"`

---

### 4. `src/training/metrics.py` — `BenchmarkReporter`

**문제**
`save_comparison_csv()`와 `print_summary()` 모두 per-class 메트릭으로 `class_ap`만 조회.
DeepLabV3+(segmentation)는 `class_dice`를 반환하므로 Class1~4가 전부 0.0000으로 표시됨.

**수정 내용 — `save_comparison_csv()`**
- fieldnames: `Class{i}_AP` → `Class{i}_Score` (detection/segmentation 공용)
- segmentation 판별: `is_segmentation = 'dice_mean' in metrics and 'mAP@0.5' not in metrics`
- per-class 소스 선택:
  ```python
  per_class = metrics.get('class_dice', {}) if is_segmentation else metrics.get('class_ap', {})
  ```

**수정 내용 — `print_summary()`**
```python
# 수정 전
cap = m.get('class_ap', {})

# 수정 후
cap = m.get('class_ap', {}) or m.get('class_dice', {})
```

---

## 발견된 핵심 사실

| 항목 | 값 |
|------|----|
| 실제 생성된 CASDA 이미지 수 | 2,901개 (`packaging_report.json` 확인) |
| 현재 quality_score 상태 | 전부 0.0 (재패키징 필요) |
| ROI suitability_score 범위 | 0.54 ~ 0.69 (`roi_metadata.csv` 확인) |
| CASDA-Pruning 현재 상태 | 0개 (재패키징 전까지) |
| Pruning threshold | 0.63 (기본값) |
| Pruning top-K | 2,000 (기본값) |

---

## 다음 단계

1. **Colab에서 재패키징** — `--roi-metadata` 옵션으로 실제 점수 반영
   ```bash
   python scripts/package_casda_data.py \
       --generated-dir <augmented_images_v4/generated> \
       --summary-json  <generation_summary.json> \
       --hint-dir      <controlnet_dataset_v4/hints> \
       --output-dir    <data/augmented> \
       --roi-metadata  data/processed/roi_patches/roi_metadata.csv \
       --suitability-threshold 0.63 \
       --pruning-top-k 2000
   ```

2. **벤치마크 실험 실행** — `next_step_mod.md` 계획에 따라
   ```bash
   python scripts/run_benchmark.py \
       --config configs/benchmark_experiment.yaml \
       --casda-dir <data/augmented> \
       --groups all
   ```

---

## 참고 파일

- `improvement_plan.md` — 이번 작업의 계획 문서
- `next_step_mod.md` — 벤치마크 실험 설계 원본
- `outputs/augmented_dataset_v4/packaging_report.json` — 현재 패키징 상태 확인용
- `data/processed/roi_patches/roi_metadata.csv` — ROI 적합도 점수 보유

---

## ControlNet v4 품질 불량 원인 분석 (실험 데이터 기반)

> 상세 진단 문서: `docs/20260225_1.docs`, `docs/20260225_2.docs`
> **주의**: `20260225_2.docs`가 최종본. `20260225_1.docs`의 3개 항목을 수정함.

### 분석 대상 지표

| 지표 | 값 | 의미 |
|------|-----|------|
| SSIM | 0.0269 | 구조적 유사성 거의 없음 |
| LPIPS | 0.5774 | 높은 지각적 불일치 |
| quality_score (전체) | 0.0 | 패키징 시 점수 미전파 (별도 버그) |
| pruning 통과율 | 0 / 2,901 | threshold 0.7 기준 전량 탈락 |
| artifact_score | 0.9258 | 이미지 자체는 깨끗함 |
| sharpness_score | 0.8830 | 이미지 자체는 선명함 |

→ 생성 이미지 자체의 품질은 나쁘지 않음. 하지만 참조 이미지와 **내용이 전혀 다름**.

---

### 이전 가설 vs 실제 (3개 항목 수정)

| 이전 진단 (오진) | 실제 |
|-----------------|------|
| ~~`training_config.json`이 원본 `controlnet_dataset` 참조~~ | `controlnet_dataset_v4` (1,000 samples) 정상 사용 ✅ |
| ~~`lr_scheduler = constant_with_warmup`~~ | `cosine` 정상 사용 ✅ |
| ~~578 optimizer steps만 수행~~ | `total_steps=578` = **로그 항목 수** (매 10스텝 기록) → 실제 **5,780 steps = 24 epochs** ✅ |

---

### 확인된 근본 원인

#### 🔴 [Critical 1] `source == target` — 동일 파일 참조
- `train.jsonl` 직접 확인: `source`와 `target` 경로가 동일
- 학습 시 모델이 배운 것: **"hint → 특정 ROI 패치의 그레이스케일 복원"**
- 추론 시 (`txt2img`, 순수 노이즈에서 시작): source ROI의 잠재 벡터가 없음
- 결과: 생성 이미지가 참조 ROI 패치와 **공간적으로 무관** → **SSIM ≈ 0**

#### 🔴 [Critical 2] ROI 극단 업스케일 (10–40×)
- `metadata.json` `roi_bbox` 실측: 13px × 29px ~ 63px × 60px 수준의 초소형 패치
- 512×512로 Resize → **10배~40배 업스케일** → 흐릿한 그레이스케일 블롭이 학습 표적
- 모델이 실제로 학습한 것 = **블롭 생성**

#### 🔴 [Critical 3] `force_grayscale_target=True` + SSIM 측정 기준 불일치
- 생성 이미지: 그레이스케일 (R=G=B)
- SSIM 비교 기준: 원본 컬러 ROI 패치
- 구조적으로 SSIM이 낮을 수밖에 없는 사과-오렌지 비교

#### 🟠 [Major] 힌트 포맷 불일치 (`sd-controlnet-canny` 사전학습 vs 커스텀 그레이스케일)
- Phase 2 결과: `guidance_scale ↑` → `quality_score ↓` (역상관)
- 낮은 guidance(=3.0)일수록 품질이 좋음 = **conditioning을 약하게 쓸수록 낫다**
- = 모델이 hint를 활용하지 못하고 base SD 자체 생성에 의존

#### 🟠 [Major] epoch 17 이후 학습 손실 정체 + early stopping
- `loss_min = 0.1768` @ step 4,280 (epoch ≈ 17)
- `early_stopping_patience=5`, `validation_steps=0` (검증 없이 train loss 기반 stopping)
- epoch 22 종료 후 트리거 → epoch 23 도중(step 5,780)에서 중단

---

### quality_score 계산 구조

역산 결과: `quality_score ≈ (color_consistency + artifact + sharpness + SSIM) / 4`

| 성분 | Phase 1 값 | 설명 |
|------|-----------|------|
| color_consistency | 0.6541 | 그레이스케일 → 단색에 가까워 낮음 |
| artifact_score | 0.9258 | 아티팩트 없음 (블롭이지만 깨끗함) |
| sharpness_score | 0.8830 | 선명함 (업스케일 sharpening 효과) |
| **SSIM** | **0.0269** | **quality_score를 0.6254까지 끌어내리는 주범** |

SSIM이 0.4 수준이라면 quality_score ≈ 0.72+ 달성 가능.

---

### v5 핵심 설계 변경 방향

| 항목 | 현재 (v4) | 개선 (v5) |
|------|----------|----------|
| `source`/`target` 관계 | 동일 파일 | source=결함제거 배경, target=결함 원본 |
| 힌트 포맷 | R*0.5+G*0.3+B*0.2 그레이스케일 | Canny 엣지 (결함 마스크 경계) |
| `force_grayscale_target` | `True` | `False` (컬러 유지) |
| 입력 단위 | ROI 패치 (13–63px, 10–40× 업스케일) | 256×256 타일 (최대 2× 업스케일) |
| 추론 방식 | `txt2img` (노이즈에서 생성) | `img2img` (배경 타일 기반) |
| early_stopping | patience=5 (train loss 기반) | 비활성화 (50 epoch 완주) |
| augmentation | False | True (flip, brightness, contrast) |

---

### 현재 벤치마크 진행 가능 여부

| 그룹 | 상태 | 비고 |
|------|------|------|
| `baseline_raw` / `baseline_trad` | ✅ 진행 가능 | 기준선 측정 |
| `casda_full` (2,901장) | ⚠️ 진행 가능 (품질 낮음) | SSIM≈0 이미지지만 파일 존재 |
| `casda_pruning` | ❌ 진행 불가 | 0장 통과 → v5 완료 후 재실행 |
