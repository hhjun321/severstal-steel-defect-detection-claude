# CASDA 벤치마크 코드 개선 계획

`next_step_mod.md` 기반으로 발견된 문제점과 수정 방향을 정리한다.

---

## 발견된 문제점 (4가지)

### 1. [Critical] CASDA-Pruning 데이터셋 항상 0개

**원인**
- ControlNet 생성 과정에서 `generation_summary.json`에 quality_score가 기록되지 않음
- 패키징 결과(`outputs/augmented_dataset_v4/packaging_report.json`) 확인 시 2,901개 이미지의 `quality_score` **전부 0.0**
- `_add_casda_to_training()` pruning 로직: `0.0 >= 0.63(threshold)` → False → 모든 이미지 제거
- 결과: `casda_pruning.total_images = 0` (벤치마크 실험 불가)

**근본 원인**
`scripts/package_casda_data.py`가 `roi_metadata.csv`의 suitability_score를 생성 이미지에 전파하지 않음.
- `roi_metadata.csv`에 **적합도 점수 존재**: 0.54~0.69 범위, `(image_id, class_id, region_id)` 키로 조회 가능
- 파일명 패턴과 완벽 매칭 가능: `454d794dc.jpg_class4_region0_gen0.png` → `(454d794dc.jpg, 4, 0)`

---

### 2. [Major] `scripts/package_casda_data.py` — ROI 적합도 점수 전파 미구현

**현재 상태**
- `--quality-json` 옵션은 있으나 `roi_metadata.csv`를 직접 읽는 기능 없음
- quality_map이 비어 있으면 `default_score=1.0`으로 전체 포함하도록 설계됐지만, 이전 실행 시 Colab의 `generation_summary.json`에 0.0이 명시되어 있어 0.0으로 기록됨

**필요한 수정**
`--roi-metadata` 옵션 추가 및 다음 함수 신규 작성:

```python
def build_roi_suitability_map(roi_metadata_csv: Path) -> dict:
    """roi_metadata.csv → {(image_id, 1indexed_class_id, region_id): suitability_score}"""
    import pandas as pd
    df = pd.read_csv(roi_metadata_csv)
    mapping = {}
    for _, row in df.iterrows():
        key = (str(row['image_id']), int(row['class_id']), int(row['region_id']))
        mapping[key] = float(row['suitability_score'])
    return mapping

def parse_roi_key_from_filename(filename: str) -> tuple:
    """'454d794dc.jpg_class4_region0_gen0.png' → ('454d794dc.jpg', 4, 0)"""
    m = re.match(r"(.+\.jpg)_class(\d+)_region(\d+)_gen\d+\.png$", filename)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Cannot parse ROI key from: {filename}")
```

`package_data()` 내에서 suitability_score 결정 순서:
1. `--roi-metadata` 지정 시 → roi_map에서 조회 (가장 신뢰성 높음)
2. `--quality-json` / `generation_summary.json`의 quality section
3. 모두 없으면 `--default-score` (기본값 1.0)

---

### 3. [Medium] Pruning Fallback 로직 미비

**영향 파일**
- `src/training/dataset_yolo.py` L344~351 (`_add_casda_to_training()`)
- `src/training/dataset.py` L352~361 (`CASDASyntheticDataset._load_metadata()`)

**현재 코드 (버그)**
```python
# 점수가 0.0이면 threshold 필터가 모든 샘플을 제거
all_samples = [s for s in all_samples if s.get('suitability_score', 1.0) >= threshold]
all_samples.sort(key=lambda x: x.get('suitability_score', 1.0), reverse=True)
all_samples = all_samples[:top_k]
```

**수정 방향**
`next_step_mod.md`의 "상위 2,000매 선별" 취지에 맞게 — threshold 필터 후 충분히 남으면 사용, 부족하면 전체에서 top-K로 fallback:

```python
threshold = casda_config.get('suitability_threshold', 0.63)
top_k = casda_config.get('pruning_top_k', 2000)

filtered = [s for s in all_samples if s.get('suitability_score', 0.0) >= threshold]
if len(filtered) >= top_k:
    # threshold 조건 충족 → 상위 top_k 선택
    filtered.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
    all_samples = filtered[:top_k]
else:
    # 점수 미산정 또는 threshold 미달 → score 기준 상위 top_k (fallback)
    all_samples.sort(key=lambda x: x.get('suitability_score', 0.0), reverse=True)
    all_samples = all_samples[:top_k]
```

---

### 4. [Medium] `BenchmarkReporter` — Segmentation per-class 메트릭 0으로 표시

**영향 파일**: `src/training/metrics.py`

**현재 코드**
```python
# save_comparison_csv() L526~529
class_ap = metrics.get('class_ap', {})   # detection 전용 키
for i in range(1, 5):
    row[f"Class{i}_AP"] = f"{class_ap.get(f'Class{i}', 0.0):.4f}"

# print_summary() L562
cap = m.get('class_ap', {})   # segmentation에서는 비어 있음
```

DeepLabV3+(segmentation)는 `class_dice`를 반환하지만 코드가 `class_ap`만 조회 → 전부 0.0000 표시.

**수정 방향**

```python
# save_comparison_csv()
is_segmentation = 'dice_mean' in metrics and 'mAP@0.5' not in metrics
per_class = metrics.get('class_dice', {}) if is_segmentation else metrics.get('class_ap', {})
col_suffix = 'Dice' if is_segmentation else 'AP'

for i in range(1, 5):
    row[f"Class{i}_{col_suffix}"] = f"{per_class.get(f'Class{i}', 0.0):.4f}"

# print_summary()
cap = m.get('class_ap', {}) or m.get('class_dice', {})
```

---

### 5. [Minor] `dataset.py` Docstring 불일치

**위치**: `src/training/dataset.py` L8

```python
# 현재 (오류)
#   - CASDA-Full: Original + all 5,000 CASDA synthetic images

# 수정
#   - CASDA-Full: Original + all ~2,901 synthetic images (ControlNet v4)
```

실제 생성 수는 `packaging_report.json`에서 확인: `casda_full.total_images = 2901`

---

## 수정 우선순위 및 파일 요약

| 우선순위 | 파일 | 수정 내용 | 효과 |
|---------|------|-----------|------|
| 1 | `scripts/package_casda_data.py` | `--roi-metadata` 옵션, `build_roi_suitability_map()` 신규 | 재패키징 시 실제 점수 반영 |
| 2 | `src/training/dataset_yolo.py` | pruning fallback (top-K) | CASDA-Pruning 실험 가능 |
| 3 | `src/training/dataset.py` | pruning fallback + docstring | 동일 |
| 4 | `src/training/metrics.py` | segmentation per-class 표시 수정 | DeepLabV3+ 결과 정상 출력 |

---

## 실행 플로우 (수정 후)

```bash
# Step 1: ROI 점수 포함 재패키징 (Colab에서 실행)
python scripts/package_casda_data.py \
    --generated-dir /content/drive/.../augmented_images_v4/generated \
    --summary-json  /content/drive/.../augmented_images_v4/generation_summary.json \
    --hint-dir      /content/drive/.../controlnet_dataset_v4/hints \
    --output-dir    /content/drive/.../data/augmented \
    --roi-metadata  data/processed/roi_patches/roi_metadata.csv \
    --suitability-threshold 0.63 \
    --pruning-top-k 2000

# Step 2: 벤치마크 실행
python scripts/run_benchmark.py \
    --config configs/benchmark_experiment.yaml \
    --casda-dir /content/drive/.../data/augmented \
    --groups all
```

---

## 참고: 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `data/processed/roi_patches/roi_metadata.csv` | ROI별 suitability_score 보유 |
| `outputs/augmented_dataset_v4/packaging_report.json` | 현재 패키징 상태 (score=0.0 확인) |
| `scripts/package_casda_data.py` | CASDA 데이터 패키징 (수정 필요) |
| `src/training/dataset_yolo.py` | YOLO 데이터셋 변환 + pruning (수정 필요) |
| `src/training/dataset.py` | PyTorch Dataset + pruning (수정 필요) |
| `src/training/metrics.py` | BenchmarkReporter (수정 필요) |
