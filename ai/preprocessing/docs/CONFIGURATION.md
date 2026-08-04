# Preprocessing — Configuration Guide

Settings resolve **environment (`PRE_*`) > YAML (`PRE_CONFIG_FILE` /
`--config`) > defaults**, validated by pydantic. Unknown keys are rejected.

## Sections

| Section | Key options | Default |
|---------|-------------|---------|
| `tabular` | `scaler` (standard/minmax/robust/none), `handle_missing` (mean/median/zero/drop), `outlier_method` (iqr/zscore), `categorical_encoding` (onehot/ordinal), `drop_constant_features`, `max_correlation`, `numeric_features`, `categorical_features`, `exclude_columns` | standard / mean / iqr / onehot / true / none |
| `image` | `size` (128/224/256), `normalize` (minmax/standard/identity), `ndvi_range`, `evi_range`, `nan_policy`, `invalid_policy`, `clip`, `resize`, `pad` | 128 / minmax / [-1,1] / [-1,1] / zero / zero / true |
| `temporal` | `max_observations`, `min_observations`, `pad_value`, `pad_mode` (right/left), `truncation` (tail/head), `mask_padding`, `sort_by_date`, `drop_duplicate_dates` | 8 / 1 / 0 / right / tail / true |
| `label` | `crop_encoding` (label/onehot), `yield_task` (regression), `yield_scaler` (standard/minmax/none) | label / regression / standard |
| `split` | `strategy` (random/stratified/spatial/temporal/group), ratios, `seed`, `group_column`, `temporal_column`, `test_years`, `val_years` | temporal / .7/.15/.15 / 42 |
| `augmentation` | `enabled`, `flip_horizontal`, `flip_vertical`, `rotation_degrees`, `random_crop`, `brightness_jitter`, `contrast_jitter`, `noise_std` | disabled |
| `dataloader` | `batch_size`, `workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`, `shuffle_train` | 32 / 0 / false / false / null / true |
| `quality` | `min_quality_score`, `require_valid_coordinates`, `require_crop_label`, `require_yield_label`, `min_observations`, `reject_unpaired` | 40 / true / false / false / 1 / false |
| `output_dir` | artifact output directory | `artifacts/preprocessing` |

## Environment variable examples

```bash
PRE_IMAGE__SIZE=224
PRE_TABULAR__SCALER=robust
PRE_SPLIT__STRATEGY=temporal
PRE_SPLIT__TEST_YEARS="[2024, 2025]"
PRE_DATALOADER__BATCH_SIZE=64
PRE_AUGMENTATION__ENABLED=true
```

## Template

```bash
python -c "from ai.preprocessing import save_preprocessing_template; save_preprocessing_template('preprocessing.yaml')"
```

## Notes

* **Feature columns** — when `numeric_features` / `categorical_features` are
  empty they are inferred from the observation's tabular fields, minus
  `exclude_columns` (list the label/identifier columns: crop, yield, year,
  season, village, district, ...).
* **No leakage** — fit every scaler/encoder on the *training* split only.
* **minmax image normalization** uses the physical index ranges (fast, no
  patch reads); `standard` samples a bounded set of patches during fit.
