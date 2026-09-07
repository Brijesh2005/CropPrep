# Methodology

## Data (frozen R5.9/R5.10)
- **Target**: R5.9 field-composition dominant crop, coconut vs pepper, from
  `reports/R5.9/field_dataset_split.csv`. `CROP_EXTENT_UNIT_STATUS = UNKNOWN`;
  Crop_Extent is used only upstream to build the composition target and is
  **never** a model feature.
- **Field identity**: `(taluk||hobli||village).upper() + '|SURVEYID=' +
  survey_id.upper()` (R5.8 rule).
- **Imagery**: per-year DK grid vegetation composites (NDVI, EVI, NDWI, NDRE,
  SAVI, S2 observation counts, Kharif/Rabi composites) matched to each field's
  survey GPS via K-NN IDW (radius 5 km, k=5). Only the **leading window**
  `grid_year <= survey_year` is used (no future information); missing slots are
  masked, never imputed.
- **Tabular environment**: rainfall, temperature, dewpoint, humidity (survey
  frame + leading-window aggregates), elevation, slope, soil clay/sand,
  organic carbon, pH, moisture; categorical land-cover/soil classes.
- **Location** (separate investigation): raw lat/lon.

## Population (balanced, pre-registered)
To avoid ordinary-accuracy inflation, the majority (coconut) class is sampled
within each split to 4x that split's pepper count, deterministically
(seed 42); all minority fields are retained. Pepper is never
oversampled; validation and test contain only real observations.

## Split
The taluk-grouped field split is inherited verbatim from R5.9: a physical field
never appears in more than one split. Train 1540 / val 1465
/ test 980 fields.

## Models and selection
- Tabular: LR, RF, GBM, MLP, XGBoost; best selected on validation balanced
  accuracy.
- Imagery: per-timestep MLP + `TemporalTransformer` (masked) + `CropHead`;
  class-balanced BCE, early stopping, cosine schedule.
- CropFusion: `TabTransformer` + `TemporalTransformer`, fused via cross
  attention (Q=imagery, K/V=tabular) + `AdaptiveGatedFusion` (existing
  `training/models` components); mechanism selected on validation.
- Threshold: fixed 0.5 for all models. No post-hoc test threshold tuning.
