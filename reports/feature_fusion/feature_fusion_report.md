# CropFusion — Feature-Level Fusion Experimental Report

Do a final pass over the results and be honest. See also the generated
`model_comparison.csv`, `metrics.csv`, `training_history.csv`, `predictions.csv`,
`run_metadata.json` (reproducible from `train_feature_fusion.py --compile`),
and the three figures in `figures/`.

## Target
- Coconut vs pepper (R5.9 field-composition dominant-crop target); binary.
- `CROP_EXTENT_UNIT_STATUS: UNKNOWN` (used only upstream to build the target).
- Frozen deterministic balanced population of **3,985 fields** (train 1,540 /
  val 1,465 / test 980), reused verbatim from the final-cropfusion campaign so
  every number here is directly comparable to `reports/final/`.

## Feature-level fusion architecture (the hypothesis under test)
| stage | component | output |
|---|---|---|
| tabular | `TabularFeatureEncoder` (MLP) | 256-D |
| imagery | `ImageFeatureEncoder` per grid-year (MLP) | 512-D per timestep |
| temporal | `TemporalAttentionPool` (masked attention over `[T,512]`) | 512-D field vector |
| fusion | `FeatureFusion` `concat` or `gated` (sigmoid gates) | 256-D fused |
| head | Linear | 1 logit (BCEWithLogitsLoss, pos_weight from train) |

The hypothesis being tested: *good unimodal representations + feature-level
fusion + masked temporal aggregation + correct splits beat a single massive
multimodal transformer.* No test information ever reaches the model.

## Imagery modality (local limitation, reported plainly)
Raw Sentinel-2 patches are only available on the Kaggle platform; the
legitimate, locally-reproducible imagery representation is the **per-year DK
grid vegetation-composite vector** (12 features/yr: NDVI/EVI/NDWI/NDRE/SAVI/S2
count + Kharif/Rabi composites) from the frozen R5.10 grid. `ImageFeatureEncoder`
is therefore an MLP over per-year composite vectors, **not a CNN over pixels**.
The identical code path accepts a CNN patch-embedding backend (config knobs
`pretrained_backbone` / `freeze_backbone`) where raw patches exist. A 4-timestep
leading window (2018–2021 survey-leading years) is padded and masked; padded
timestamps have exactly zero influence (verified by unit test).

## Evaluation protocol (honest constraints)
- Preprocessing (median/std, categorical codes, imagery scaling, pos-weight)
  fit on **train only**.
- Early stopping, model selection and the classification threshold on
  **validation only** (`threshold_val` per model); test evaluated **exactly
  once** per model with the frozen threshold (fixed-0.5 reference also reported
  in `metrics.json`).
- Leakage audit (feature columns + field-level split overlap) runs before every
  training and aborts on any violation.

## Frozen-test results (val-selected threshold)
| model | val balanced acc | test balanced acc | test ROC AUC | test macro-F1 | thr |
|---|---|---|---|---|---|
| tabular (no loc) | 0.5661 | 0.4904 | 0.4627 | 0.4712 | 0.5742 |
| tabular + lat/lon | 0.5623 | 0.4828 | 0.4235 | 0.4576 | 0.8069 |
| imagery only | 0.5785 | 0.4930 | 0.4817 | 0.2885 | 0.5000 |
| **fusion concat** | 0.6003 | 0.4994 | 0.4962 | 0.3090 | 0.5148 |
| **fusion gated (best)** | 0.6049 | **0.5242** | **0.5259** | 0.3656 | 0.5841 |
| --- legacy (fixed 0.5) --- | | | | |
| original tabular | — | 0.4962 | 0.4224 | 0.4757 | 0.5 |
| original imagery | — | 0.4847 | 0.4566 | 0.2449 | 0.5 |
| original cropfusion | — | 0.4885 | 0.5337 | 0.2558 | 0.5 |

## Fusion delta (best = fusion gated)
- vs feature-fusion tabular: balanced acc **+0.0338**, AUC **+0.0632**
- vs feature-fusion imagery: balanced acc **+0.0312**, AUC **+0.0442**
- vs original CropFusion: balanced acc **+0.0357**, AUC **-0.0078**

Balanced accuracy improves over every baseline; ROC AUC edges past the
unimodal models but sits marginally below the original CropFusion transformer
(0.5259 vs 0.5337). The improvement is driven by the imagery-fused signal —
gated fusion rebalances pepper recall (test recall pepper 0.7857 vs tabular's
0.0612) at the cost of coconut recall (0.2628).

## Verdict
> Feature-level fusion with masked temporal aggregation is a simpler and
> competitive alternative to the multimodal transformer: it beats both
> unimodal baselines and the original CropFusion on balanced accuracy, at a
> fraction of the parameters (0.9 M vs ~3 M) and with no per-modality
> attention complexity. Absolute performance remains far below practical
> utility — the field-level coconut-vs-pepper signal is weak.

## 90% target
- Any frozen-test metric >= 0.90: **False**.
- No metric was manufactured and no methodology was altered to pursue it.