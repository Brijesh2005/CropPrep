# CropFusion — Final Experimental Report

## Target
- Coconut vs pepper (R5.9 field-composition dominant-crop target).
- CROP_EXTENT_UNIT_STATUS: UNKNOWN (used only upstream to build the target).

## Leakage audit — PASS
- Crop_Extent / fractions / dominance / Yield_Proxy_NPP / benchmark-eligibility absent from every feature matrix.
- No target-derived features. - No future information (leading-window imagery).
- Missing imagery slots masked, never imputed as real.

## Spatial split — PASS
- Taluk-grouped field split inherited from R5.9; physical field never spans two splits. - Population balancing sampled only the majority class, deterministically (seed 42).

## Frozen-test results
| model | balanced acc | macro-F1 | ROC AUC |
|---|---|---|---|
| tabular | 0.4962 | 0.4757 | 0.4224 |
| imagery | 0.4847 | 0.2449 | 0.4566 |
| cropfusion | 0.4885 | 0.2558 | 0.5337 |

## Fusion delta
- vs tabular (balanced acc): -0.0077
- vs imagery (balanced acc): 0.0038
- vs tabular (ROC AUC): 0.1113

## Verdict
> The experiments indicate that multimodal fusion does not consistently outperform the individual modalities under the current field-level labeling conditions.

## 90% target
- Any frozen-test metric >= 0.90: False
- Note: 90%+ is a project target; no metric was manufactured and no methodology was altered to pursue it.
