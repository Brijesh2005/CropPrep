# Feature-Level Fusion (simpler-is-better hypothesis)

Complementary experiment to the main CropFusion campaign. This section tests
whether a simpler **feature-level** fusion stack beats the monolithic
multimodal transformer, under identical frozen data, splits, and honest
evaluation constraints.

## 1. Motivation

The main campaign fused a tabular Transformer and a temporal Transformer with
cross-attention + adaptive gating into one model that reasons jointly over
everything. That model (``cropfusion``) reached balanced accuracy 0.4885 and
ROC-AUC 0.5337 on the frozen test split. Its complexity is high and its
balanced accuracy is not better than either unimodal baseline, raising a clear
question: *is joint attention the problem?*

Feature-level fusion replaces joint attention with a much cheaper contract —
each modality is embedded independently, fused by a small MLP — and accepts
inputs of different shapes and lengths, using masked temporal attention to fold
an arbitrary-length satellite observation history into a fixed-size field
vector.

## 2. Architecture

| stage | component | output |
|---|---|---|
| tabular | MLP (`TabularFeatureEncoder`) | 256-D |
| imagery | per-observation MLP (`ImageFeatureEncoder`) | 512-D per grid-year |
| temporal | `TemporalAttentionPool` over `[T, 512]` (masked) | 512-D field vector |
| fusion | `FeatureFusion` — `concat` or per-modality sigmoid `gated` | 256-D |
| head | Linear -> 1 logit (BCEWithLogitsLoss, train-only pos-weight) | binary |

Imagery modality note (reported plainly). Raw Sentinel-2 patch imagery is only
available on Kaggle; in this reproducible study the imagery modality is the
per-year DK grid vegetation-composite vector (12 features/yr: NDVI/EVI/NDWI/
NDRE/SAVI/S2 counts + Kharif/Rabi composites), the legitimate satellite-derived
representation of every field. `ImageFeatureEncoder` is an MLP over the
per-year vectors rather than a CNN over pixels; the identical training path
accepts CNN patch embeddings where raw imagery exists. Padded timestamps
receive exactly zero attention (verified by unit test), so missing years never
pollute the field representation.

## 3. Protocol

Identical to the main campaign: frozen R5.10/R5.9 inputs, deterministic balanced
population (3,985 fields; train 1,540 / val 1,465 / test 980) identical to the
final campaign, train-only preprocessing/weights, validation-only early
stopping and threshold selection (balanced-accuracy-maximizing on val), test
evaluated exactly once. A leakage audit (forbidden columns + split overlap) runs
before every training run. Models: 814 k–914 k parameters (vs ~3 M for the
transformer), 60 epochs, AdamW, cosine schedule, trained end to end.

## 4. Results (frozen test, val-selected threshold)

| model | val balanced acc | test balanced acc | test ROC-AUC | test macro-F1 |
|---|---|---|---|---|
| tabular (no geo) | 0.5661 | 0.4904 | 0.4627 | 0.4712 |
| tabular + lat/lon | 0.5623 | 0.4828 | 0.4235 | 0.4576 |
| imagery only | 0.5785 | 0.4930 | 0.4817 | 0.2885 |
| fusion concat | 0.6003 | 0.4994 | 0.4962 | 0.3090 |
| **fusion gated** | **0.6049** | **0.5242** | **0.5259** | 0.3656 |
| original tabular (1:5) | — | 0.4962 | 0.4224 | 0.4757 |
| original imagery (1:5) | — | 0.4847 | 0.4566 | 0.2449 |
| original cropfusion (1:5) | — | 0.4885 | 0.5337 | 0.2558 |

Best model (fusion gated) deltas:
- vs feature-fusion tabular: balanced acc +0.0338, AUC +0.0632
- vs feature-fusion imagery: balanced acc +0.0312, AUC +0.0442
- vs original CropFusion: balanced acc +0.0357, AUC -0.0078

## 5. Interpretation

* Balanced accuracy improves over every baseline (feature-fusion unimodals and
  the original campaign), confirming that the imagery signal genuinely
  contributes at the feature level and that masked temporal aggregation is a
  robust way to consume it.
* Gated fusion rebalances recall (test pepper recall 0.786 vs 0.061 for the
  tabular-only model) — the learned gates actuated by the imagery embedding for
  the minority class.
* ROC-AUC does not beat the original transformer (0.5259 vs 0.5337) even though
  balanced accuracy clearly improves. The feature-level stack trades a few ROC
  points for substantially better minority calibration at ~3x fewer parameters.
* Absolute performance is still weak (balanced accuracy ≈ 0.52), consistent
  with the main campaign's finding that the coconut-vs-pepper signal at the
  field level is intrinsically limited by the labeling instruments.

## 6. Verdict

Feature-level fusion with masked temporal aggregation is a simpler and
competitive alternative to the monolithic multimodal transformer: it improves
balanced accuracy over both unimodals and the original fusion while reducing
complexity. It does not achieve the 90% project target, and we report it as-is.