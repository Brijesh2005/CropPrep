# New Training Architecture — Hybrid Crop Yield Forecasting

> **Generated:** 2026-09-09 (Phase 1)
> **Status:** Target design. Phase 2 will fill in implementations.
> **Project:** "Hybrid Approach to Crop Yield Forecasting: Integrating Weather Data and Satellite Imagery with Machine Learning"

---

## 1. Goal

Replace the legacy **CropFusion training chain**
(`run_pipeline.py → frozen_corpus.py → dataset_manager → STAM → preprocessing → experiment.py → CropFusionModel → build_release.py`)
with a **simpler, fixed architecture** that produces a **fusion model which
experimentally beats both tabular-only and image-only baselines** on the same
validations and an untouched spatial test set.

The legacy architecture was an exploratory R2.x/R5.x experiment framework. The
new architecture is a single, reproducible training flow with explicit
baselines (tabular, image) and a fusion model that must outperform both.

---

## 2. Supervised Data Sources (ONLY these)

1. **Karnataka Government OGD Crop Survey** — crop/season labels with
   village-level GPS (source of the frozen corpus label + crop extent).
2. **Dakshina Kannada DK_Features** (`training/datasets/tabular/DK_Features_*.csv`,
   2018–2024) — environmental grid features.
3. **Sentinel-2 imagery** (Kaggle
   `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`) —
   NDVI/EVI GeoTIFFs.

**Explicitly NOT training sources:** `data_season.csv`,
`ICRISAT-District Level Data.csv`, `cropdata_updated.csv`, `All-India*`,
`dataset.csv`, and `Yield_Proxy_NPP` as a yield target.

---

## 3. Fixed Architecture (Data Flow)

```
                     MASTER DATASET   (tabular + Sentinel-2, frozen split)
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
      TABULAR BRANCH                          IMAGE (Sentinel-2) BRANCH
   DK_Features + OGD crop/survey             NDVI/EVI GeoTIFF patches
              │                                         │
   tabular encoder (benchmarked)            image encoder (benchmarked)
   XGBoost / CatBoost / LightGBM            EfficientNetV2 / ConvNeXt / ResNet
              │                                         │
              └──────────────┬──────────────────────────┘
                             │  feature concatenation
                   ┌─────────▼────────────┐
                   │     FUSION MLP       │  (concatenated learned embeddings)
                   └─────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
      CROP CLASSIFICATION HEAD       YIELD REGRESSION HEAD
      (softmax: coconut/pepper/...)  (continuous yield)
```

### 3.1 Branches

| Branch | Input | Encoders (benchmarked) | Output |
|--------|-------|------------------------|--------|
| Tabular | DK_Features + OGD labels/features | XGBoost, CatBoost, LightGBM | tabular embedding |
| Image | Sentinel-2 NDVI/EVI patches | EfficientNetV2, ConvNeXt, ResNet | image embedding |

### 3.2 Fusion

- Concatenate the **learned** tabular and image embeddings.
- Feed into a **small fusion MLP**.
- Attach two heads: **crop classification** (softmax) + **yield regression**.
- Train/fine-tune the fusion jointly.

### 3.3 Evaluation

- Use the **exact same** spatial leave-one-taluk-out split and the **untouched
  spatial test set** (Sullia) for every baseline and the fusion model.
- Report the same metric set for tabular-only, image-only, and fusion.
- **Constraint:** fusion must experimentally exceed both single-modality models.
  No manipulation of test labels or splits.

---

## 4. Frozen Corpus (REUSED as-is, NEVER modified)

- **File:** `govt_crop_matched_v2/crop_supervised_v2.csv` (10,674 samples).
- **Manifest:** `training_manifests/crop_supervised_v2.0_manifest.json`.
- **Split:** spatial leave-one-taluk-out.
  - Train: Belthangady, Mangalore, Bantwal (5,924)
  - Val: Puttur (2,459)
  - Test: Sullia (2,291)
- **Classes:** coconut (6,865), pepper (3,695), coffee (101), cardamom (11), blackgram (2).
- **Provenance:** `govt_crop_matched_v2/provenance.json` (per-record matching:
  `env_match_distance_m`, `env_dk_index`, `env_match_method=knn_idw`,
  `env_match_confidence`, `env_season_for_features`, `env_support`).

Phase 1 does **not** modify the frozen corpus, its split, labels, Sentinel-2
data, or the yield target. These are frozen inputs to all training.

> The frozen corpus loader is `training/kaggle/frozen_corpus.py`. Its
> **data-loading / provenance-reading** responsibilities are retained for the
> new `prepare_data` step; its **routing into the legacy model** is retired.

---

## 5. Tabular / Image Matching (REUSED)

- `training/matching/spatial_tabular_matcher.py` — K-NN / IDW matching of OGD
  survey points to DK_Features grid cells.
  - Leakage guard: excludes `Yield_Proxy_NPP` when used as a target.
  - Year-/season-aware matching.
- Provenance fields above are produced by this matcher and preserved.
- `training/dataset_manager/providers/kaggle_image.py` — Sentinel-2 retrieval.
- `training/stam/patch_generator.py` / `coordinate_transform.py` — patch
  extraction from GeoTIFFs (reused by the image branch).

---

## 6. New Training Entry Points (skeletons in Phase 1)

| Entry point | Role | Phase |
|-------------|------|-------|
| `training/prepare_data.py` | Build the master dataset: load frozen corpus → tabular features + Sentinel-2 patches, frozen split | 2 (skeleton now) |
| `training/train_tabular.py` | Benchmark tabular models (XGBoost/CatBoost/LightGBM) | 2 (skeleton now) |
| `training/train_image.py` | Benchmark image models (EfficientNetV2/ConvNeXt/ResNet) | 2 (skeleton now) |
| `training/train_fusion.py` | Fusion: concat embeddings → MLP → crop + yield heads, joint training | 2 (skeleton now) |
| `training/evaluate.py` | Evaluate all models on the untouched test set, same metrics | 2 (skeleton now) |

---

## 7. What Is Retired (legacy training only)

The following form the **legacy training path** and are replaced by the new
flow (they may remain as read-only references but are not used to train the
final model):

- `training/kaggle/scripts/run_pipeline.py` — legacy driver (retired).
- `training/training/experiment.py` — legacy orchestrator (retired).
- `training/models/cropfusion.py` (CropFusionModel) — legacy architecture
  (replaced by new tabular/image/fusion modules).
- R5.x experiment scripts/notebooks (diagnostics, audits, separation probes).

> See `LEGACY_TRAINING_MAP.md` for the full per-module classification.

---

## 8. Inference Path (unchanged for now)

The backend continues to load a release package:

```
application/backend/run.py → app/main.py
  → app/services/release_model_registry.py
  → app/modules/inference/service.py (InferenceEngine)
  → inference_package/release/loader.py
```

A future Phase will re-export the **new** fusion model through
`training/kaggle/scripts/build_release.py` (kept for release packaging).
