# Clean Architecture — CropPrep/CropFusion Final Project

> **Generated:** 2026-09-09
> **Project Title:** "Hybrid Approach to Crop Yield Forecasting: Integrating Weather Data and Satellite Imagery with Machine Learning"

---

## 1. Overview

This document defines the target architecture after cleanup. The final project uses **only** these supervised training sources:

1. **Karnataka Government OGD Crop Survey** (`kodi.karnataka.gov.in` API) — provides supervised crop labels (coconut, pepper, coffee, cardamom) with village-level GPS coordinates, year, and season.
2. **Dakshina Kannada DK_Features datasets** (2018–2024) — Earth Engine exports providing environmental grid features (NDVI, EVI, NDWI, rainfall, temperature, soil, elevation, etc.).
3. **Sentinel-2 satellite imagery** from Kaggle (`shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`) — NDVI/EVI GeoTIFFs.

**Explicitly excluded:** `data_season.csv`, `ICRISAT-District Level Data.csv`, `cropdata_updated.csv`, `All-India` crop dataset, `dataset.csv`, and `Yield_Proxy_NPP` as a yield target.

---

## 2. Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + TS)                    │
│     No login · No auth · DK-restricted Leaflet map          │
│     Select location → Select year/season → Predict          │
│     Shows: best crop + expected yield                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP (JSON)
┌──────────────────────────▼──────────────────────────────────┐
│                     BACKEND (FastAPI)                       │
│  /predict (location → crop + yield)                         │
│  Loads cropfusion_release/ package                          │
│  Uses: ReleasePackageLoader + InferenceEngine               │
└──────────────────────────┬──────────────────────────────────┘
                           │ (offline, separate from runtime)
┌──────────────────────────▼──────────────────────────────────┐
│                     TRAINING (Kaggle CLI)                   │
│  Frozen corpus (OGD × DK_Features × Sentinel-2)             │
│  CropFusionModel (tabular + image + temporal fusion)        │
│  Exports cropfusion_release/                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline

### 3.1 Source Data

| Source | Description | Format | Location |
|--------|-------------|--------|----------|
| **OGD Crop Survey** | Village/hobli-level crop labels (coconut, pepper, coffee, cardamom) with GPS | JSON → CSV | `govt_crop_survey_data/` (already downloaded) |
| **DK_Features 2018–2024** | Environmental grid observations for Dakshina Kannada | CSV | `training/datasets/tabular/DK_Features_*.csv` |
| **Sentinel-2 (Kaggle)** | NDVI/EVI GeoTIFF imagery, 10m resolution, 2018–2025 | GeoTIFF | Downloaded at runtime via `kagglehub` |

### 3.2 Training Corpus (Frozen)

The frozen supervised corpus is **`crop_supervised_v2.csv`** (10,674 samples), stored under `govt_crop_matched_v2/` with its manifest `training_manifests/crop_supervised_v2.0_manifest.json`.

- **Split:** Spatial leave-one-taluk-out
  - Train: Belthangady, Mangalore, Bantwal
  - Val: Puttur
  - Test: Sullia
- **Classes:** coconut (6865), pepper (3695), coffee (101), cardamom (11), blackgram (2)
- **Features:** 28 tabular numeric + 4 categorical + Sentinel-2 imagery + temporal (year/season)

### 3.3 Yield Target

The final system returns **expected yield**. The yield target must NOT be `Yield_Proxy_NPP`. Instead the yield target comes from the **OGD crop survey crop extent / production data** (the `Crop_Extent` column referenced in the R5.8/R5.9 provenance contract) OR a yield regression head trained jointly with crop classification. **This is the single most important open design decision.**

---

## 4. Training Pipeline (Kaggle)

The training runs as a **Kaggle Kernel** pushed via **Kaggle CLI** (`kaggle kernels push`).

### 4.1 Orchestration Flow

```
scripts/run_full_pipeline.py           (local orchestrator)
  → pushes training/kaggle/notebooks/train.ipynb to Kaggle
  → mounts:
      - cropfusion-checkpoints         (private checkpoint dataset)
      - Sentinel-2 imagery dataset     (shathanandabhatn/...)
  → polls kernel status
  → downloads output (cropfusion_release/)
```

### 4.2 Inside the Kernel

```
training/kaggle/scripts/run_pipeline.py
  → frozen_corpus.py                    (load crop_supervised_v2.csv)
  → dataset_manager/                    (tabular provider + kaggle image provider)
  → stam/                               (attach imagery to observations)
  → preprocessing/                      (build tensors)
  → training/experiment.py              (orchestrate training)
    → models/cropfusion.py              (CropFusionModel)
    → training/trainer.py               (training loop)
    → training/evaluator.py             (evaluate on test taluk)
  → build_release.py                    (export cropfusion_release/)
```

---

## 5. Inference Pipeline (Backend)

### 5.1 Load Phase

```
application/backend/app/services/release_model_registry.py
  → inference_package/release/loader.py    (ReleasePackageLoader)
  → validates version/manifest/checksums
  → loads model (TorchScript preferred)
  → loads scaler, label_encoder, yield_scaler
  → loads historical_context, location_index (parquet)
  → loads metadata.db (village metadata)
```

### 5.2 Predict Phase

```
application/backend/app/modules/inference/service.py  (InferenceEngine)
  → inference/feature_builder.py       (build tabular + NDVI/EVI tensors)
  → inference/location_resolver.py     (nearest village in DK)
  → CropFusionModel.forward()          (crop probs + yield)
  → POST /predict returns:
      { recommended_crop, crop_probs, expected_yield, confidence }
```

### 5.3 API Endpoints (Final)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v1/predictions/predict` | Predict crop + yield for (lon, lat) |
| `GET` | `/api/v1/predictions/locations` | List served DK locations (map markers) |
| `GET` | `/api/v1/predictions/boundaries` | DK boundary geometries for map |
| `GET` | `/health` / `/ready` | Health probes |
| `GET` | `/model` | Model + dataset version info |

---

## 6. Frontend

### 6.1 Pages

| Route | Purpose |
|-------|---------|
| `/` | Landing page (brief project intro) |
| `/map` | **Main page** — DK-restricted Leaflet map, click location, select year/season, predict |
| `/predict` | Alt form-based prediction (manual coords) |

### 6.2 Key Components

- `MapView` — Leaflet map with DK boundaries
- `PredictionForm` — Location + Year + Season selection
- `PredictionCard` — Shows best crop + expected yield + confidence
- `CropComparison` — Top-3 crops side by side

---

## 7. Required Modules (Final Scope)

### 7.1 Training Platform — KEEP + REWRITE

| Module | Status | Notes |
|--------|--------|-------|
| `training/models/` | **KEEP** | CropFusion architecture |
| `training/preprocessing/` | **KEEP** (rewrite data_contract) | Tensor pipeline |
| `training/stam/` | **KEEP** (rewrite config) | STAM matching |
| `training/dataset_manager/` | **KEEP** | Dataset access + Kaggle provider |
| `training/training/` | **KEEP** | Training engine |
| `training/matching/` | **KEEP** | Spatial-tabular matcher |
| `training/inference/` | **KEEP** | Release packaging |
| `training/kaggle/` | **REWRITE** | Strip R5.x experiment scripts |
| `training/evaluation/` | **REWRITE** | Keep metrics/evaluator only |
| `training/feature_engineering/` | **DELETE** | Only used by export pipeline |
| `training/explainability/` | **DELETE** | Not required |
| `training/export/` | **DELETE** | Not required |
| `training/mlops/` | **DELETE** | Enterprise only |
| `training/runtime/` | **DELETE** | Backend uses inference_package directly |
| `training/quality/` | **DELETE** | Enterprise only |
| `training/feature_store/` | **DELETE** | Perf optimization only |
| `training/hyperparameter_search/` | **DELETE** | Not required |

### 7.2 Application — KEEP + REWRITE

| Module | Status | Notes |
|--------|--------|-------|
| `application/backend/` | **REWRITE** | FastAPI, no auth, simplified modules |
| `application/inference_package/release/` | **KEEP** | Core loader + manifest |
| `application/inference/models.py` | **KEEP** | DTOs |
| `application/gis/` (Taluk shapefiles) | **KEEP** | Map boundaries |
| `application/frontend/` | **REWRITE** | No auth, DK-only map |
| `application/inference/` (ports) | **DELETE** | ABCs not needed |
| `application/inference_package/manifest.py` | **DELETE** | R1.4 contract replaced |
| `application/database/` | **DELETE** | Enterprise DB not needed |
| `application/history/` | **DELETE** | No per-user history |
| `application/authentication/` | **DELETE** | No auth |
| `application/admin/` | **DELETE** | No admin |
| `application/monitoring/` | **DELETE** | Not needed |
| `application/docker/` | **REWRITE** | Simplify |
| `application/config/*.yaml` | **DELETE** | Use settings.yaml |

### 7.3 Shared Library — KEEP + REWRITE

| Module | Status | Notes |
|--------|--------|-------|
| `shared/enums/` | **KEEP** | Core enums |
| `shared/config/` | **KEEP** | Config helpers |
| `shared/exceptions/` | **KEEP** (trim unused) | Error base |
| `shared/logging/` | **KEEP** (trim audit) | Logging |
| `shared/versioning/` | **KEEP** | Versioning |
| `shared/utils/` | **KEEP** | Utilities |
| `shared/constants/` | **KEEP** | Constants |
| `shared/types/` | **KEEP** | Type aliases |
| `shared/schemas/` | **DELETE** | Unused at runtime |
| `shared/dto/` | **DELETE** | Empty placeholder |
| `shared/interfaces/` | **DELETE** | Unused at runtime |
| `shared/validation/` | **DELETE** | Over-engineered, inline |
| `shared/serialization/` | **DELETE** | Unused at runtime |

---

## 8. Removed Features (vs. Current)

| Current Feature | Final Status |
|-----------------|--------------|
| User login / registration / JWT | **REMOVED** |
| Per-user prediction history | **REMOVED** (or simplified to local history) |
| Admin dashboard | **REMOVED** |
| Explainability (GradCAM, SHAP, IG) | **REMOVED** (optional future) |
| Enterprise database layer (PostGIS/Alembic) | **REMOVED** (simple SQLite if needed) |
| MLOps registry/gates/drift | **REMOVED** |
| Prometheus/Grafana monitoring | **REMOVED** |
| Rate limiting | **REMOVED** |
| All R5.x experiment scripts/notebooks | **REMOVED** |
| All OGD discovery scripts | **ARCHIVED** (data already downloaded) |
| All legacy tabular sources (data_season, ICRISAT, etc.) | **REMOVED** |

---

## 9. Target Directory Layout (After Cleanup)

```
CropPrep/
├── shared/                          # Core shared library
│   ├── enums/
│   ├── config/
│   ├── exceptions/
│   ├── logging/
│   ├── versioning/
│   ├── utils/
│   ├── constants/
│   └── types/
├── training/
│   ├── models/
│   ├── preprocessing/
│   ├── stam/
│   ├── dataset_manager/
│   ├── training/
│   ├── matching/
│   ├── inference/
│   ├── evaluation/
│   ├── kaggle/
│   │   ├── scripts/                 # Run pipeline, evaluate, build release
│   │   ├── notebooks/               # train.ipynb, evaluate.ipynb, export.ipynb
│   │   ├── environment/
│   │   └── tests/
│   ├── config/                      # dataset, model, preprocessing, training, stam, seasons, kaggle, paths, logging
│   └── datasets/
│       └── tabular/                 # DK_Features_*.csv only
├── application/
│   ├── backend/                     # FastAPI (simplified, no auth)
│   ├── frontend/                    # React (no auth, DK map)
│   ├── inference_package/
│   │   └── release/                 # manifest.py + loader.py
│   ├── gis/
│   │   └── Taluk/                   # DK shapefiles
│   └── tests/
├── scripts/
│   ├── run_full_pipeline.py         # Kaggle orchestrator
│   ├── run_kaggle_notebook.py       # Kaggle CLI helper
│   ├── clean_dk_features.py         # DK feature cleaner
│   └── tests/
├── govt_crop_matched_v2/            # frozen corpus + provenance
├── training_manifests/              # dataset manifests
├── releases/
│   └── latest/                      # active cropfusion_release/
├── paper/                           # project paper
├── pyproject.toml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 10. Key Design Decisions (Open Questions)

### 10.1 Yield Target
**Question:** What is the exact yield target column? The OGD Crop Survey provides `Crop_Extent` (area classification per crop, per village). This is a *per-area* classification, not a direct yield per plant. Options:
- (A) Use `Crop_Extent` as a proxy and train a **yield regression head** on it jointly with the classification head.
- (B) Scale/model `Crop_Extent` as a continuous target.
- (C) Use the DK_Features `Yield_Proxy_NPP` *only as an input feature*, never as target (as currently handled by the leakage guard).

**Recommendation:** Train with a **multihead model** — crop classification head (softmax) + yield regression head (from the OGD Crop_Extent / field composition data). The yield head is what the user sees as "expected yield."

### 10.2 Season/Year Selection
The user selects a year/season in the frontend. But the pretrained model is trained on 2020–2021 observations. Options:
- (A) Accept year/season as **input features** to the model (temporal embedding).
- (B) Use year/season to **select the nearest imagery date** from Sentinel-2 for that location.
- (C) Both.

**Recommendation:** (C) — pass year/season as features AND use them to select the Sentinel-2 composite.

### 10.3 Frontend History Without Auth
Without login, there's no per-user history. Options:
- (A) No history at all (just compute and display).
- (B) LocalStorage-based history per browser.

**Recommendation:** (B) is a nice UX touch if desired; (A) is simplest.

---

## 11. Build & Run Commands (Target)

```bash
# Install
pip install -e ./shared ./training/models ./training/preprocessing ./training/stam ./training/dataset_manager ./training/training

# Train (local readiness check)
python training/kaggle/scripts/run_pipeline.py --repo-root . --years 2020,2021 --seasons Kharif,Rabi

# Train (Kaggle)
kaggle kernels push -p training/kaggle/notebooks/train

# Backend
python application/backend/run.py

# Frontend
cd application/frontend && npm run dev
```
