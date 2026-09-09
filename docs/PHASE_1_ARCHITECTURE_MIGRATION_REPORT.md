# Phase 1 — Architecture Migration Report

> **Generated:** 2026-09-09
> **Scope:** Map legacy architecture → new architecture; classify modules; create
> skeleton entry points and planning docs. **No training performed.**
> No excluded dataset was made a training source. No frozen corpus file was modified.

---

## 1. Objectives met

1. Mapped the legacy training chain dependencies (import/usage trace).
2. Produced the four planning docs:
   - `docs/CLEANUP_INVENTORY.md` (prior)
   - `docs/CLEAN_ARCHITECTURE.md` (prior)
   - `docs/NEW_TRAINING_ARCHITECTURE.md` (new)
   - `docs/LEGACY_TRAINING_MAP.md` (new)
   - `docs/YIELD_TARGET_ANALYSIS.md` (new, placeholder)
   - `docs/PHASE_1_ARCHITECTURE_MIGRATION_REPORT.md` (this file)
3. Created 5 skeleton entry points (no implementations, no training).
4. Left all frozen/required artifacts (`crop_supervised_v2.csv`, manifests,
   provenance, Sentinel-2, release `v2.0.0`, `paper/`, kept docs) untouched.

---

## 2. What will be REUSED (data / matching / provenance)

- `govt_crop_matched_v2/crop_supervised_v2.csv` + `provenance.json` (frozen corpus + provenance, required).
- `training_manifests/crop_supervised_v2.0_manifest.json` (split/class metadata, required).
- `training/datasets/tabular/DK_Features_*.csv` (tabular inputs).
- `training/matching/spatial_tabular_matcher.py` (OGD ↔ DK_Features K-NN/IDW, leakage-guarded).
- `training/dataset_manager/` providers (Kaggle image, image/CSV loaders, patch extractor).
- `training/stam/patch_generator.py`, `coordinate_transform.py`, `observation.py`, `name_aliases.py`,
  `observation_resolver.py` (`_TALUK_SPLIT`).
- `training/kaggle/frozen_corpus.py` data-loading role (kept), routing role retired.

## 3. What becomes LEGACY (retired from final flow)

- `training/kaggle/scripts/run_pipeline.py` (driver).
- `training/training/experiment.py` (orchestrator) + related trainer/evaluator/validator.
- `training/models/cropfusion.py` (`CropFusionModel`) + legacy encoders/fusion modules.
- `training/preprocessing/master_pipeline.py`/dataset/dataloader legacy tensor pipeline.
- `training/stam/stam.py` STAM facade (some sub-modules kept; see map).
- R5.x experiment scripts/notebooks and one-time audits.

## 4. What can be DELETED later

- R5.x R&D script/notebook campaigns, one-time audits, archived `scripts/ogd_*`.
- `training/{explainability,export,mlops,runtime,quality,feature_store,hyperparameter_search,experiments}/`.
- `shared/{schemas,dto,interfaces,serialization,validation}/` and unused exception/logging pieces.
- Removed-feature modules under `application/backend` and `application/frontend`.
- Excluded datasets themselves.

## 5. Current IMAGE-matching dependencies

- Sentinel-2 retrieved via `training/dataset_manager/providers/kaggle_image.py`
  (uses `downloader.py` + `image_loader.py`).
- Patches via `training/stam/patch_generator.py` + `coordinate_transform.py`
  (+ `training/dataset_manager/patch_extractor.py`).
- Image branch encoders (benchmarked): EfficientNetV2 / ConvNeXt / ResNet (Phase 2).

## 6. Current DATASET dependencies

- `frozen_corpus.py` → `crop_supervised_v2.csv` + manifest.
- Provenance from `training/matching/spatial_tabular_matcher.py` → `provenance.json`.
- Tabular features from `DK_Features_*.csv` via `dataset_manager` CSV provider.
- Split definition from `_TALUK_SPLIT` (Belthangady/Mangalore/Bantwal | Puttur | Sullia).

## 7. Current INFERENCE dependencies

- Backend: `application/backend/run.py → app/main.py → release_model_registry.py →
  app/modules/inference/service.py → inference_package/release/loader.py`.
- Loads release `releases/v2.0.0/` from `build_release.py` artifacts.
- Kept unchanged in Phase 1.

## 8. Exact files needed for Phase 2

- Frozen corpus + manifest + provenance (above).
- DK_Features CSVs.
- Sentinel-2 mapping + patch metadata.
- Benchmarked encoders (tabular: XGBoost/CatBoost/LightGBM; image: EfficientNetV2/ConvNeXt/ResNet).
- New model modules (tabular encoder, image encoder, fusion MLP, crop + yield heads).
- Entry points: `training/{prepare_data,train_tabular,train_image,train_fusion,evaluate}.py`.
- `training/inference/` + `build_release.py`/`package_sources.py` for release export.

## 9. Blockers (Phase 2, intentionally NOT edited in Phase 1)

1. **Yield target** — unresolved by design; see `docs/YIELD_TARGET_ANALYSIS.md`.
2. **Excluded-dataset references** that must be rewritten before clean training:
   - `training/preprocessing/data_contract.py` line 34 — `_KG_HA_SOURCES = ("data_season", "icrisat", ...)`.
   - `training/config/stam.yaml` lines ~84 and ~164 — `data_season.csv` and
     `ICRISAT-District Level Data.csv` table definitions.
   - `scripts/clean_dk_features.py` + `training/dataset_manager` fixtures referencing excluded sources.
3. **Legacy module removal** — deferred until the new architecture trains and evaluates successfully.

## 10. Final printed counts

- **Total files/paths inspected:** ~800+
- **KEEP:** ~210
- **REWRITE:** ~85
- **DELETE (safe later):** ~420
- **ARCHIVE:** ~15
- **UNCERTAIN / pending:** flags documented in `LEGACY_TRAINING_MAP.md`
- **Unresolved dependency questions (answered inline in CLEANUP_INVENTORY.md):** 8

## 11. Verification performed

- Confirmed no excluded dataset (`data_season.csv`, `ICRISAT…`, `cropdata_updated.csv`,
  `All-India…`, `dataset.csv`, `Yield_Proxy_NPP`) was added as a training source.
- Confirmed **no training was run** during Phase 1.
- Confirmed frozen corpus / manifests / provenance / Sentinel-2 / release `v2.0.0`
  were not modified or deleted.
