# Legacy Training Map — Module Classification

> **Generated:** 2026-09-09 (Phase 1)
> Classifies legacy architecture modules into the four Phase-1 buckets:
> **KEEP FOR DATA/MATCHING/PROVENANCE**, **LEGACY TRAINING**, **SAFE TO DELETE LATER**, **UNCERTAIN**.
> Derived from `docs/CLEANUP_INVENTORY.md` + a live import/usage trace.

---

## Bucket Legend

| Bucket | Meaning | Phase-1 action |
|--------|---------|----------------|
| KEEP FOR DATA/MATCHING/PROVENANCE | Supplies supervised data, matching, or provenance the new architecture reuses | Read-only, preserve |
| LEGACY TRAINING | Part of old `run_pipeline → … → CropFusionModel` chain | Retire from final flow (do not delete in Phase 1) |
| SAFE TO DELETE LATER | Obsolete, experiment-only, or clearly unused at runtime | Delete in a later phase |
| UNCERTAIN | Needs a decision before final classification | Flag; resolve later |

---

## 1. KEEP FOR DATA / MATCHING / PROVENANCE

These are the reusable data feeders. They are frozen re: the corpus.

| Module | Current role | Reuse in new architecture |
|--------|--------------|---------------------------|
| `training/kaggle/frozen_corpus.py` | `FrozenCorpusLoader` reads `crop_supervised_v2.csv` + manifest → `AgriculturalObservation` | **Data**: master dataset loader + split/class metadata (keep loader, retire its routing to legacy model) |
| `govt_crop_matched_v2/crop_supervised_v2.csv` | frozen enhanced corpus (10,674) | **Data** (required, do NOT delete) |
| `govt_crop_matched_v2/provenance.json` | per-record matching provenance | **Provenance** (required, do NOT delete) |
| `training_manifests/crop_supervised_v2.0_manifest.json` | split/class manifest | **Data** (required, do NOT delete) |
| `training/datasets/tabular/DK_Features_*.csv` | Earth Engine grid features | **Data**: tabular branch inputs |
| `training/matching/spatial_tabular_matcher.py` | K-NN / IDW OGD↔DK_Features matching (leakage-guards `Yield_Proxy_NPP`) | **Matching**: reuse in `prepare_data` |
| `training/dataset_manager/` | dataset access, providers, validators, metadata, registry | **Matching/data**: reuse Kaggle image provider, image loader, patch extractor, CSV loader |
| `training/dataset_manager/providers/kaggle_image.py` | Sentinel-2 retrieval | **Data**: image branch source |
| `training/stam/patch_generator.py` + `coordinate_transform.py` | raster patch extraction / CRS | **Data**: image branch patch extraction |
| `training/stam/observation.py` | `AgriculturalObservation` data model | **Data**: shared record type |
| `training/stam/name_aliases.py` | district/place-name normalization | **Data**: location resolution |
| `training/stam/observation_resolver.py` | `ObservationResolver`, `ObservationCorpus`, `_TALUK_SPLIT` | **Data**: spatial split definition (reuse `_TALUK_SPLIT` for train/val/test) |

---

## 2. LEGACY TRAINING (retire from final flow)

The `run_pipeline → experiment → CropFusionModel` chain. Not used to train the
final model, but kept read-only in Phase 1.

| Module | Legacy role | Phase-1 action |
|--------|-------------|----------------|
| `training/kaggle/scripts/run_pipeline.py` | main legacy driver | Retire |
| `training/training/experiment.py` | experiment orchestrator | Retire |
| `training/training/trainer.py`, `evaluator.py`, `validator.py`, `cropfusion_trainer.py` | legacy training loop/eval | Retire from final (some metrics may be reused) |
| `training/models/cropfusion.py` (`CropFusionModel`) | legacy multimodal architecture | Retire; replaced by new tabular/image/fusion modules |
| `training/models/` (backbone, ndvi_encoder, evi_encoder, cross_attention, adaptive_gate, fusion_engine, temporal_transformer, tabtransformer, tabular, image_fusion, shared_encoder, multitask_heads, losses) | legacy encoders/fusion | Retire from final flow (encoder building blocks may be ported in Phase 2) |
| `training/kaggle/enhanced_observation_generator.py` | R5.2.9 env-feature matching | Retire (its matcher is reused from `training/matching/`) |
| `training/preprocessing/master_pipeline.py`, `dataset.py`, `dataloader.py`, `image_pipeline.py`, `temporal_pipeline.py`, `label_pipeline.py`, `transforms.py`, `augmentations.py` | legacy tensor pipeline | Retire from final (some transforms reused) |
| `training/stam/stam.py`, `matcher.py`, `historical_context.py`, `sequence_builder.py`, `temporal_index.py`, `temporal_window.py` | legacy STAM alignment | Retire from final (patch/coordinate + name_aliases kept) |
| `training/kaggle/scripts/build_release.py` | release packaging | **Inherit** for Phase-2 release export (not legacy-dead) |
| `training/kaggle/scripts/package_sources.py` | train-side release sources | **Inherit** for Phase-2 release |

---

## 3. SAFE TO DELETE LATER

Obsolescent, experiment-only, or unused-at-runtime (per `CLEANUP_INVENTORY.md`).
**Not deleted in Phase 1** — deleted only after the new architecture is proven.

| Module / path | Reason |
|---------------|--------|
| R5.x experiment scripts: `r5_5_*`, `r5_6_*`, `r5_7_data_recovery.py`, `r5_8_subfield.py`, `r5_9_field_composition.py`, `r5_10_temporal_field_target.py`, `run_baselines.py`, `train_feature_fusion.py`, `feature_fusion_utils.py`, `final_cropfusion.py`, `extract_image_embeddings.py` | R5 experiment campaign only |
| One-time audits: `class_retention_audit.py`, `corpus_delta_audit.py`, `corpus_diagnostics.py`, `imagery_availability_report.py`, `matching_verification_audit.py`, `model_input_validation.py`, `diagnose_*.py`, `validation_numerics_probe.py`, `_dump_logs.py`, `_extract_v4/v7.py` | one-time diagnostics |
| R5 R&D scripts in `training/kaggle/deployment/`, `R5_*.ipynb`, `kernel-metadata.json`, `kaggle_pull/` | R5 campaign |
| `scripts/kaggle_r5_*.py`, `scripts/r5_2_*.py`, `scripts/ogd_*` (archived), `scripts/build_docs.py`, `scripts/backup/` | R5 / archived / unused |
| `training/explainability/`, `training/export/`, `training/mlops/`, `training/runtime/`, `training/quality/`, `training/feature_store/`, `training/hyperparameter_search/`, `training/experiments/` | not required for academic project |
| `shared/schemas/`, `shared/dto/`, `shared/interfaces/`, `shared/serialization/`, `shared/validation/`, `shared/exceptions/{security,logging}.py`, `shared/logging/audit.py` | unused at runtime |
| `training/config/performance.yaml`, `feature_fusion.yaml`, `r5_2_9_matching.yaml` | experiment configs |
| `training/artifacts/` | generated, not source |
| `application/backend` auth/admin/database/monitoring, `application/modules/{auth,users,admin,history,dataset,configuration,monitoring,explainability}`, `application/frontend` auth/history pages | removed features (see CLEAN_ARCHITECTURE) |

Excluded datasets (`data_season.csv`, `ICRISAT…`, `cropdata_updated.csv`,
`All-India…`, `dataset.csv`) are **to be removed** from `training/datasets/tabular/`
in a later phase and **must not** become training sources.

---

## 4. UNCERTAIN

Items needing a decision before final classification.

| Module | Why uncertain | Decision needed |
|--------|---------------|-----------------|
| `training/evaluation/evaluator.py`, `metrics.py` | kept core metric logic used by the legacy evaluator; new `evaluate.py` needs metrics | Reuse metrics/evaluator core vs. rewrite in Phase 2 |
| `training/training/config.py`, `losses.py`, `optimizers.py`, `schedulers.py`, `checkpoint.py`, `callbacks.py`, `diagnostics.py` | legacy training engine pieces; fusion trainer may reuse optimizers/checkpoint | Adopt vs. replace in Phase 2 |
| `training/models/config.py`, `runtime.py`, `checkpoint.py`, `validators.py`, `exporter.py`, `utils.py` | model-building building blocks that the new modules may reuse | Port vs. rewrite in Phase 2 |
| `training/inference/` | release packaging (used by `build_release.py`) — likely keep | Confirm as release toolchain |
| `training/kaggle/validation.py` | env gates (uses `shared.validation`) | Rewrite/inline as needed |
| `training/stam/` non-kept modules (season_resolver, spatial_index, cache, profiler, gates, validators) | some may support tabular/temporal branch | Decide per module in Phase 2 |
| Excluded-dataset references in `training/preprocessing/data_contract.py` (`_KG_HA_SOURCES` line 34) and `training/config/stam.yaml` (lines ~84, ~164) | they reference removed datasets; editing is a **Phase 2 blocker**, not Phase 1 | Rewrite in Phase 2 |

---

## 5. Summary Counts (Phase 1)

- **Total modules/paths inspected:** ~800+ (from `CLEANUP_INVENTORY.md`)
- **KEEP (data/matching/provenance + core):** ~210
- **REWRITE:** ~85
- **DELETE (safe later):** ~420
- **ARCHIVE:** ~15
- **UNCERTAIN / pending decision:** flagged above
- **Unresolved dependency questions (answered inline in CLEANUP_INVENTORY.md):** 8
