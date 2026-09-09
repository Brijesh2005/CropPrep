# CropPrep/CropFusion Cleanup Inventory

> **Generated:** 2026-09-09
> **Goal:** Simplify into "Hybrid Approach to Crop Yield Forecasting: Integrating Weather Data and Satellite Imagery with Machine Learning"
> **Supervised sources:** Karnataka OGD Crop Survey + DK_Features + Sentinel-2 (Kaggle)
> **Excluded datasets:** data_season.csv, ICRISAT, cropdata_updated.csv, All-India, dataset.csv
> **Training platform:** Kaggle CLI only
> **Frontend:** No auth, interactive map (DK only), predict crop + yield

---

## Current Working Training Path

The **active training pipeline** flows through:

```
training/kaggle/scripts/run_pipeline.py          (main driver)
  → training/kaggle/frozen_corpus.py             (load crop_supervised_v2.csv + manifest)
  → training/dataset_manager/                    (tabular + Kaggle image providers)
  → training/stam/                               (spatial-temporal alignment)
  → training/preprocessing/                      (tensor building)
  → training/training/experiment.py              (orchestrator)
    → training/models/cropfusion.py              (CropFusionModel)
    → training/training/trainer.py               (training loop)
    → training/training/evaluator.py             (evaluation)
  → training/kaggle/scripts/build_release.py     (export release package)
```

The **current inference path** flows through:

```
application/backend/run.py → app/main.py
  → app/services/release_model_registry.py      (loads cropfusion_release/ package)
  → app/modules/inference/service.py            (InferenceEngine)
  → app/modules/inference/feature_builder.py    (builds tensors from release artifacts)
  → app/modules/inference/location_resolver.py  (nearest village + context)
  → inference_package/release/loader.py         (ReleasePackageLoader)
```

---

## Master Table

### Legend
- **KEEP** = Required for final project as-is
- **REWRITE** = Required but needs significant modification
- **DELETE** = Safe to remove (legacy/obsolete/unrelated)
- **CONDITIONAL** = Keep if specific condition met, otherwise delete

---

### 1. Top-Level Project Files

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `README.md` | Project overview | Project overview | **REWRITE** | none | Needs simplification to match new scope |
| `pyproject.toml` | Monorepo build config | Simplified build | **REWRITE** | setuptools | Remove 8 sub-package builds, simplify to 3 packages |
| `pytest.ini` | Test config | Test config | **REWRITE** | none | Remove integration/quality markers, simplify |
| `Makefile` | Dev shortcuts | Dev shortcuts | **REWRITE** | pip | Simplify targets |
| `requirements.txt` | Pinned deps | Pinned deps | **REWRITE** | none | Trim to actual deps |
| `requirements-dev.txt` | Dev deps | Dev deps | **REWRITE** | none | Trim to actual deps |
| `environment.yml` | Conda env | Conda env | **DELETE** | none | Redundant with requirements.txt |
| `.editorconfig` | Editor config | Editor config | **KEEP** | none | Standard |
| `.gitignore` | Git ignores | Git ignores | **REWRITE** | none | Remove R5.x artifact patterns |
| `VERSION` | Version file | Version file | **KEEP** | none | Single version |
| `LICENSE` | License | License | **KEEP** | none | Standard |
| `CODE_OF_CONDUCT.md` | CoC | CoC | **KEEP** | none | Standard |
| `SECURITY.md` | Security policy | Security policy | **DELETE** | none | Not needed for academic project |
| `.env` | Environment vars | Environment vars | **KEEP** | none | Runtime config |
| `r5-6-image-statistics-export.log` | Old log | — | **DELETE** | none | Stale artifact |

---

### 2. `shared/` — Core Contract Library

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `shared/__init__.py` | Package root | Package root | **KEEP** | none | Version + umbrella |
| `shared/enums/__init__.py` | `CropType`, `Season`, etc. | Core enums | **KEEP** | none | Used by everything |
| `shared/enums/crop_taxonomy.py` | OGD label resolution | OGD label resolution | **KEEP** | shared.enums | Used by corpus pipeline |
| `shared/constants/__init__.py` | Framework constants | Core constants | **KEEP** | none | Used by config loaders |
| `shared/types/__init__.py` | Shared type aliases | Type aliases | **KEEP** | none | Used by training modules |
| `shared/config/__init__.py` + `loader.py` | `deep_merge`, `parse_env`, `apply_case_insensitive` | Config utilities | **KEEP** | none | Used by all 12 config loaders |
| `shared/config/__init__.py` | Re-exports | Re-exports | **KEEP** | .loader | Convenience |
| `shared/schemas/__init__.py` | Validation schemas | Schemas | **DELETE** | none | Only used by shared/tests, not imported by any runtime code |
| `shared/dto/__init__.py` | Placeholder DTOs | — | **DELETE** | none | Empty placeholder |
| `shared/exceptions/__init__.py` | `CropFusionError` base | Error base | **KEEP** | none | Imported by all training/*/exceptions.py |
| `shared/exceptions/base.py` | Base error | Error base | **KEEP** | none | — |
| `shared/exceptions/config.py` | Config errors | Config errors | **KEEP** | .base | — |
| `shared/exceptions/data.py` | Data errors | Data errors | **KEEP** | .base | — |
| `shared/exceptions/io.py` | IO errors | IO errors | **KEEP** | .base | — |
| `shared/exceptions/logging.py` | Logging errors | Logging errors | **DELETE** | .base | Not imported by runtime |
| `shared/exceptions/model.py` | Model errors | Model errors | **KEEP** | .base | — |
| `shared/exceptions/prediction.py` | Prediction errors | Prediction errors | **KEEP** | .base | — |
| `shared/exceptions/security.py` | Security errors | — | **DELETE** | .base | Auth removed from final |
| `shared/exceptions/validation.py` | Validation errors | Validation errors | **KEEP** | .base | — |
| `shared/exceptions/versioning.py` | Version errors | Version errors | **KEEP** | .base | — |
| `shared/interfaces/__init__.py` + `data.py` + `platform.py` + `providers.py` | ABCs | ABCs | **DELETE** | none | Only imported by shared/tests, not runtime |
| `shared/logging/__init__.py` + `audit.py` + `formatters.py` + `setup.py` | Logging setup | Logging | **KEEP** | .formatters | Used by 8+ logger.py modules |
| `shared/logging/audit.py` | Audit logging | — | **DELETE** | none | Enterprise-only |
| `shared/validation/__init__.py` + `base.py` + `registry.py` + `validators.py` | Validation framework | Validation | **DELETE** | shared.exceptions | Only used by shared/tests + training/kaggle/validation.py (can inline) |
| `shared/serialization/__init__.py` + `formats.py` + `registry.py` | Serialization | — | **DELETE** | none | Only used by shared/tests |
| `shared/versioning/__init__.py` + `semver.py` + `versions.py` + `provider.py` | Semantic versioning | Versioning | **KEEP** | shared.exceptions | Used by inference, runtime, kaggle checkpoints |
| `shared/utils/__init__.py` + 10 modules | Utility functions | Utilities | **KEEP** | none | `yaml_safe`, `sha256_file`, etc. used everywhere |
| `shared/Checkpoint` | Changelog file | — | **DELETE** | none | Not code |
| `shared/tests/` (11 test files) | Shared lib tests | — | **DELETE** | none | Rebuild tests for kept modules only |

---

### 3. `training/models/` — CropFusion Neural Architecture

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/models/__init__.py` | Package exports | Package exports | **KEEP** | all below | — |
| `training/models/pyproject.toml` | Package build | Package build | **REWRITE** | setuptools | Simplify deps |
| `training/models/config.py` | Model config (pydantic) | Model config | **KEEP** | shared.config | — |
| `training/models/cropfusion.py` | **CropFusionModel** | **Core model** | **KEEP** | all model submodules | Main architecture |
| `training/models/factory.py` | ModelFactory | Model factory | **KEEP** | .config, .runtime, .checkpoint | Used by training + inference fallback |
| `training/models/backbone.py` | TimmImageEncoder | Image backbone | **KEEP** | timm, torch | — |
| `training/models/tabtransformer.py` | TabTransformer | Tabular encoder | **KEEP** | .config | — |
| `training/models/temporal_transformer.py` | Temporal transformer | Temporal encoder | **KEEP** | .config | — |
| `training/models/cross_attention.py` | Cross-modal attention | Attention | **KEEP** | torch | — |
| `training/models/adaptive_gate.py` | Gated fusion | Gating | **KEEP** | torch | — |
| `training/models/fusion_engine.py` | CrossModalFusionEngine | Fusion | **KEEP** | .adaptive_gate, .cross_attention | — |
| `training/models/ndvi_encoder.py` | NDVI encoder | NDVI encoder | **KEEP** | .backbone | — |
| `training/models/evi_encoder.py` | EVI encoder | EVI encoder | **KEEP** | .backbone | — |
| `training/models/image_fusion.py` | NDVI+EVI fusion | Image fusion | **KEEP** | torch | — |
| `training/models/shared_encoder.py` | Shared encoder | Shared encoder | **KEEP** | .config | — |
| `training/models/multitask_heads.py` | Crop + yield heads | Output heads | **KEEP** | torch | — |
| `training/models/losses.py` | Multi-task losses | Losses | **KEEP** | .config | — |
| `training/models/interfaces.py` | Abstract encoders | ABCs | **KEEP** | torch | — |
| `training/models/runtime.py` | Device/AMP context | Device mgmt | **KEEP** | torch | — |
| `training/models/checkpoint.py` | Checkpoint save/load | Checkpointing | **KEEP** | torch | — |
| `training/models/exporter.py` | TorchScript/ONNX export | Export | **KEEP** | torch, onnx | — |
| `training/models/validators.py` | Shape validation | Validation | **KEEP** | torch | — |
| `training/models/utils.py` | Activations, positional enc | Utilities | **KEEP** | torch | — |
| `training/models/exceptions.py` | Model errors | Errors | **KEEP** | shared.exceptions | — |
| `training/models/feature_fusion.py` | Feature-level fusion | — | **DELETE** | torch | R5.9/R5.10 experiment, not core architecture |

---

### 4. `training/preprocessing/` — Tensor Pipeline

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/preprocessing/__init__.py` | Package exports | Exports | **KEEP** | all below | — |
| `training/preprocessing/pyproject.toml` | Package build | Build | **KEEP** | setuptools | — |
| `training/preprocessing/config.py` | PreprocessingConfig | Config | **KEEP** | shared.config | — |
| `training/preprocessing/master_pipeline.py` | **Preprocessor** orchestrator | Core pipeline | **KEEP** | .augmentations, stam.observation | Main entry |
| `training/preprocessing/dataset.py` | CropFusionDataset | Dataset class | **KEEP** | .config, .master_pipeline | — |
| `training/preprocessing/dataloader.py` | DataLoader builder | DataLoader | **KEEP** | .config, .dataset | — |
| `training/preprocessing/tabular_pipeline.py` | Tabular features | Tabular | **KEEP** | .config, .transforms | — |
| `training/preprocessing/image_pipeline.py` | Image tensors | Images | **KEEP** | .config | — |
| `training/preprocessing/temporal_pipeline.py` | Temporal sequences | Temporal | **KEEP** | .config | — |
| `training/preprocessing/label_pipeline.py` | Label encoding | Labels | **KEEP** | .config, .transforms | — |
| `training/preprocessing/transforms.py` | Scalers, label encoder | Transforms | **KEEP** | numpy | — |
| `training/preprocessing/augmentations.py` | Image augmentation | Augmentation | **KEEP** | .config | — |
| `training/preprocessing/statistics.py` | Dataset stats | Stats | **KEEP** | numpy | — |
| `training/preprocessing/utils.py` | Tensor helpers | Utilities | **KEEP** | numpy | — |
| `training/preprocessing/validators.py` | Quality validation | Validation | **KEEP** | .config | — |
| `training/preprocessing/data_contract.py` | Yield unit contract | Contract | **REWRITE** | none | Remove `data_season.csv` string match in `_KG_HA_SOURCES` |
| `training/preprocessing/supervised_contract.py` | Supervised hash/dedup | Contract | **KEEP** | .data_contract | — |
| `training/preprocessing/interfaces.py` | Pipeline ABCs | ABCs | **KEEP** | none | — |
| `training/preprocessing/exceptions.py` | Errors | Errors | **KEEP** | shared.exceptions | — |
| `training/preprocessing/logger.py` | Logger | Logger | **KEEP** | shared.logging | — |

---

### 5. `training/stam/` — Spatial-Temporal Alignment Module

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/stam/__init__.py` | Package exports | Exports | **KEEP** | all below | — |
| `training/stam/pyproject.toml` | Package build | Build | **KEEP** | setuptools | — |
| `training/stam/config.py` | StamConfig | Config | **KEEP** | pydantic, yaml | — |
| `training/stam/stam.py` | **STAM** orchestration facade | Core STAM | **KEEP** | all stam modules + dataset_manager | Main entry |
| `training/stam/observation.py` | AgriculturalObservation | Data model | **KEEP** | pydantic | — |
| `training/stam/observation_resolver.py` | Observation resolution | Resolver | **KEEP** | .exceptions | — |
| `training/stam/matcher.py` | Spatial/temporal matching | Matcher | **KEEP** | dataset_manager | — |
| `training/stam/historical_context.py` | Historical context builder | Context | **KEEP** | dataset_manager, .season_resolver | — |
| `training/stam/season_resolver.py` | Season year resolution | Season resolver | **KEEP** | .config | — |
| `training/stam/sequence_builder.py` | Observation sequences | Sequences | **KEEP** | .observation, .temporal_index | — |
| `training/stam/temporal_index.py` | Temporal index/bucketing | Temporal index | **KEEP** | .config | — |
| `training/stam/temporal_window.py` | Seasonal window | Window | **KEEP** | calendar | — |
| `training/stam/coordinate_transform.py` | CRS/projection | CRS | **KEEP** | pyproj, rasterio | — |
| `training/stam/patch_generator.py` | Raster patch generation | Patch gen | **KEEP** | rasterio, .coordinate_transform | — |
| `training/stam/spatial_index.py` | STRtree spatial index | Spatial index | **KEEP** | shapely | — |
| `training/stam/name_aliases.py` | District name normalization | Name utils | **KEEP** | none | — |
| `training/stam/cache.py` | STAM cache | Cache | **KEEP** | .interfaces | — |
| `training/stam/interfaces.py` | ABCs | ABCs | **KEEP** | .observation | — |
| `training/stam/validators.py` | Quality assessment | Validators | **KEEP** | .config, .observation | — |
| `training/stam/tabular_profiler.py` | Tabular column profiler | Profiler | **KEEP** | .name_aliases, pandas | — |
| `training/stam/tabular_gates.py` | Tabular quality gating | Gates | **KEEP** | none | — |
| `training/stam/exceptions.py` | Errors | Errors | **KEEP** | shared.exceptions | — |
| `training/stam/logger.py` | Logger | Logger | **KEEP** | shared.logging | — |
| `training/stam/seasons.yaml` | Season calendar (copy) | — | **DELETE** | none | Duplicate of training/config/seasons.yaml |

---

### 6. `training/dataset_manager/` — Dataset Access Layer

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/dataset_manager/__init__.py` | Package exports | Exports | **KEEP** | all below | — |
| `training/dataset_manager/pyproject.toml` | Package build | Build | **KEEP** | setuptools | — |
| `training/dataset_manager/config.py` | Settings (dataset.yaml) | Config | **KEEP** | pydantic, yaml | — |
| `training/dataset_manager/manager.py` | **DatasetManager** facade | Core manager | **KEEP** | all providers | Main entry |
| `training/dataset_manager/manager_paths.py` | Path resolution | Paths | **KEEP** | .config | — |
| `training/dataset_manager/csv_loader.py` | CSV loading | CSV loader | **KEEP** | pandas | — |
| `training/dataset_manager/image_loader.py` | Rasterio image loader | Image loader | **KEEP** | rasterio | — |
| `training/dataset_manager/downloader.py` | Kaggle dataset downloader | Downloader | **KEEP** | kagglehub | — |
| `training/dataset_manager/scanner.py` | Directory scanner | Scanner | **KEEP** | .cache_manager | — |
| `training/dataset_manager/validator.py` | Dataset integrity validation | Validator | **KEEP** | .config | — |
| `training/dataset_manager/cache_manager.py` | SQLite cache | Cache | **KEEP** | sqlite3 | — |
| `training/dataset_manager/metadata.py` | Metadata generation + SQLite store | Metadata | **KEEP** | pandas | — |
| `training/dataset_manager/metadata_repository.py` | Metadata CRUD | Metadata repo | **KEEP** | ._db | — |
| `training/dataset_manager/dataset_registry.py` | Dataset versioning | Registry | **KEEP** | ._db | — |
| `training/dataset_manager/version_manager.py` | Version entries | Versions | **KEEP** | ._db | — |
| `training/dataset_manager/patch_extractor.py` | Raster patch extraction | Patch extractor | **KEEP** | rasterio | — |
| `training/dataset_manager/spatial_index.py` | Spatial records index | Spatial index | **KEEP** | numpy | — |
| `training/dataset_manager/statistics.py` | Dataset stats | Stats | **KEEP** | — | — |
| `training/dataset_manager/reports.py` | Dataset reports | Reports | **KEEP** | — | — |
| `training/dataset_manager/models.py` | Dataclasses + enums | Models | **KEEP** | shared.enums | — |
| `training/dataset_manager/interfaces.py` | ABCs | ABCs | **KEEP** | — | — |
| `training/dataset_manager/utils.py` | Hashing, geotiff detect | Utilities | **KEEP** | — | — |
| `training/dataset_manager/exceptions.py` | Errors | Errors | **KEEP** | shared.exceptions | — |
| `training/dataset_manager/logger.py` | Logger | Logger | **KEEP** | shared.logging | — |
| `training/dataset_manager/cli.py` | CLI entrypoint | CLI | **REWRITE** | .manager | Simplify for DK-only |
| `training/dataset_manager/manage_dataset.py` | CLI shim | CLI shim | **KEEP** | .cli | — |
| `training/dataset_manager/_db.py` | SQLite helpers | DB utils | **KEEP** | — | — |
| `training/dataset_manager/providers/__init__.py` | Provider registry | Providers | **KEEP** | all providers | — |
| `training/dataset_manager/providers/base.py` | Abstract providers | ABCs | **KEEP** | .models | — |
| `training/dataset_manager/providers/models.py` | Provider models | Models | **KEEP** | ..models | — |
| `training/dataset_manager/providers/git_tabular.py` | Git-versioned tabular provider | Tabular provider | **KEEP** | ..csv_loader | — |
| `training/dataset_manager/providers/kaggle_image.py` | **KaggleHub image provider** | Image provider | **KEEP** | ..downloader, ..image_loader | Core for Sentinel-2 |
| `training/dataset_manager/historical_context_builder.py` | Context builder | Context | **KEEP** | .metadata_repository | — |

---

### 7. `training/training/` — Training Engine

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/training/__init__.py` | Package exports | Exports | **KEEP** | all below | — |
| `training/training/pyproject.toml` | Package build | Build | **KEEP** | setuptools | — |
| `training/training/config.py` | TrainingConfig | Config | **KEEP** | shared.config | — |
| `training/training/experiment.py` | **Experiment** orchestrator | Core experiment | **KEEP** | models, preprocessing | Main entry |
| `training/training/trainer.py` | **Trainer** training loop | Core trainer | **KEEP** | torch | — |
| `training/training/evaluator.py` | Evaluation loop | Evaluator | **KEEP** | .metrics | — |
| `training/training/validator.py` | Validation + CV folds | Validator | **KEEP** | .metrics, .diagnostics | — |
| `training/training/cropfusion_trainer.py` | High-level orchestrator | Orchestrator | **KEEP** | .trainer, .checkpoint | — |
| `training/training/losses.py` | MultiTaskLoss wrapper | Losses | **KEEP** | training.models.losses | — |
| `training/training/metrics.py` | Metric trackers | Metrics | **KEEP** | numpy, torch | — |
| `training/training/optimizers.py` | Optimizer factory | Optimizers | **KEEP** | torch | — |
| `training/training/schedulers.py` | LR scheduler factory | Schedulers | **KEEP** | torch | — |
| `training/training/checkpoint.py` | Training checkpoint manager | Checkpointing | **KEEP** | training.models.checkpoint | — |
| `training/training/callbacks.py` | History recorder, fine-tuning | Callbacks | **KEEP** | .interfaces | — |
| `training/training/curriculum.py` | Curriculum learning | Curriculum | **KEEP** | .config | — |
| `training/training/diagnostics.py` | NaN hooks, batch asserts | Diagnostics | **KEEP** | torch | — |
| `training/training/benchmark.py` | Throughput benchmark | Benchmark | **KEEP** | torch | — |
| `training/training/visualizer.py` | Loss/regression plots | Visualizer | **KEEP** | matplotlib | — |
| `training/training/reports.py` | Report rendering | Reports | **KEEP** | .config | — |
| `training/training/logger.py` | Experiment logger | Logger | **KEEP** | — | — |
| `training/training/profiler.py` | Training profiler | Profiler | **DELETE** | none | Optional perf tool, not needed |
| `training/training/interfaces.py` | Callback ABCs | ABCs | **KEEP** | — | — |
| `training/training/exceptions.py` | Errors | Errors | **KEEP** | shared.exceptions | — |
| `training/training/utils.py` | Device/git/param utils | Utilities | **KEEP** | torch | — |

---

### 8. `training/feature_engineering/` — Feature Engineering

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/feature_engineering/` (all files) | Feature engineering pipeline | Feature engineering | **REWRITE** | shared.config, pandas | Used by export module; needs simplification for DK-only features |

---

### 9. `training/explainability/` — MXAI Explainability

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/explainability/` (all files) | GradCAM, SHAP, IG, counterfactual, etc. | Explainability | **DELETE** | torch, shap, matplotlib | **Not required** for final-year project (can add later) |

---

### 10. `training/export/` — Data Export

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/export/` (all files) | JSON/Parquet/Torch export | Data export | **DELETE** | pandas, pyarrow | Not needed — inference uses release package directly |

---

### 11. `training/mlops/` — MLOps

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/mlops/` (all files) | Experiment tracking, gates, registry, drift | MLOps | **DELETE** | training.quality | Enterprise MLOps, not needed for academic project |

---

### 12. `training/inference/` — Inference Packaging

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/inference/` (all files) | Release package builder, versioning, validation | Inference packaging | **KEEP** | shared.versioning, training.models | Used by build_release.py and backend loader |

---

### 13. `training/runtime/` — Runtime Release Manager

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/runtime/` (all files) | Release layout, model/preprocess loading, runtime cache | Runtime manager | **DELETE** | training.models, training.inference | Enterprise runtime; backend uses inference_package directly |

---

### 14. `training/quality/` — Quality Gates

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/quality/drift/` (all files) | Drift detection | Drift | **DELETE** | scipy | Not needed |
| `training/quality/fairness/` (all files) | Fairness evaluation | Fairness | **DELETE** | sklearn | Not needed |
| `training/quality/monitoring/` (all files) | Monitoring dashboards | Monitoring | **DELETE** | — | Not needed |
| `training/quality/optimization/` (all files) | ONNX optimization | Optimization | **DELETE** | onnxruntime | Not needed |
| `training/quality/samples/` (all files) | Sample quality reports | Reports | **DELETE** | — | Not needed |

---

### 15. `training/matching/` — Spatial-Tabular Matcher

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/matching/spatial_tabular_matcher.py` | IDW/KNN grid matching (R5.2.9) | Matcher | **KEEP** | scipy, shared.logging | Used by enhanced_observation_generator + r5_7 |

---

### 16. `training/evaluation/` — Evaluation Framework

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/evaluation/` (all files) | Ablation, comparison, error analysis | Evaluation | **REWRITE** | training.models, torch | Simplify — keep metrics/evaluator, remove ablation/comparison |

---

### 17. `training/feature_store/` — Performance Cache

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/feature_store/` (all files) | Image/tabular/temporal/patch caching | Feature cache | **DELETE** | — | Performance optimization, not needed for project |

---

### 18. `training/hyperparameter_search/` — HPO

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/hyperparameter_search/` | Hyperparameter search | HPO | **DELETE** | — | Not needed |

---

### 19. `training/experiments/` — Experiment Tracking

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/experiments/` | Experiment tracking | Experiments | **DELETE** | — | Not needed |

---

### 20. `training/kaggle/` — Kaggle Orchestration

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/kaggle/__init__.py` | Package init | Init | **KEEP** | — | — |
| `training/kaggle/config.py` | PathsConfig/KaggleConfig | Config | **KEEP** | shared.config | — |
| `training/kaggle/frozen_corpus.py` | **Frozen corpus loader** | Core loader | **KEEP** | — | Loads crop_supervised_v2.csv |
| `training/kaggle/validation.py` | TrainingValidator env gates | Validation | **KEEP** | shared.enums, shared.validation | — |
| `training/kaggle/workspace.py` | WorkspaceManager | Workspace | **KEEP** | .cache, .checkpoints | — |
| `training/kaggle/cache.py` | TrainingCache | Cache | **KEEP** | — | — |
| `training/kaggle/checkpoints.py` | Kaggle CheckpointManager | Checkpoints | **KEEP** | shared.versioning | — |
| `training/kaggle/logging.py` | TrainingLogger | Logger | **KEEP** | shared.logging | — |
| `training/kaggle/reports.py` | Pipeline reports | Reports | **KEEP** | .config | — |
| `training/kaggle/setup.py` | setuptools config | Setup | **KEEP** | setuptools | — |
| `training/kaggle/enhanced_observation_generator.py` | R5.2.9 env-feature matching | Enhanced obs gen | **REWRITE** | matching.spatial_tabular_matcher | Needs DK-only adaptation |
| `training/kaggle/scripts/bootstrap.py` | Clone + install editable pkgs | Bootstrap | **KEEP** | .config, .environment, .workspace | Kaggle kernel setup |
| `training/kaggle/scripts/run_pipeline.py` | **Main pipeline driver** | Core pipeline | **KEEP** | training.* | Primary entry point |
| `training/kaggle/scripts/run_training.py` | Training launcher | Training launcher | **KEEP** | training.* | — |
| `training/kaggle/scripts/evaluate.py` | Model evaluation | Evaluator | **KEEP** | training.* | — |
| `training/kaggle/scripts/build_release.py` | Release packaging | Release builder | **KEEP** | training.models, training.inference | — |
| `training/kaggle/scripts/backfill_release.py` | Backfill release | Release backfill | **REWRITE** | training.* | Remove data_season.csv default path |
| `training/kaggle/scripts/package_sources.py` | Package sources | Source packager | **KEEP** | training.dataset_manager | — |
| `training/kaggle/scripts/export_release.py` | Export model | Exporter | **KEEP** | training.models | — |
| `training/kaggle/environment/` (5 files) | GPU/runtime/dep detection | Environment | **KEEP** | — | — |
| `training/kaggle/scripts/verify_*.py` (12 files) | Verification scripts | Verification | **KEEP** | training.* | Sanity checks |
| `training/kaggle/scripts/system_check.py` | System check | System check | **KEEP** | — | — |
| `training/kaggle/scripts/smoke_test_full_pipeline.py` | Full pipeline smoke test | Smoke test | **KEEP** | training.* | — |
| `training/kaggle/scripts/r5_5_*.py` (3 files) | Collapse diagnostics | Diagnostics | **DELETE** | training.* | R5.5 experiment only |
| `training/kaggle/scripts/r5_6_*.py` (2 files) | Image stats/separability | Analysis | **DELETE** | training.* | R5.6 experiment only |
| `training/kaggle/scripts/r5_7_data_recovery.py` | Data recovery | Recovery | **DELETE** | training.matching | R5.7 experiment only |
| `training/kaggle/scripts/r5_8_subfield.py` | Subfield analysis | Analysis | **DELETE** | — | R5.8 experiment only |
| `training/kaggle/scripts/r5_9_field_composition.py` | Field composition | Analysis | **DELETE** | — | R5.9 experiment only |
| `training/kaggle/scripts/r5_10_temporal_field_target.py` | Temporal field targets | Analysis | **DELETE** | — | R5.10 experiment only |
| `training/kaggle/scripts/feature_fusion_utils.py` | Feature fusion utilities | Fusion utils | **DELETE** | — | R5.9/R5.10 experiment |
| `training/kaggle/scripts/train_feature_fusion.py` | Feature fusion training | Fusion trainer | **DELETE** | — | R5.9/R5.10 experiment |
| `training/kaggle/scripts/final_cropfusion.py` | Final CropFusion model | Final model | **DELETE** | — | R5.10 experiment |
| `training/kaggle/scripts/class_retention_audit.py` | Class retention audit | Audit | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/corpus_delta_audit.py` | Corpus delta audit | Audit | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/corpus_diagnostics.py` | Corpus diagnostics | Diagnostics | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/imagery_availability_report.py` | Imagery availability | Report | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/matching_verification_audit.py` | Matching verification | Audit | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/model_input_validation.py` | Model input validation | Audit | **DELETE** | — | One-time audit |
| `training/kaggle/scripts/extract_image_embeddings.py` | Embedding extraction | Extraction | **DELETE** | — | R5.9 experiment |
| `training/kaggle/scripts/_dump_logs.py` | Log dump utility | Utility | **DELETE** | — | Dev utility |
| `training/kaggle/scripts/_extract_v4.py` | V4 extraction | Utility | **DELETE** | — | One-time |
| `training/kaggle/scripts/_extract_v7.py` | V7 extraction | Utility | **DELETE** | — | One-time |
| `training/kaggle/scripts/validation_numerics_probe.py` | FP16/FP32 probe | Probe | **DELETE** | — | R5.4 diagnostic |
| `training/kaggle/scripts/diagnose_collapse.py` | Collapse diagnosis | Diagnostic | **DELETE** | — | R5.5 experiment |
| `training/kaggle/scripts/diagnose_collapse_kaggle.py` | Kaggle collapse diagnosis | Diagnostic | **DELETE** | — | R5.5 experiment |
| `training/kaggle/scripts/diagnose_model.py` | Model diagnosis | Diagnostic | **DELETE** | — | One-time |
| `training/kaggle/scripts/run_baselines.py` | Baseline runners | Baselines | **DELETE** | — | R5.6 experiment |
| `training/kaggle/notebooks/train.ipynb` | Training notebook | Training | **KEEP** | training.* | Primary Kaggle notebook |
| `training/kaggle/notebooks/evaluate.ipynb` | Evaluation notebook | Evaluation | **KEEP** | training.* | — |
| `training/kaggle/notebooks/export.ipynb` | Export notebook | Export | **KEEP** | training.* | — |
| `training/kaggle/notebooks/system_check.ipynb` | System check notebook | System check | **KEEP** | — | — |
| `training/kaggle/notebooks/R5_3_*.ipynb` (2) | R5.3 notebooks | — | **DELETE** | — | R5.3 only |
| `training/kaggle/notebooks/R5_4_train.ipynb` | R5.4 training notebook | — | **DELETE** | — | R5.4 only |
| `training/kaggle/notebooks/R5_5_diagnose_collapse.ipynb` | R5.5 diagnostic | — | **DELETE** | — | R5.5 only |
| `training/kaggle/notebooks/R5_6_image_stats.ipynb` | R5.6 image stats | — | **DELETE** | — | R5.6 only |
| `training/kaggle/notebooks/kernel-metadata.json` | Kaggle kernel metadata | — | **DELETE** | — | R5.4 specific |
| `training/kaggle/notebooks/kaggle_pull/` | Pulled notebook output | — | **DELETE** | — | One-time pull |
| `training/kaggle/deployment/r5_3/` | R5.3 deployment | — | **DELETE** | — | R5.3 only |
| `training/kaggle/deployment/r5_3_benchmark/` | R5.3 benchmark | — | **DELETE** | — | R5.3 only |
| `training/kaggle/deployment/r5_4/` | R5.4 deployment | — | **DELETE** | — | R5.4 only |
| `training/kaggle/deployment/r5_5_diagnostic/` | R5.5 deployment | — | **DELETE** | — | R5.5 only |
| `training/kaggle/deployment/r5_6_image_stats/` | R5.6 deployment | — | **DELETE** | — | R5.6 only |
| `training/kaggle/docs/` (6 files) | Kaggle docs | — | **DELETE** | — | Rewrite as needed |
| `training/kaggle/tests/` (~25 files) | Kaggle tests | — | **REWRITE** | training.* | Keep core tests, delete R5.x experiment tests |

---

### 21. `training/config/` — Platform Configuration

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/config/dataset.yaml` | Dataset manager config | Dataset config | **REWRITE** | — | Remove non-DK references, simplify providers |
| `training/config/model.yaml` | Model architecture config | Model config | **KEEP** | — | CropFusionModel config |
| `training/config/preprocessing.yaml` | Preprocessing config | Preproc config | **KEEP** | — | — |
| `training/config/training.yaml` | Training config | Training config | **KEEP** | — | — |
| `training/config/stam.yaml` | STAM config | STAM config | **REWRITE** | — | Remove data_season.csv + ICRISAT references |
| `training/config/seasons.yaml` | Season calendar | Season calendar | **KEEP** | — | Kharif/Rabi/Zaid |
| `training/config/validation.yaml` | Validation defaults | Validation | **REWRITE** | — | Simplify |
| `training/config/paths.yaml` | Workspace paths | Paths | **REWRITE** | — | Simplify for DK-only |
| `training/config/kaggle.yaml` | Kaggle runtime config | Kaggle config | **KEEP** | — | — |
| `training/config/logging.yaml` | Logging config | Logging | **KEEP** | — | — |
| `training/config/performance.yaml` | Performance config | — | **DELETE** | — | Not needed |
| `training/config/feature_fusion.yaml` | Feature fusion config | — | **DELETE** | — | R5.9/R5.10 only |
| `training/config/r5_2_9_matching.yaml` | R5.2.9 matching config | — | **DELETE** | — | One-time pipeline config |

---

### 22. `training/datasets/` — Data Files

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/datasets/tabular/DK_Features_*.csv` (7) | Earth Engine features | **Core features** | **KEEP** | — | Primary tabular data |
| `training/datasets/tabular/data_season.csv` | Legacy seasonal data | — | **DELETE** | — | **Excluded dataset** |
| `training/datasets/tabular/ICRISAT-District Level Data.csv` | District crop stats | — | **DELETE** | — | **Excluded dataset** |
| `training/datasets/tabular/cropdata_updated.csv` | Crop suitability | — | **DELETE** | — | **Excluded dataset** |
| `training/datasets/tabular/All-India_*.csv` | National aggregates | — | **DELETE** | — | **Excluded dataset** |
| `training/datasets/tabular/dataset.csv` | Small climate data | — | **DELETE** | — | **Excluded dataset** |
| `training/datasets/.cropfusion/` | SQLite state | State DB | **KEEP** | — | Dataset manager state |
| `training/datasets/raw/` | Kaggle image cache | Image cache | **KEEP** | — | Runtime only |

---

### 23. `training/artifacts/` — Build Artifacts

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/artifacts/` | Contract splits, diagnostics, smoke test outputs | Artifacts | **DELETE** | — | All generated, not source |

---

### 24. `training/tests/` — Top-Level Training Tests

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/tests/` | Top-level training integration tests | Tests | **REWRITE** | training.* | Simplify for new architecture |

---

### 25. Sub-package `tests/` Directories

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `training/models/tests/` (12 files) | Model unit tests | Tests | **KEEP** | training.models | Core model tests |
| `training/preprocessing/tests/` (~10 files) | Preprocessing tests | Tests | **REWRITE** | training.preprocessing | Remove data_season references |
| `training/stam/tests/` (~8 files) | STAM tests | Tests | **REWRITE** | training.stam | Remove data_season fixtures |
| `training/dataset_manager/tests/` (~10 files) | Dataset manager tests | Tests | **KEEP** | training.dataset_manager | — |
| `training/training/tests/` (~20 files) | Training engine tests | Tests | **REWRITE** | training.training | Remove obsolete R5.x tests |
| `training/explainability/tests/` | Explainability tests | — | **DELETE** | — | Module deleted |
| `training/export/tests/` | Export tests | — | **DELETE** | — | Module deleted |
| `training/evaluation/tests/` | Evaluation tests | Tests | **REWRITE** | training.evaluation | Simplify |
| `training/mlops/tests/` | MLOps tests | — | **DELETE** | — | Module deleted |
| `training/quality/*/tests/` | Quality tests | — | **DELETE** | — | Module deleted |
| `training/feature_store/` (no tests) | Feature store | — | **DELETE** | — | Module deleted |
| `training/matching/tests/` | Matcher tests | Tests | **KEEP** | training.matching | — |
| `training/inference/tests/` | Inference tests | Tests | **KEEP** | training.inference | — |
| `training/runtime/tests/` | Runtime tests | — | **DELETE** | — | Module deleted |
| `shared/tests/` (11 files) | Shared lib tests | — | **DELETE** | — | Rebuild for kept modules |

---

### 26. `scripts/` — Orchestration & Data Scripts

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `scripts/run_full_pipeline.py` | End-to-end orchestrator | Orchestrator | **KEEP** | kagglehub | Useful for full pipeline |
| `scripts/run_kaggle_notebook.py` | Single notebook runner | Notebook runner | **KEEP** | kagglehub | Kaggle CLI helper |
| `scripts/kaggle_r5_3.py` | R5.3 deployment script | — | **DELETE** | — | R5.3 only |
| `scripts/kaggle_r5_3_benchmark.py` | R5.3 benchmark | — | **DELETE** | — | R5.3 only |
| `scripts/kaggle_r5_4.py` | R5.4 deployment | — | **DELETE** | — | R5.4 only |
| `scripts/kaggle_r5_5_diagnostic.py` | R5.5 diagnostic | — | **DELETE** | — | R5.5 only |
| `scripts/kaggle_r5_6_image_stats.py` | R5.6 image stats | — | **DELETE** | — | R5.6 only |
| `scripts/kaggle_r5_7_image_stats.py` | R5.7 image stats | — | **DELETE** | — | R5.7 only |
| `scripts/ogd_*.py` (14 files) | OGD data discovery/download | — | **ARCHIVE** | curl, requests, datagovindia | Historical; data already downloaded |
| `scripts/r5_2_3_*.py` (3 files) | R5.2.3 audit | — | **DELETE** | — | One-time audit |
| `scripts/r5_2_4_*.py` (2 files) | R5.2.4 matching | — | **DELETE** | shared.enums | One-time pipeline |
| `scripts/r5_2_5_*.py` (3 files) | R5.2.5 corpus merge | — | **DELETE** | — | One-time pipeline |
| `scripts/r5_2_6_*.py` (6 files) | R5.2.6 rare class expansion | — | **DELETE** | — | One-time pipeline |
| `scripts/r5_2_7_manifest.py` | R5.2.7 manifest freeze | — | **DELETE** | shared.enums | One-time pipeline |
| `scripts/audit_existing_data.py` | Existing data audit | — | **DELETE** | — | One-time audit |
| `scripts/clean_dk_features.py` | DK feature cleaner | Cleaner | **KEEP** | geopandas | Useful preprocessing tool |
| `scripts/match_audit.py` | Match correctness audit | — | **DELETE** | — | One-time audit |
| `scripts/build_docs.py` | Markdown → HTML docs | Docs builder | **DELETE** | markdown, pygments | Not needed |
| `scripts/tests/test_kaggle_pipeline.py` | Kaggle pipeline tests | Tests | **KEEP** | — | Tests orchestration |
| `scripts/tests/test_build_release.py` | Build release tests | Tests | **KEEP** | — | Tests release packaging |
| `scripts/tests/test_package_sources.py` | Package sources tests | Tests | **KEEP** | — | Tests packaging |
| `scripts/backup/` | Backup scripts | Backup | **DELETE** | — | Dev utility |

---

### 27. `application/backend/` — FastAPI Backend

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/backend/run.py` | Backend launcher | Launcher | **KEEP** | uvicorn | Entry point |
| `application/backend/app/main.py` | App factory | App factory | **REWRITE** | fastapi | Remove auth/middleware complexity |
| `application/backend/app/core/config.py` | Settings tree | Config | **REWRITE** | shared.config | Remove auth/Redis/rate-limit sections |
| `application/backend/app/core/app_container.py` | DI composition root | Container | **REWRITE** | app.* | Simplify wiring |
| `application/backend/app/core/container.py` | DI Container | DI | **KEEP** | — | Generic DI |
| `application/backend/app/core/database.py` | SQLAlchemy async DB | DB | **KEEP** | sqlalchemy | — |
| `application/backend/app/core/logging.py` | Structured logging | Logging | **KEEP** | loguru | — |
| `application/backend/app/core/security.py` | JWT + password hashing | — | **DELETE** | jose, passlib | **No auth in final** |
| `application/backend/app/core/handlers.py` | Exception handlers | Handlers | **KEEP** | fastapi | — |
| `application/backend/app/core/exceptions.py` | Error hierarchy | Errors | **REWRITE** | — | Remove auth errors |
| `application/backend/app/core/paths.py` | sys.path bootstrap | Paths | **KEEP** | — | — |
| `application/backend/app/core/tracing.py` | OpenTelemetry | — | **DELETE** | opentelemetry | Not needed |
| `application/backend/app/api/router.py` | API router aggregator | Router | **REWRITE** | app.modules.* | Remove auth/user/admin/history routers |
| `application/backend/app/middleware/` (7 files) | Auth/logging/timing/rate-limit/security headers/prometheus | Middleware | **REWRITE** | — | Keep logging + timing, remove auth/rate-limit/security headers |
| `application/backend/app/services/release_model_registry.py` | **Release model registry** | Core registry | **KEEP** | inference_package.release | Primary model loading |
| `application/backend/app/services/model_registry.py` | Deprecated R1-R5 registry | — | **DELETE** | training.models | Replaced by release registry |
| `application/backend/app/services/cache.py` | Cache abstraction | Cache | **KEEP** | — | — |
| `application/backend/app/services/rate_limiter.py` | Rate limiter | — | **DELETE** | redis | No rate limiting needed |
| `application/backend/app/services/metrics.py` | In-memory metrics | Metrics | **DELETE** | — | Enterprise only |
| `application/backend/app/services/prometheus.py` | Prometheus metrics | — | **DELETE** | prometheus_client | Not needed |
| `application/backend/app/workers/tasks.py` | Background tasks | Tasks | **REWRITE** | — | Simplify to warmup only |
| `application/backend/app/models/user.py` | User ORM | — | **DELETE** | — | **No auth** |
| `application/backend/app/models/prediction.py` | Prediction ORM | ORM | **KEEP** | — | Stores prediction history |
| `application/backend/app/models/__init__.py` | Model imports | Imports | **REWRITE** | — | Remove user model |
| `application/backend/app/repositories/` | CRUD repos | Repos | **REWRITE** | — | Remove user repo |
| `application/backend/app/dependencies/` | Request-scoped deps | Deps | **REWRITE** | — | Remove auth deps |
| `application/backend/app/modules/auth/` | Auth module | — | **DELETE** | — | **No auth** |
| `application/backend/app/modules/users/` | User module | — | **DELETE** | — | **No auth** |
| `application/backend/app/modules/predictions/` | Prediction orchestration | Predictions | **KEEP** | inference | Core endpoint |
| `application/backend/app/modules/dataset/` | Dataset status | Dataset | **DELETE** | training.dataset_manager | Not exposed to frontend |
| `application/backend/app/modules/gis/` | GIS service | GIS | **REWRITE** | — | Simplify to DK-only spatial lookup |
| `application/backend/app/modules/explainability/` | Explainability module | — | **DELETE** | training.explainability | Not needed |
| `application/backend/app/modules/history/` | History module | History | **DELETE** | — | Not needed for final (no auth = no per-user history) |
| `application/backend/app/modules/admin/` | Admin module | — | **DELETE** | — | No admin |
| `application/backend/app/modules/inference/` | **InferenceEngine** | Core inference | **KEEP** | release_model_registry | Primary inference path |
| `application/backend/app/modules/model_info/` | Model info endpoint | Model info | **KEEP** | — | Useful for status |
| `application/backend/app/modules/configuration/` | Config view | — | **DELETE** | — | Not needed |
| `application/backend/app/modules/monitoring/` | Metrics endpoint | — | **DELETE** | — | Not needed |
| `application/backend/app/modules/health/` | Health endpoints | Health | **KEEP** | — | Standard |

---

### 28. `application/frontend/` — React Frontend

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/frontend/package.json` | Deps & scripts | Deps | **REWRITE** | npm | Remove auth/admin deps |
| `application/frontend/vite.config.ts` | Build config | Build | **KEEP** | vite | — |
| `application/frontend/src/App.tsx` | Routes | Routes | **REWRITE** | react-router | Remove auth/admin/history routes |
| `application/frontend/src/main.tsx` | Entry point | Entry | **REWRITE** | react | Remove auth provider |
| `application/frontend/src/config/index.ts` | FE config | Config | **REWRITE** | — | DK-only defaults |
| `application/frontend/src/types/api.ts` | API types | Types | **REWRITE** | — | Remove auth/user types |
| `application/frontend/src/types/index.ts` | Type re-exports | Types | **REWRITE** | — | — |
| `application/frontend/src/services/api.ts` | Axios instance | API client | **REWRITE** | axios | Remove auth interceptor |
| `application/frontend/src/services/auth.ts` | Auth service | — | **DELETE** | — | **No auth** |
| `application/frontend/src/services/prediction.ts` | Prediction service | Prediction | **KEEP** | — | — |
| `application/frontend/src/services/gis.ts` | GIS service | GIS | **KEEP** | — | — |
| `application/frontend/src/services/explainability.ts` | Explainability | — | **DELETE** | — | Not needed |
| `application/frontend/src/services/admin.ts` | Admin service | — | **DELETE** | — | Not needed |
| `application/frontend/src/store/authStore.ts` | Auth store | — | **DELETE** | zustand | **No auth** |
| `application/frontend/src/store/predictionStore.ts` | Prediction store | Prediction | **KEEP** | zustand | — |
| `application/frontend/src/store/mapStore.ts` | Map store | Map | **KEEP** | zustand | — |
| `application/frontend/src/store/themeStore.ts` | Theme store | Theme | **KEEP** | zustand | — |
| `application/frontend/src/store/uiStore.ts` | UI store | UI | **KEEP** | zustand | — |
| `application/frontend/src/hooks/useAuth.ts` | Auth hook | — | **DELETE** | — | **No auth** |
| `application/frontend/src/hooks/usePrediction.ts` | Prediction hook | Prediction | **KEEP** | — | — |
| `application/frontend/src/hooks/useMap.ts` | Map hook | Map | **KEEP** | — | — |
| `application/frontend/src/hooks/useHistory.ts` | History hook | — | **DELETE** | — | No history |
| `application/frontend/src/hooks/useToast.ts` | Toast hook | Toast | **KEEP** | — | — |
| `application/frontend/src/contexts/AuthContext.tsx` | Auth context | — | **DELETE** | — | **No auth** |
| `application/frontend/src/contexts/ThemeContext.tsx` | Theme context | Theme | **KEEP** | — | — |
| `application/frontend/src/pages/LandingPage.tsx` | Landing page | Landing | **REWRITE** | — | Simplify |
| `application/frontend/src/pages/LoginPage.tsx` | Login | — | **DELETE** | — | **No auth** |
| `application/frontend/src/pages/RegisterPage.tsx` | Register | — | **DELETE** | — | **No auth** |
| `application/frontend/src/pages/PredictionPage.tsx` | Prediction page | Prediction | **REWRITE** | — | DK-only, no auth |
| `application/frontend/src/pages/MapPage.tsx` | Map page | **Core map** | **REWRITE** | leaflet | DK-restricted, click-to-predict |
| `application/frontend/src/pages/HistoryPage.tsx` | History page | — | **DELETE** | — | No auth |
| `application/frontend/src/pages/DashboardPage.tsx` | Dashboard | — | **DELETE** | — | No auth |
| `application/frontend/src/pages/ProfilePage.tsx` | Profile | — | **DELETE** | — | No auth |
| `application/frontend/src/pages/SettingsPage.tsx` | Settings | — | **DELETE** | — | No auth |
| `application/frontend/src/pages/ExplainPage.tsx` | Explainability | — | **DELETE** | — | Not needed |
| `application/frontend/src/pages/AdminDashboard.tsx` | Admin | — | **DELETE** | — | No admin |
| `application/frontend/src/components/layout/AppLayout.tsx` | App layout | Layout | **REWRITE** | — | Simplify nav |
| `application/frontend/src/components/layout/AuthLayout.tsx` | Auth layout | — | **DELETE** | — | No auth |
| `application/frontend/src/components/layout/Header.tsx` | Header | Header | **REWRITE** | — | Simplify |
| `application/frontend/src/components/layout/Sidebar.tsx` | Sidebar | — | **DELETE** | — | Simplified UI |
| `application/frontend/src/components/layout/Footer.tsx` | Footer | Footer | **KEEP** | — | — |
| `application/frontend/src/components/Map/` (4 files) | Map components | Map | **REWRITE** | leaflet | DK-restricted bounds |
| `application/frontend/src/components/prediction/` (4 files) | Prediction UI | Prediction | **KEEP** | — | — |
| `application/frontend/src/components/explainability/` (3 files) | Explainability UI | — | **DELETE** | — | Not needed |
| `application/frontend/src/components/ui/` (9 files) | UI primitives | UI | **KEEP** | — | — |
| `application/frontend/src/components/PredictionCard.tsx` | Prediction card | Prediction | **KEEP** | — | — |
| `application/frontend/src/utils/` (3 files) | Utilities | Utilities | **KEEP** | — | — |
| `application/frontend/src/tests/` (6 files) | Frontend tests | Tests | **REWRITE** | vitest | Remove auth tests |

---

### 29. `application/database/` — Enterprise Database

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/database/` (entire directory) | Enterprise ORM, migrations, seeds, repos, services, API, security, policies | Enterprise DB | **DELETE** | sqlalchemy, alembic | **Entire enterprise database layer not needed** — use simple SQLite for predictions only |

---

### 30. `application/inference/` — Inference Ports/ABCs

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/inference/models.py` | DTOs | DTOs | **KEEP** | shared.enums | — |
| `application/inference/engine/__init__.py` | InferenceEngine ABC | ABC | **DELETE** | — | Concrete implementation exists in modules/inference |
| `application/inference/loaders/__init__.py` | ModelLoader ABC | ABC | **DELETE** | — | Replaced by release loader |
| `application/inference/versioning/__init__.py` | VersionResolver ABC | ABC | **DELETE** | — | Replaced |
| `application/inference/validation/__init__.py` | PackageValidator ABC | ABC | **DELETE** | — | Replaced |
| `application/inference/cache/__init__.py` | PredictionCache ABC | ABC | **DELETE** | — | Not needed |
| `application/inference/services/__init__.py` | PredictionService ABC | ABC | **DELETE** | — | Not needed |
| `application/inference/explainability/__init__.py` | Explainer ABC | ABC | **DELETE** | — | Not needed |
| `application/inference/__init__.py` | Package init | Init | **REWRITE** | — | Keep models only |

---

### 31. `application/inference_package/` — Release Package Contract

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/inference_package/manifest.py` | R1.4 flat contract | — | **DELETE** | — | Replaced by release/manifest.py |
| `application/inference_package/__init__.py` | Package init | Init | **REWRITE** | — | Re-export release only |
| `application/inference_package/release/__init__.py` | Release exports | Release | **KEEP** | .loader, .manifest | — |
| `application/inference_package/release/manifest.py` | **R6 release contract** | Core contract | **KEEP** | — | Defines cropfusion_release/ layout |
| `application/inference_package/release/loader.py` | **ReleasePackageLoader** | Core loader | **KEEP** | torch, pandas, yaml | Loads exported model |
| `application/inference_package/README.md` | Documentation | Docs | **DELETE** | — | Rewrite |

---

### 32. `application/gis/` — GIS Service

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/gis/models.py` | GeoPoint, ResolvedPlace, etc. | DTOs | **KEEP** | shared.enums | — |
| `application/gis/resolver.py` | LocationResolver ABC | ABC | **REWRITE** | — | Simplify to DK-only |
| `application/gis/spatial_resolver/` | Spatial resolver ABC | ABC | **DELETE** | — | Simplify to direct lookup |
| `application/gis/reverse_geocoding/` | Reverse geocoder ABC | ABC | **DELETE** | — | Use release location_index directly |
| `application/gis/historical_context/` | Historical context ABC | ABC | **DELETE** | — | Use release historical_context directly |
| `application/gis/Taluk/` | Taluk shapefiles | Shapefiles | **KEEP** | — | DK boundary data |
| `application/gis/kml/` | KML boundary | Boundary | **KEEP** | — | Map display |
| `application/gis/README.md` | GIS docs | Docs | **DELETE** | — | — |

---

### 33. `application/history/` — Prediction History

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/history/` (all files) | Prediction history ABCs | — | **DELETE** | — | No user-specific history |

---

### 34. `application/config/` — App Config Templates

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/config/application.yaml` | App template | — | **DELETE** | — | Use settings.yaml directly |
| `application/config/database.yaml` | DB template | — | **DELETE** | — | No enterprise DB |
| `application/config/inference.yaml` | Inference template | — | **DELETE** | — | Use settings.yaml |
| `application/config/model.yaml` | Model template | — | **DELETE** | — | Use settings.yaml |
| `application/config/security.yaml` | Security template | — | **DELETE** | — | No auth |
| `application/config/logging.yaml` | Logging template | — | **DELETE** | — | Use settings.yaml |

---

### 35. `application/monitoring/` — Monitoring Stack

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/monitoring/` (all files) | Prometheus/Grafana/Loki configs | — | **DELETE** | — | Not needed |

---

### 36. `application/docker/` — Docker Config

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/docker/docker-compose.yml` | Full stack compose | — | **REWRITE** | docker | Simplify to frontend + backend |
| `application/docker/docker-compose.prod.yml` | Production compose | — | **DELETE** | docker | Not needed |

---

### 37. `application/tests/` — Platform Tests

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/tests/conftest.py` | Test config | Config | **REWRITE** | — | Simplify |
| `application/tests/smoke/test_startup.py` | Startup smoke test | Smoke test | **KEEP** | — | — |
| `application/tests/test_r1_4_skeletons.py` | Skeleton tests | — | **DELETE** | — | One-time validation |

---

### 38. `application/backend/app/tests/` — Backend Tests

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `application/backend/app/tests/conftest.py` | Test fixtures | Fixtures | **REWRITE** | — | Remove auth fixtures |
| `application/backend/app/tests/test_auth.py` | Auth tests | — | **DELETE** | — | No auth |
| `application/backend/app/tests/test_users.py` | User tests | — | **DELETE** | — | No auth |
| `application/backend/app/tests/test_predictions.py` | Prediction tests | Tests | **KEEP** | — | Core endpoint test |
| `application/backend/app/tests/test_inference.py` | Inference tests | Tests | **KEEP** | — | — |
| `application/backend/app/tests/test_history.py` | History tests | — | **DELETE** | — | No history |
| `application/backend/app/tests/test_health.py` | Health tests | Tests | **KEEP** | — | — |
| `application/backend/app/tests/test_gis.py` | GIS tests | Tests | **REWRITE** | — | Simplify |
| `application/backend/app/tests/test_dataset.py` | Dataset tests | — | **DELETE** | — | Dataset module removed |
| `application/backend/app/tests/test_repository.py` | Repo tests | Tests | **REWRITE** | — | Remove user repo |
| `application/backend/app/tests/test_observability.py` | Observability tests | — | **DELETE** | — | Prometheus removed |
| `application/backend/app/tests/test_integration.py` | Integration tests | Tests | **REWRITE** | — | Simplify for DK-only |
| `application/backend/app/tests/test_enterprise_api.py` | Enterprise tests | — | **DELETE** | — | Enterprise removed |

---

### 39. Data Directories (Non-Code)

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `Tabular_Datasets/` | Original data files | — | **DELETE** | — | Superseded by training/datasets/tabular/ |
| `Tabular_Datasets/cleaned/` | Cleaned features | — | **DELETE** | — | Use training/datasets/tabular/ |
| `govt_crop_survey_data/` | Raw OGD downloads | Archive | **ARCHIVE** | — | Source data already consumed |
| `govt_crop_matched_v1/` | V1 matched corpus | Archive | **KEEP** | — | Provenance record |
| `govt_crop_matched_v2/` | V2 matched corpus | Archive | **KEEP** | — | Provenance record |
| `training_manifests/` | Dataset manifests | Manifests | **KEEP** | — | Provenance |
| `reports/` (34 entries) | R5.x analysis reports | — | **DELETE** | — | Experiment reports, not source |
| `reports/R5.5_*.md` (many) | Diagnostic reports | — | **DELETE** | — | — |
| `reports/R5.6/` through `reports/R5.10/` | Experiment reports | — | **DELETE** | — | — |
| `reports/final/` | Final reports | — | **DELETE** | — | — |
| `reports/feature_fusion/` | Fusion reports | — | **DELETE** | — | — |
| `releases/v1.0/` | Empty release | — | **DELETE** | — | Empty |
| `releases/v1.1/` | v1.1 release | Archive | **KEEP** | — | Historical |
| `releases/v2.0.0/` | **Current release** | Release | **KEEP** | — | Active release package |
| `releases/latest/` | Symlink to v2.0.0 | Release | **KEEP** | — | Active |
| `paper/` (10 .md files) | Paper sections | Paper | **KEEP** | — | Project paper |

---

### 40. `cropfusion/` — Umbrella Package

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `cropfusion/__init__.py` | Umbrella package | Package root | **KEEP** | none | Top-level namespace |

---

### 41. `docs/` — Documentation

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `docs/` (all .md files) | Project documentation | — | **DELETE** | — | Rewrite entirely for simplified project |

---

### 42. `.github/` — CI/CD

| PATH | CURRENT ROLE | FINAL ROLE | ACTION | DEPENDENCIES | REASON |
|------|-------------|-----------|--------|--------------|--------|
| `.github/` | CI/CD workflows | — | **REWRITE** | github actions | Simplify to build + test |

---

## Duplicate Implementation Summary

| Function | Existing Implementations | Recommended for Final |
|----------|------------------------|----------------------|
| **Dataset loading** | `dataset_manager/csv_loader.py`, `kaggle/frozen_corpus.py`, `stam/matcher.py`, scripts (ogd_*), `preprocessing/data_contract.py` | `frozen_corpus.py` + `dataset_manager/csv_loader.py` |
| **Preprocessing** | `preprocessing/master_pipeline.py`, `preprocessing/tabular_pipeline.py`, `preprocessing/image_pipeline.py`, `preprocessing/temporal_pipeline.py`, `preprocessing/label_pipeline.py` | All (this is the canonical pipeline) |
| **Splitting** | `preprocessing/dataset.py` (split_observations), `training/validator.py` (cross_validation), `kaggle/frozen_corpus.py` (spatial split) | `frozen_corpus.py` (spatial leave-one-taluk-out) |
| **Training** | `training/trainer.py`, `training/cropfusion_trainer.py`, `kaggle/scripts/run_pipeline.py`, `kaggle/scripts/run_training.py` | `run_pipeline.py` → `Experiment.run()` |
| **Inference** | `application/modules/inference/service.py`, `application/modules/predictions/service.py`, `training/runtime/` | `application/modules/inference/service.py` |
| **Model definitions** | `training/models/cropfusion.py`, `training/models/feature_fusion.py`, `training/kaggle/scripts/final_cropfusion.py` | `training/models/cropfusion.py` |
| **Kaggle orchestration** | `scripts/kaggle_r5_3.py`, `scripts/kaggle_r5_4.py`, `scripts/run_full_pipeline.py`, `scripts/run_kaggle_notebook.py` | `scripts/run_full_pipeline.py` + `run_kaggle_notebook.py` |
| **Model registry** | `app/services/model_registry.py` (deprecated), `app/services/release_model_registry.py` (active) | `release_model_registry.py` |

---

## Excluded Dataset Usage in Active Code

| File | Reference | Criticality | Fix Required |
|------|-----------|-------------|-------------|
| `training/config/stam.yaml:84` | Lists `data_season.csv` as primary tabular table | **CRITICAL** | Rewrite to use DK_Features only |
| `training/config/stam.yaml:164` | Lists `ICRISAT-District Level Data.csv` as fallback | **CRITICAL** | Remove |
| `training/preprocessing/data_contract.py:34` | `_KG_HA_SOURCES = ("data_season",...)` | **CRITICAL** | Update yield unit inference for DK_Features |
| `training/stam/tests/conftest.py` | Creates fake `data_season.csv` for tests | Moderate | Rewrite test fixtures |
| `training/preprocessing/tests/test_data_contract.py` | Asserts on `"data_season.csv"` string | Moderate | Rewrite tests |
| `training/kaggle/scripts/backfill_release.py:464` | Default CLI path `Tabular_Datasets/data_season.csv` | Moderate | Update default path |
| `application/database/seeds/boundaries.py:21` | Hardcoded ICRISAT path | **CRITICAL** | Rewrite or delete (enterprise DB deleted) |
| `training/config/r5_2_9_matching.yaml:51` | `dk_dir: Tabular_Datasets` | Moderate | Update path |
| `training/matching/spatial_tabular_matcher.py` | `Yield_Proxy_NPP` excluded column guard | **KEEP** | This is a leakage guard, not a dependency |

---

## Unresolved Dependency Questions

1. **`training/feature_engineering/`**: Only imported by `training/export/` which is being deleted. Does any active training pipeline path actually use it? **Answer: No — it's used by the export pipeline only. Can delete if export is deleted.**

2. **`training/evaluation/`**: Imported by `training/kaggle/scripts/evaluate.py` and `training/training/cropfusion_trainer.py`. Is the evaluation framework needed? **Answer: Yes — `evaluator.py` is used by `experiment.py`. Keep core metrics/evaluator, delete ablation/comparison/error_analysis.**

3. **`training/inference/`**: Imported by `training/kaggle/scripts/build_release.py`. Is it needed? **Answer: Yes — `build_release.py` uses `training.inference.exporter` and `training.inference.package_builder` for release packaging.**

4. **`application/database/`**: Huge enterprise layer. Can it all be deleted? **Answer: Yes — the simplified frontend has no auth, no user management, no enterprise features. The only persistence needed is a simple SQLite for prediction history (if even that).**

5. **`shared/validation/`**: Used by `training/kaggle/validation.py`. Can it be inlined? **Answer: Yes — validation is a simple gate check. The shared/validation framework is over-engineered for this.**

6. **`shared/schemas/`**: Only imported by `shared/tests/`. Safe to delete? **Answer: Yes — no runtime code imports it.**

7. **`shared/interfaces/`**: Only imported by `shared/tests/`. Safe to delete? **Answer: Yes — no runtime code imports it.**

8. **`training/kaggle/enhanced_observation_generator.py`**: Uses `data_season.csv` path indirectly through `matching.spatial_tabular_matcher`. Is this still active? **Answer: It was active in R5.2.9 for the enhanced observation campaign. For the final project, if you use the frozen v2 corpus, this is not needed at training time.**

---

## Statistics

| Category | Count |
|----------|-------|
| **Total files inspected** | ~800+ |
| **Files proposed for KEEP** | ~210 |
| **Files proposed for REWRITE** | ~85 |
| **Files proposed for DELETE** | ~420 |
| **Files proposed for ARCHIVE** | ~15 |
| **Unresolved dependency questions** | 8 (answered above) |
