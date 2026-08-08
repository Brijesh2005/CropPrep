# COMPLETE SYSTEM AUDIT — CropPrep / CropFusion (R2.4 training stack)

Status: verified against source at commit HEAD, plus local end-to-end runs of the
integration driver (`training/kaggle/scripts/run_pipeline.py`) and the full test
suites.

Everything in this document is code-faithful. Where a module or capability is
missing or not yet runnable, it is explicitly labelled **GAP** rather than
invented. Facts marked *observed* were reproduced on this machine
(`D:\CropPrep`, Windows, CPU-only torch).

---

## 1. System overview and repository layout

CropPrep is a monorepo with three cooperating sub-systems:

| Path | Role |
|---|---|
| `training/` | Research/AI: dataset management, STAM (Spatio-Temporal Agricultural Matcher), preprocessing, multimodal model, training, evaluation, explainability, inference packaging, Kaggle orchestration. Installed editable as the `cropfusion_training` distribution (import namespace `training.*`). |
| `application/` | Product: FastAPI backend (`application/backend/app/`), React frontend, database models, GIS/inference/history service packages. Installed under the `application.*` namespace. |
| `shared/` | Cross-cutting helpers consumed by several packages: `shared.config` (env parsing, `deep_merge`, `apply_case_insensitive`) is imported by the model, preprocessing and dataset-manager config modules. |
| `docs/` | Phase completion reports, migration reports, research docs, user guides, diagrams. |

The AI side is organised by **phases** (Phase 2 dataset manager → Phase 3 STAM →
Phase 4 preprocessing → Phase 5 model → Phase 6 training → Phase 9 runtime
packaging → Phase 10 release manager), and reworked in **releases R2.1–R2.4/R5**
(documented under `docs/migration/`).

### `training/` top-level layout

```
training/
  dataset_manager/     # providers, download, scan, validate, metadata, patch extraction, CLI
  stam/                # STAM: spatial/temporal matching, patch generation, observation resolver
  feature_engineering/ # dataset glue: build_cropfusion_datasets, tabular/image/temporal stats
  preprocessing/       # master_pipeline Preprocessor, pipelines, dataset, dataloader, split
  models/              # CropFusionModel architecture, factory, checkpoint, exporter, runtime
  training/            # Experiment, Trainer, Evaluator, losses, optimizers, schedulers, CV
  evaluation/          # metrics, comparison, ablation study, error analysis, reports
  explainability/      # Grad-CAM, SHAP, integrated gradients, cross-modal attention, uncertainty
  export/              # dataset export (JSON / Parquet / torch)
  inference/           # inference package builder + versioning
  runtime/             # InferenceRuntime: model/preprocess loaders, health, release manager
  feature_store/       # caches: observation/patch/tabular/temporal/image embedding
  quality/             # drift, fairness, monitoring, optimization (benchmark/ONNX)
  mlops/               # experiments registry, gates, scheduler, CLI
  kaggle/              # orchestration: config, workspace, environment, validation, scripts
  config/              # repo-level YAML: paths, dataset, training, model, stam, seasons, validation
```

The R2.4 integration driver ties the chain together:
`training/kaggle/scripts/run_pipeline.py` (see §3).

---

## 2. Configuration system

Every sub-system follows the same resolution pattern, described once here.

1. **Defaults** are pydantic `BaseModel` fields (`extra="forbid"`).
2. **YAML file** overrides defaults.
3. **Environment variables** with a subsystem prefix override YAML
   (`KAGGLE_*`, `MODEL_*`, `PRE_*`, `DM_*`, etc.).
4. Final document is validated by pydantic; malformed input raises the
   subsystem's `ConfigurationError`.

Loaders (all accept a path or env var, return a validated model):

| Sub-system | Loader | Prefix | Repo config |
|---|---|---|---|
| dataset_manager | `load_settings(path)` (`dataset_manager/config.py:329`) | `DM_*` | `training/config/dataset.yaml` |
| stam | `load_stam_config(path)` (`stam/config.py:198`) | `STAM_*` | `training/config/stam.yaml` |
| preprocessing | `load_preprocessing_config(path)` (`preprocessing/config.py:197`) | `PRE_*` | `training/config/preprocessing.yaml` (template; env-driven) |
| models | `load_model_config(path)` (`models/config.py:497`) | `MODEL_*` (`MODEL_CONFIG_FILE`) | `training/config/model.yaml` |
| training | `load_training_config(path)` (`training/config.py:455`) | `TRAIN_*` | `training/config/training.yaml` |
| kaggle | `load_kaggle_config()` / `load_paths_config(path)` (`kaggle/config.py:228/189`) | `KAGGLE_*` | `training/config/paths.yaml` + `training/kaggle/config/*.yaml` |

Notable details:
- `paths.yaml` supports `extends:` chains (`_resolve_extends`, `kaggle/config.py:216`).
- `WorkspaceLayout.resolve(paths, repo_root=...)` (`kaggle/config.py:279`) turns
  the relative `paths.yaml` entries into absolute paths rooted at the repo, so
  `logs/outputs/checkpoints/cache/configs` are stable regardless of CWD.
- STAM's season calendar is loaded via `stam_cfg.temporal.season_file`
  (`training/config/seasons.yaml`), resolved relative to the STAM config's own
  directory by `run_pipeline.py` so it works from any working directory
  (`run_pipeline.py:175-183`).
- `ModelConfig.from_preprocessor(preprocessor)` / `ModelFactory.build_config`
  derive the tabular schema, crop `num_classes`, image `input_size` and temporal
  `max_len` from a fitted Phase 4 `Preprocessor` (`models/config.py:414-457`), so
  the model always matches the preprocessing output.

---

## 3. End-to-end execution flow

### 3.1 The integration driver (`run_pipeline.py`)

`training/kaggle/scripts/run_pipeline.py` is the single entry point for the full
R2.4 chain. Exact stages (with the exact methods invoked):

1. **sys.path hardening** — `_add_repo_root()` (`run_pipeline.py:34`) inserts the
   repo root at position 0 and removes any `sys.path` entry that shadows
   `training/` (protects against a stale `/kaggle/working/training` or a CWD
   `training` package).
2. **Config** — loads `paths`, `kaggle`, `logging`, `dataset`, `training`,
   `model`, `stam` configs; resolves the season file (see §2).
3. **Environment / logging / workspace** — `EnvironmentManager(repo_root).report()`
   → `WorkspaceLayout.resolve(paths, repo_root)` → `TrainingLogger(...).setup()`
   → `WorkspaceManager(layout).create()`.
4. **DatasetManager** — `DatasetManager(dataset_settings)`; records
   `manager.provider_manifests()` and `manager.tabular_names()`.
5. **Imagery** — reads the `kaggle_hub_image` manifest; when available runs
   `manager.ensure_image()` then `manager.generate_image_metadata(force=...)`.
   When unavailable (local machines) the stage records a warning and the rest of
   the chain degrades (see §6).
6. **STAM** — `STAM(manager, stam_cfg)`; `stam.initialize()` (builds the
   spatial index, temporal index and season resolver). Report captures
   `stam.matcher.spatial_stats()`, `stam.season_resolver.names()` and
   `stam.season_resolver.source`.
7. **Corpus** — `ObservationResolver(stam)`;
   `resolver.plan(years=..., seasons=..., max_locations=...)` → optional
   `--max-cells` truncation via `plan.model_copy(update={"cells": ...})` →
   `resolver.resolve(plan)` → `ObservationCorpus`. The corpus is saved to
   `reports/corpus.json` and summarised with `corpus.summary()` /
   `corpus.status_counts()`.
8. **Experiment** — if not `--skip-training` and `corpus.accepted_observations()`
   is non-empty: `workspace.run_output(training_cfg.name)` →
   `Experiment(training_cfg, accepted, extractor=stam.get_patch,
   model_config=model_cfg, run_dir=run_dir, run_name=name).run()` →
   `ExperimentReport.to_dict()`.
9. **Validation** — `TrainingValidator(paths, layout, env_report).validate(
   provider_manifests=manifests)` → `ValidationResult.to_dict()` /
   `.passed` / `.by_severity()`.
10. **Report** — writes `pipeline.json` to the workspace outputs (or `--output`).

All helper calls were re-verified against source this audit:
`manager.provider_manifests` (`manager.py:848`), `tabular_names` (`:721`),
`ensure_image` (`:783`), `generate_image_metadata` (`:803`), `close` (`:1361`);
`ObservationCorpus.{total,counts,accepted,rejected,errors,accepted_observations,
status_counts,summary,save}` (`stam/observation_resolver.py:76-210`);
`Experiment.__init__/run` (`training/experiment.py:80/113`);
`TrainingValidator.__init__/validate` (`kaggle/validation.py:51/66`);
`WorkspaceLayout.resolve` (`kaggle/config.py:279`);
`WorkspaceManager.{create,output_path,run_output,report}`;
`EnvironmentManager.report`; `TrainingLogger.{setup,log_experiment}`.

### 3.2 The Experiment flow (`training/training/experiment.py`)

`Experiment.run()` dispatches on `validation.strategy`:

- **holdout** (`_run_holdout`):
  1. `_holdout_split()` → train/val/test (leakage-free).
  2. `_ensure_fitted(train)` → `Preprocessor.fit(train, extractor=...)`
     (**fit on train only — no label leakage**).
  3. `_resolve_model_config(preprocessor)` → `ModelFactory.from_preprocessor`.
  4. `_build_loader(...)` → `build_dataloader(CropFusionDataset.build(...))`.
  5. `ModelFactory.create(model_config)`.
  6. `_run_trainer(...)` → `Trainer`.
  7. `_evaluate(model, test_loader)` → `Evaluator`.
  8. `_benchmark(model)` → `Benchmark`.
  9. `_visualize(...)` → `Visualizer`; `logger.save_config_snapshot(...)`.
  10. Returns `ExperimentReport` (`to_dict()`).
- **cv** (`_run_cross_validation`): builds fold generators
  (`build_fold_generator`, `cross_validation_splits`) and runs per-fold fits,
  aggregating into a CV report.

### 3.3 The Trainer flow (`training/training/trainer.py`)

`Trainer` receives an already-built model + train/val loaders + loss + optimizer
+ scheduler + validator + checkpoint manager + callbacks, and provides: AMP
(`amp_context`), gradient clipping/accumulation, NaN detection, early stopping,
resume, and DDP-with-CPU fallback. Returns `TrainingResult(epochs, steps,
history, best_metrics, ...)`.

---

## 4. Per-module inventory (verified signatures)

### 4.1 dataset_manager (`training/dataset_manager/`)

Provider-based dataset source layer. Interfaces in `interfaces.py`; registry in
`provider_registry.py`; providers in `providers/` (`git_tabular.py`,
`kaggle_image.py` — the KaggleHub image provider gained `/kaggle/input` mount
support in the B4 fix). Key classes:

- `DatasetManager` (`manager.py:98`) — facade: inventory, ensure (tabular/image),
  metadata, spatial, patches, historical context, versioning, cache, reports.
- `DatasetScanner` (`scanner.py:52`), `PandasCSVLoader` (`csv_loader.py:40`),
  `RasterioImageLoader` (`image_loader.py:49`, with a light-weight TIFF reader
  that avoids full rasterio for header info), `PatchExtractorImpl`
  (`patch_extractor.py:36`), `SpatialIndexImpl` (`spatial_index.py:31`).
- `SQLiteRegistry` (`dataset_registry.py:53`), `MetadataGeneratorImpl`
  (`metadata.py:79`), `SQLiteMetadataStore` (`metadata.py:213`),
  `MetadataRepository` (`metadata_repository.py:79`), `CacheManager`
  (`cache_manager.py:43`), `KaggleDownloader` (`downloader.py:44`),
  `HistoricalContextBuilderImpl` (`historical_context_builder.py:45`).
- CLI `python -m training.dataset_manager` (`cli.py`) with ~35 subcommands
  (download/scan/validate/inventory/patch/historical-context/versions/cache…).
- Models (`models.py`): `MetadataRecord`, `RasterMetadata`, `CSVProfile`,
  `SpatialRecord`, `TemporalRecord`, `PatchMetadata`, `HistoricalContext`,
  `DatasetInventory`, `DatasetSummary`, `DatasetStatistics`, `ValidationReport`.

### 4.2 stam (`training/stam/`)

Spatio-Temporal Agricultural Matcher. Interfaces: `ImageMetadataSource`,
`ImageReader`, `TabularSource`, `LocationCatalog`, `BoundaryProvider`, `StamCache`.
Adapters over the dataset manager live in `matcher.py`:
`DatasetManagerImageSource`, `DatasetManagerImageReader`,
`DatasetManagerTabularSource`, `DatasetManagerBoundaryProvider`,
`DatasetManagerLocationCatalog`, `SpatialTemporalMatcher` (`matcher.py:390`, with
`match_tabular`/`match_images`/`spatial_stats`).

- `STAM` facade (`stam.py:56`): `initialize()`, `build_observation(...)`,
  `get_patch(path, lon, lat, size=...)` (the extractor consumed by Phase 4).
- `SpatialPatchGenerator` (`patch_generator.py:84`) + `RasterPatch`.
- `ImagePairBuilder` / `ObservationSequenceBuilder` (`sequence_builder.py`).
- `SeasonResolver` (`season_resolver.py:41`) — `names()`, `source`;
  `TemporalIndex` + `SeasonCalendar` (`temporal_index.py`).
- `ObservationResolver` (`observation_resolver.py:229`) — `plan()`, `resolve()`,
  `resolve_cell()`, `available_years()`, `available_seasons()`, `locations()`;
  `ObservationPlan`, `SamplingCell`, `ResolvedSample`, `ObservationCorpus`.
- `HistoricalContextBuilder` (`historical_context.py:24`), CRS helpers
  (`coordinate_transform.py`), quality assessment (`validators.py`).

### 4.3 feature_engineering (`training/feature_engineering/`)

- `build_cropfusion_datasets(corpus_or_obs, preprocessor, split_config,
  extractor=...)` (`dataset.py`) — corpus → accepted-only → leakage-free split →
  train/val/test `CropFusionDataset`s.
- `tabular.py`, `image.py`, `temporal.py`, `statistics.py`, `balancing.py`,
  `builder.py` — per-modality feature builders + class balancing.

### 4.4 preprocessing (`training/preprocessing/`)

- `Preprocessor` (`master_pipeline.py:44`) — facade over four pipelines:
  `TabularPipeline`, `ImagePipeline`, `TemporalPipeline`, `LabelPipeline` plus
  `ImageAugmentation`. Contract:
  `fit(train_obs, extractor=...)`, `transform(obs, extractor=..., augment=False)`
  → sample dict `{observation_id, tabular[F], ndvi[T,1,H,W], evi[T,1,H,W],
  temporal_mask[T], crop_label, yield_label, metadata}`.
  Quality filter (`filter`/`filter_one`), `validate`, `summary`, `save`/`load`.
- `CropFusionDataset.build(preprocessor, observations, split=..., extractor=...)`
  and `build_dataloader` (`dataset.py`, `dataloader.py`); lazy `__getitem__`
  calls preprocessor + extractor on demand.
- `split_observations` (random/stratified/temporal/spatial/group) and
  `SplitConfig` (`preprocessing/config.py:112`).
- `transforms.py` (incl. `OrdinalEncoder`), `validators.py`, `augmentations.py`,
  `statistics.py`.

### 4.5 models (`training/models/`)

- `CropFusionModel` (`cropfusion.py:89`) — the full Phase 5 architecture:
  `TabTransformer` (CLS pool) → tabular embedding; NDVI/EVI `TimmImageEncoder`
  backbones (default `efficientnetv2_s`) → `ImageFusion` (concat/weighted_sum/
  learnable/attention) → `TemporalTransformer` (variable-length, mask-aware) →
  `CrossModalFusionEngine` (cross-attention Q=image K=V=tabular → adaptive gated
  fusion → `SharedMultimodalEncoder`) → `CropHead` + `YieldHead`. Single-modality
  models use a standalone `SharedMultimodalEncoder`. Optional fourth gate
  (`fusion.use_temporal_stream`). `CropFusionOutput` dataclass incl. per-sample
  `gates` for explainability.
- `ModelFactory` (`factory.py:40`) — `create`, `from_config_file`,
  `from_preprocessor`, `from_checkpoint`, architecture registry
  (`cropfusion_v1`), `freeze_backbone`, `load_backbone`, runtime helpers.
- `ModelConfig` (pydantic, `config.py`) — every subsection is a validated model;
  `ModelConfig._derived_schema`/`from_preprocessor` bridge Phase 4→5.
- `CheckpointManager` + `LoadReport`/`ResumeState` (`checkpoint.py:70`).
- `ModelExporter` (`exporter.py:78`) — TorchScript / ONNX via
  `forward_export`.
- `runtime.py` — `resolve_device`, `amp_context`, `apply_precision`,
  `compile_model`, `enable_gradient_checkpointing`, `wrap_data_parallel`,
  `wrap_distributed`, `apply_runtime`.
- Sub-modules: `adaptive_gate.py` (`AdaptiveGatedFusion`), `cross_attention.py`,
  `fusion_engine.py` (`CrossModalFusionEngine` + `FusionOutput`), `image_fusion.py`,
  `multitask_heads.py` (`CropHead`/`YieldHead`/`MultiTaskHeads`),
  `shared_encoder.py`, `tabtransformer.py`, `temporal_transformer.py`,
  `backbone.py` (`TimmImageEncoder` base), `ndvi_encoder.py`/`evi_encoder.py`,
  `losses.py` (`CrossEntropyLoss`, `LabelSmoothingLoss`, `FocalLoss`, `MSELoss`,
  `HuberLoss`, `WeightedMultiTaskLoss`), `validators.py`
  (`validate_model_config`, `validate_batch`, `expected_batch_shapes`),
  `interfaces.py` (`ImageEncoder`, `Head`, `TaskLoss`).

### 4.6 training engine (`training/training/`)

- `Experiment`/`ExperimentReport` (`experiment.py:77/49`) + `run_experiment()`.
- `Trainer` (`trainer.py`) + `TrainingResult`; `CropFusionTrainer`
  (`cropfusion_trainer.py`).
- `Evaluator` + `EvaluationResult` (`evaluator.py`).
- `Validator` + `ValidationResult` + fold generators
  (`validator.py:51` — hold-out, KFold, StratifiedKFold, Spatial, Temporal) +
  `cross_validation_splits`.
- `losses.py` — `MultiTaskLoss`, `GradNormController`,
  `build_class_weights`/`class_frequency_weights`, `build_multi_task_loss`.
- `optimizers.py` — `build_optimizer` + custom `Lion`.
- `schedulers.py` — `build_scheduler` (warmup, step, polynomial, cosine…).
- `benchmark.py` (`Benchmark`/`BenchmarkReport`), `visualizer.py` (`Visualizer`),
  `callbacks.py`, `curriculum.py`, `checkpoint.py`, `profiler.py`, `metrics.py`,
  `ablation.py`, `reports.py`.

### 4.7 evaluation (`training/evaluation/`)

`MultimodalEvaluator`/`EvaluationOutcome` (`evaluator.py:34/67`),
`compute_classification_metrics` / `compute_regression_metrics` / `EvaluationAccumulator`
(`metrics.py`), `ComparisonTable` + comparison builders (`comparison.py`),
`AblationStudy` (`ablation.py:209`, variant surgery on config+model),
`ErrorAnalysis` (`error_analysis.py:42`), report/markdown + figures (`reports.py`),
`EvaluationConfig` (`config.py:110`).

### 4.8 explainability (`training/explainability/`)

`facade.py`, `gradcam.py`, `shap_explainer.py`, `integrated_gradients.py`,
`cross_modal_attention.py`, `temporal_attention.py`, `uncertainty.py`,
`counterfactual.py`, `visualization.py`, `exporter.py`, `report_generator.py`.

### 4.9 export / inference / runtime / feature_store

- `export/` — `JsonExporter`, `ParquetExporter`, `TorchExporter`, `export_dataset`
  (`builder.py:34`), `records.py` helpers.
- `inference/` — `package_builder.py` (`build_inference_package`), `exporter.py`,
  `validate.py`, `versioning.py`, `dataset_sources.py`.
- `runtime/` — `InferenceRuntime` (`runtime.py:49`), `ModelLoader`/`ModelHealth`
  (`model_loader.py:55`), `PreprocessLoader`, `MetadataLoader`, `RuntimeCache`,
  `HealthMonitor`/`HealthReport`, `ReleasePackager`/`ReleaseReport`
  (`packager.py:99`), `ReleaseManager`/`RuntimeState` (`release_manager.py:75`),
  `ReleaseValidator` (`validation.py:83`), `ReleaseLayout` (`layout.py:179`).
- `feature_store/` — `FeatureStoreManager`, `ObservationCache`, `PatchCache`,
  `TabularFeatureCache`, `TemporalSequenceCache`, `ImageEmbeddingGenerator`,
  `CacheFingerprint`/`CacheInvalidator`.

### 4.10 quality / mlops / kaggle

- `quality/` — drift (`feature/label/prediction/spatial/temporal`), fairness
  (`evaluator`, `metrics`, `regional`), monitoring (`dashboard`, `exporters`),
  optimization (`benchmark`, `onnx_runtime`, `runtime`), samples (`report`).
- `mlops/` — `registry.py`, `experiments.py`, `gates.py`, `scheduler.py`,
  `reports.py`, `cli.py`.
- `kaggle/` — `config.py` (`PathsConfig`, `KaggleConfig`, `LoggingConfig`,
  `WorkspaceConfig`, `WorkspaceLayout`), `workspace.py` (`WorkspaceManager`,
  `CheckpointManager`, `TrainingCache`), `validation.py` (`TrainingValidator`,
  `ValidationResult`), `environment/` (`EnvironmentManager`, GPU/deps/system/
  runtime), `logging.py` (`TrainingLogger`), `reports.py`, `cache.py`,
  `checkpoints.py`, `setup.py`; scripts `bootstrap.py`, `system_check.py`,
  `run_training.py`, `run_pipeline.py`, `evaluate.py`, `export_release.py`.

---

## 5. Config files under `training/config/`

| File | Content |
|---|---|
| `paths.yaml` | Relative paths for logs/outputs/checkpoints/cache/configs + `extends` chain. |
| `dataset.yaml` | `DatasetManager.Settings` (providers incl. `kaggle_hub_image`, cache, metadata). |
| `training.yaml` | `TrainingConfig` (general/data/optimizer/scheduler/loss/train/checkpoint/metrics/validation/ablation/benchmark/visualization/curriculum). |
| `model.yaml` | `ModelConfig` defaults for the architecture. |
| `stam.yaml` | **Added during onboarding repair** — STAM tabular mapping: `table: data_season.csv`, `village_column: Location`, `season_column: Season`, `year_column: Year`, `crop_column: Crops`, `yield_column: yeilds`, `feature_columns: [Area, Rainfall, Temperature, Soil type, Irrigation, Humidity, price]`, `season_file: seasons.yaml`. |
| `seasons.yaml` | Project season calendar: Kharif (Jun–Oct), Rabi (Nov–Mar), Zaid (Apr–May). |
| `validation.yaml` | `TrainingValidator` thresholds (python/gpu/deps/disk). |

> `training/stam/seasons.yaml` (Kharif/Rabi/Summer) is intentionally left
> untouched — STAM's own tests assert it. The pipeline uses
> `training/config/seasons.yaml` via `stam_cfg.temporal.season_file`.

---

## 6. Kaggle vs local execution — verified differences

| Concern | Kaggle | Local (`D:\CropPrep`) |
|---|---|---|
| Imagery source | `/kaggle/input/...` mount (detected by `KaggleHubImageProvider`; B4 fix) | No mount → `kaggle_hub_image` manifest `available=false` → imagery stage warns; chain degrades. |
| Image metadata | `generate_image_metadata()` produces records | Same code path, but zero rasters → zero records. |
| STAM initialize | Full indexes over catalog | Initializes fine: seasons `['Kharif','Rabi','Zaid']`, calendar `training/config/seasons.yaml`, years 2004–2019 (from `data_season.csv`). *observed* |
| Plan | Locations/cells present | `{'locations': 0, 'years': 2–16 (per season), 'seasons': 2–3, 'cells': 0}` *observed* |
| Corpus | accepted observations → training runs | 0 accepted → training skipped with recorded reason *observed* |
| GPU | T4/P100 available | `GPU_UNAVAILABLE` (CPU-only torch, no CUDA) *observed* |
| Training | Full `Experiment.run()` executes | Only reachable if imagery is materialised locally (rasterio is installed; no Sentinel-2 rasters present). |

**GAP**: the imagery-dependent chain (image metadata → STAM centroids → patches →
`CropFusionDataset` → `Trainer`) is only verifiable on Kaggle. Locally the
pipeline runs end-to-end but ends at corpus resolution + validation.

---

## 7. Data-flow traces

### 7.1 Tabular: CSV → STAM → observations

`DatasetManager` (via `DatasetManagerTabularSource`) reads
`data_season.csv` → `SpatialTemporalMatcher.match_tabular` builds
`TabularFeatures` (Location/Season/Year/Crops/yeilds + feature columns) →
`STAM.build_observation` assembles an `AgriculturalObservation` (geo + temporal +
tabular + sequence + quality). STAM config in `training/config/stam.yaml` makes
the column mapping explicit and fixes the earlier `ST-RESOLVE-001: No sample
years available` (the STAM defaults referenced `village/district/season/year/
crop/yield` which do not exist in the tabular file).

### 7.2 Imagery: rasters → patches → tensors

`DatasetManagerImageSource` yields `MetadataRecord`s (CRS, resolution, bounds,
date) → `SpatialPatchGenerator` (KD-tree spatial index + boundary index) maps a
`(lon, lat)` to raster windows (`world_to_pixel`, `patch_window`, window padding
for out-of-bounds) → `RasterPatch(array, mask)` → `STAM.get_patch` →
`ImagePipeline.transform_patch` → per-pair NDVI/EVI tensors →
`TemporalPipeline.transform_sequence` → `[T,1,H,W]` + `temporal_mask`.

### 7.3 Preprocess → model contract

`Preprocessor.transform` returns the exact dict `CropFusionModel.forward`
consumes: `tabular [B,F]`, `ndvi/evi [B,T,1,H,W]`, `temporal_mask [B,T]`.
`ModelConfig.from_preprocessor` infers `F` from the fitted
`TabularPipeline` (numeric_dim + categorical cardinalities), `num_classes` from
`LabelPipeline`, `input_size` from `config.image.size`, `max_len` from
`temporal.max_observations`. `validate_batch` re-checks shapes at runtime.

### 7.4 Observed local run (recorded in `training/kaggle/outputs/reports/pipeline.json`)

Environment report captured: CPU-only, no CUDA. Imagery unavailable warning.
STAM initialized (3 seasons, 2004–2019). Corpus: 0 locations / 0 cells /
0 accepted → training `skipped` (`no accepted observations`). Validation result
with non-critical issues (`GPU_UNAVAILABLE`, `kaggle_hub_image` provider
unavailable). `pipeline.json` written; exit code 0.

---

## 8. Tests and validation state

Suites verified this audit:

| Suite | Result |
|---|---|
| `dataset_manager` (tests/) + `training` (tests/) | 361 pass |
| `stam` (tests/) | all pass (exit 0) |
| `kaggle` (tests/) | 67 pass; one flaky checkpoint test (`test_checkpoints.py`) passed on re-run — not a regression |
| `models`, `preprocessing`, `evaluation`, `export`, `inference` | 468 pass combined (all) |
| `runtime` (tests/) | **129 pass (all)** — the 4 pre-existing failures from §9.7 were fixed |

Re-run command: `python -m pytest training/dataset_manager/tests training/training/tests training/stam/tests training/kaggle/tests` (adjust scope as needed; there is no single lint/typecheck command configured).

Run: `python -m pytest training/dataset_manager/tests training/training/tests
training/stam/tests training/kaggle/tests` (adjust to your suite scope; there is
no single lint/typecheck command configured — confirm with the maintainer before
relying on one).

---

## 9. Bugs, gaps and how they were verified

1. **STAM defaults vs real tabular schema (fixed).** Defaults
   (`village/district/season/year/crop/yield`) didn't match `data_season.csv`
   columns; `plan()` raised `ST-RESOLVE-001: No sample years available`.
   Fixed by `training/config/stam.yaml` (see §5). Verified: `run_pipeline.py`
   runs twice locally, `stam` tests green.
2. **Season-file path resolution (fixed).** The relative `season_file` was
   resolved against CWD. `run_pipeline.py` now resolves it against the STAM
   config's directory (`run_pipeline.py:175-183`). Verified by repeated local
   runs.
3. **Bootstrap path fixes (done earlier).** B1/B2/B5 path-resolution +
   `config_file` handling in the two config loaders; B3 `sys.path` hardening in
   the five Kaggle scripts; B4 `/kaggle/input` mount support in
   `KaggleHubImageProvider`; B10 verified correct (all callers already construct
   `ObservationResolver(stam, ...)`).
4. **GAP — end-to-end training not runnable without mounted imagery.** The
   image-dependent chain (§6) only executes on Kaggle. Locally the corpus is
   empty by design.
5. **GAP — no shipped launch script for the Experiment on Kaggle.** The
   Experiment path in `run_pipeline.py` is complete, but a full training run
   still requires either `run_pipeline.py` (with imagery attached) or a notebook
   invoking it. `run_training.py` is intentionally readiness-only
   (`No model / no data required`).
6. **Flaky test.** `training/kaggle/tests/test_checkpoints.py` occasionally
   fails under a full-suite run but passes standalone — timing/temp-file
   sensitivity, not a code defect.

7. **Pre-existing runtime test failures — FIXED this audit.** Four
   `training/runtime/tests/` failures (all in unmodified committed code, none in
   the R2.4 pipeline) were repaired; the suite now passes 129/129:
   - `test_load_metadata` (`test_model_loader.py:61`) — stale assertion. The
     packager stored `formats` as artifact *paths* (`packager.py:288-296`,
     `["model/cropfusion.pt"]`) while the test asserted the format *name*
     `["pytorch"]`. **Fix:** `ReleasePackager._build_model_metadata` now writes
     format *names* (mirroring the release manifest's own
     `formats` at `packager.py:423-431`), so `model_loader.load_metadata()`
     and `ReleaseInfo.formats` / `layout.formats` all report the same names.
   - `test_rollback_after_restart` (`test_release_manager.py`) —
     `FileExistsError` on `cropfusion_release-v1.1.0`. Order-dependent:
     `test_rollback_flow` created that dir in the module-scoped `releases_root`
     and never cleaned up.
   - `test_status_with_releases` (`test_release_manager.py`) — same pollution:
     the leftover v1.1.0 made `versions()` return `['1.1.0', '1.0.0']`.
     **Fix (both):** `test_rollback_flow` and `test_rollback_after_restart` now
     clone the base release into their own `tmp_path` root, so they never mutate
     the shared module fixture.
   - `test_onnx_format_requires_onnxruntime` (`test_validation.py`) —
     referenced the module-global `release_validator` fixture object (a
     `FixtureFunctionDefinition`) instead of requesting it as a parameter.
     **Fix:** added `release_validator` to the test signature.

---

## 10. Run steps and expected outputs

### Local (degraded end-to-end)

```
python training/kaggle/scripts/run_pipeline.py --repo-root . --years 2018,2019 --seasons Kharif,Rabi --max-cells 4
```

Expected: STAM initializes; corpus resolves (0 accepted locally); training
skipped with reason; `pipeline.json` written to
`training/kaggle/outputs/reports/`; exit 0. Non-critical validation issues
`GPU_UNAVAILABLE` + `kaggle_hub_image` unavailable are expected.

### Kaggle (full training)

```
!python training/kaggle/scripts/run_pipeline.py   # imagery attached under /kaggle/input
```

Expected: imagery ensured → metadata generated → corpus with accepted
observations → `Experiment.run()` → checkpoint + report under the run dir;
`pipeline.json` reflects `training.status == "completed"`.

### Other entry points

- `python -m training.dataset_manager --help` — dataset CLI (~35 subcommands).
- `python training/kaggle/scripts/system_check.py` — environment report.
- `python training/kaggle/scripts/run_training.py` — readiness-only
  orchestration report (`orchestration.json`).
- `python -m training.mlops --help` — experiment registry/gates/scheduler CLI.

---

## 11. Debugging guide

- **"ST-RESOLVE-001: No sample years available"** → the STAM `tablular.yaml`
  mapping no longer matches the CSV columns. Fix `training/config/stam.yaml`
  (`year_column`/`season_column`/`village_column`), not the bundled
  `stam/seasons.yaml`.
- **Season calendar looks wrong** → check `stam_cfg.temporal.season_file`
  resolution: it is anchored to `training/config/`, not CWD
  (`run_pipeline.py:175-183`).
- **`import training` resolves to the wrong package** → a `training/` folder
  earlier in `sys.path` (e.g. stale `/kaggle/working/training`). Run through
  `run_pipeline.py`/`_add_repo_root`, or delete the stale folder.
- **Training skipped "no accepted observations"** → no imagery mount/materialised
  catalog; check the `imagery` stage of `pipeline.json` and the
  `kaggle_hub_image` provider manifest.
- **Model/checkpoint won't build** → verify `ModelConfig` matches a fitted
  preprocessor (use `ModelFactory.from_preprocessor`, not hand-built
  `numeric_dim`/`cardinalities`).
- **Batch shape errors** → the batch contract is
  `tabular[B,F]`, `ndvi/evi[B,T,1,H,W]`, `temporal_mask[B,T]`; `validate_batch`
  will name the offending key.
- **GPU_UNAVAILABLE locally** → expected; everything still runs on CPU up to the
  imagery-dependent stage. For GPU debugging use Kaggle.
- **Flaky checkpoint test** → run `pytest training/kaggle/tests/test_checkpoints.py`
  standalone; if it passes, the failure is environmental, not code.
