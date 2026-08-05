# Migration Report — R1.1 → R1.2 (Dataset Manager Provider Pattern)

Detailed record of the R1.2 refactor: turning the Dataset Manager into a
multi-source data access layer behind a **provider pattern**, so the Training
Platform reads tabular data from Git and satellite imagery from Kaggle through
two interchangeable, independently-testable providers. Companion to
[MIGRATION_REPORT](MIGRATION_REPORT.md) (the monolith → two-platform report) and
[MIGRATION_GUIDE](MIGRATION_GUIDE.md).

- **Status**: complete
- **Date**: 2026-08-05
- **Branch**: `main` (not yet committed)

## 1. Scope and principles

- **Additive, no deletions**: the existing Dataset Manager API
  (`download / scan / validate / generate_metadata / summary / inventory /
  list_csvs / load_csv / list_images / preview_csv /
  get_historical_context`) is unchanged and still passes its full test suite.
- **Strict dependency rules** (enforced by the layout, verified by tests):
  `training → shared`; `DatasetManager → providers`; `providers →
  independent` (no imports from the manager — no circular dependencies).
- **No direct file access by the manager**: every dataset read happens through
  a provider. Providers reuse the existing engines
  (`PandasCSVLoader`, `RasterioImageLoader`, `KaggleDownloader`,
  `DatasetScanner`, `DatasetValidator`,
  `MetadataGeneratorImpl` / `SQLiteMetadataStore`) — no new engines.
- **Kaggle data stays off-GitHub**: imagery is downloaded-or-reused by the
  provider into `training/datasets/raw/` (git-ignored); the small Git CSVs are
  versioned at `training/datasets/tabular/`.
- **No algorithm changes**: STAM, preprocessing, models, losses, training and
  evaluation are untouched. `HistoricalContextBuilder` still reads through the
  manager.

## 2. New components

### Provider layer — `training/dataset_manager/providers/`

| File | Responsibility |
| --- | --- |
| `models.py` | Shared data contracts: `ProviderManifest`, `TabularCatalog`, `TabularJoinSpec`, `ImageCatalog`, `ImageDatasetLocation`, `PatchRequest`. |
| `base.py` | `Provider`, `TabularProvider`, `ImageProvider` ABCs + `ProviderStatus`. |
| `git_tabular.py` | `GitRepositoryTabularProvider` — fnmatch CSV discovery, schema/statistics/missing profiling, join (merge on every spec), missing-value handling (drop / fill mean / fill constant). |
| `kaggle_image.py` | `KaggleHubImageProvider` — download-or-reuse, NDVI/EVI catalog, lazy `read_metadata` / windowed `read` / `patch`, `get_historical_context`, integrity verify, `SQLiteMetadataStore` fallback. |
| `__init__.py` | Re-exports base ABCs + models; lazy `__getattr__` imports of the two concrete providers (avoids heavy raster/Kaggle imports at import time). |

### Manager wiring — `training/dataset_manager/manager.py`

- `__init__` now constructs both providers: `tabular_provider`
  (`GitRepositoryTabularProvider`) and `image_provider`
  (`KaggleHubImageProvider`).
- `download()` delegates to `image_provider.ensure(force, materialize)`.
- New delegating section (50+ methods): `tabular_catalog / tabular_names /
  load_tabular / stream_tabular / tabular_schema / validate_tabular_schema /
  tabular_statistics / tabular_missing / handle_missing_tabular / join_tabular /
  tabular_metadata / ensure_image / image_location / image_catalog /
  validate_image / generate_image_metadata / discover_ndvi / discover_evi /
  read_image / patch_image / image_historical_context / provider_manifests`.
- `info()` now includes a `"providers"` section.

### Configuration — `training/dataset_manager/config.py`

New Pydantic sections: `TabularProviderConfig` (root, patterns, chunk_size,
join_suffixes), `ImageProviderConfig` (handle, catalog_name, materialize,
verify_integrity, link_method, force_download), `ProviderConfig` wrapping both,
`Settings.providers`, and the `Settings.tabular_root` property.

### CLI — `training/dataset_manager/cli.py`

New subcommands: `tabulars`, `tabular-schema`, `tabular-statistics`,
`image-ensure`, `image-catalog`, `image-patch`, `providers`.

### Versioned tabular datasets — `training/datasets/tabular/`

Five CSVs copied from `Tabular_Datasets/` and committed:
`cropdata_updated.csv`, `ICRISAT-District Level Data.csv`, `data_season.csv`,
`dataset.csv`, `All-India_-Crop-wise-Area,-Production-&-Yield (2).csv`.

## 3. Kaggle workspace — `training/kaggle/`

| File | Responsibility |
| --- | --- |
| `scripts/bootstrap.py` | Environment + install + provider-manifest readiness gate (no model code). |
| `scripts/run_training.py` | Config loading → data ensure → STAM observations → `run_experiment`. |
| `scripts/evaluate.py` | Checkpoint reload → hold-out test split → `Evaluator`. |
| `scripts/export_release.py` | TorchScript / ONNX export + `release.json` manifest. |
| `notebooks/train.ipynb` | Kaggle orchestration: bootstrap → run_training. |
| `notebooks/evaluate.ipynb` | Kaggle orchestration: bootstrap → evaluate. |
| `notebooks/export.ipynb` | Kaggle orchestration: bootstrap → export_release. |

## 4. Platform configuration — `training/config/`

| File | Loader | Notes |
| --- | --- | --- |
| `dataset.yaml` | `dataset_manager.config.load_settings` | `providers.tabular.root`, `providers.image.*`, download/validate/… |
| `training.yaml` | `training.training.config.load_training_config` | general/data/optimizer/scheduler/loss/train/checkpoint/metrics/logging/validation/ablation/benchmark/visualization. |
| `model.yaml` | `training.models.config.load_model_config` | tabular/image_encoder/image_fusion/temporal/cross_attention/gated_fusion/shared_encoder/heads/loss/checkpoint/export. |
| `logging.yaml` | `load_settings` | `logging:` section (LogConfig fields). |
| `validation.yaml` | `stam.config.load_stam_config` | patch/spatial/temporal/image/quality/cache/admin/tabular/seasons. |
| `kaggle.yaml` | plain `yaml.safe_load` | Kaggle runtime paths + editable install list. |

## 5. Verification

| Check | Result |
| ----- | ------ |
| `pytest training/dataset_manager/tests` | 155 passed (incl. new `test_providers.py`, updated `test_cli.py`) |
| New provider tests | discovery, missing-root → `MISSING_DATA`, load/schema/statistics/missing, handle_missing (drop/fill), join, stream, metadata; image location/catalog/discover/read/patch/historical context/validate/generate_metadata/ensure/manifest/path-rejection; manager + CLI delegation |
| `image-ensure` CLI test removed | triggered a real Kaggle download + 120s timeout — kept out of CI |
| Config load smoke test | all 6 YAMLs load through their loaders |
| `py_compile` Kaggle scripts | exit 0 |
| `bootstrap.py --help` / `evaluate.py --help` / `export_release.py --help` | argparse OK |
| `bootstrap.py` against real configs | tabular provider READY (5 datasets), image provider `not_initialized` (no download without `--ensure-data`) |
| Notebooks (`train/evaluate/export.ipynb`) | valid nbformat 4 JSON |

## 6. Intentional leftovers

- Logger identifiers `cropfusion.dataset_manager.*` /
  `cropfusion.spatial_alignment.*` remain runtime logger names (self-consistent,
  not imports).
- `training/config/kaggle.yaml` has no pydantic loader — it is consumed by the
  Kaggle scripts via `yaml.safe_load` (runtime paths, not pydantic settings).
- Kaggle scripts and notebooks are orchestration scaffolds: the heavy engines
  live in `training/{dataset_manager, stam, preprocessing, training, models}`.
  Full end-to-end training on Kaggle is exercised by the notebooks.

## 7. Rollback / safety

Nothing was deleted and no algorithm was changed. Providers are additive
behind the existing manager API; removing them reverts to R1.1 behaviour. All
data stays recoverable from Git (`training/datasets/tabular/`) or re-downloaded
from Kaggle (`training/datasets/raw/`, git-ignored).
