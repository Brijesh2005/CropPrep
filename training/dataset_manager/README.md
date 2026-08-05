# CropFusion Dataset Management System (DMS)

The **single access point** for every dataset in the CropFusion platform. No
other module (AI, GIS, backend, frontend) reads CSVs or GeoTIFFs directly —
they all communicate with the Dataset Manager.

Part of the **CropFusion** research platform:
*Hybrid ML–DL spatio-temporal cross-modal fusion for multi-crop
recommendation and yield prediction* (Phase 2 of the Software Design
Document).

---

## What it does

| Capability | Module |
|-----------|--------|
| Automatic Kaggle download (detect / re-download / verify) | `downloader.py` |
| Parallel directory scanning & inventory (CSV / GeoTIFF / NDVI / EVI / R10m / R20m) | `scanner.py` |
| Validation (structure, duplicates, empty CSV, corrupt TIFF, CRS, metadata) | `validator.py` |
| Metadata generation + SQLite store (+ Parquet export) | `metadata.py` |
| Dataset registry (versions, paths, status, checksum) | `dataset_registry.py` |
| Semantic versioning (major/minor/patch, rollback) | `version_manager.py` |
| SQLite-backed cache with TTL + prefix invalidation | `cache_manager.py` |
| CSV loading (discovery, schema, missing values, statistics, streaming) | `csv_loader.py` |
| Lazy GeoTIFF access (header metadata, windowed reads) | `image_loader.py` |
| Validated YAML + env configuration | `config.py` |
| Structured, rotating JSON logging | `logger.py` |
| Custom exceptions with stable error codes | `exceptions.py` |
| Command-line interface | `cli.py` / `manage_dataset.py` |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r training/dataset_manager/requirements.txt

# 2. Download the primary Kaggle dataset
python training/dataset_manager/manage_dataset.py download

# 3. Scan, validate and index it
python training/dataset_manager/manage_dataset.py scan
python training/dataset_manager/manage_dataset.py validate
python training/dataset_manager/manage_dataset.py metadata

# 4. Inspect
python training/dataset_manager/manage_dataset.py summary
python training/dataset_manager/manage_dataset.py inventory --json
python training/dataset_manager/manage_dataset.py versions
```

Or from Python:

```python
from training.dataset_manager import DatasetManager

manager = DatasetManager.from_config()   # honours DM_* env vars / YAML config
manager.download()
report = manager.validate()              # -> ValidationReport
manager.generate_metadata()              # -> int (records written)

summary = manager.summary()
df = manager.load_csv(manager.list_csvs()[0])
meta = manager.image_metadata(manager.list_images()[0])
```

---

## Primary dataset

The primary image dataset is downloaded **automatically** via `kagglehub`
(no manual downloading):

```text
shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada
```

- Downloads are cached by kagglehub and **reused** when present.
- Use `manage_dataset.py download --force` to re-download.
- The materialised copy lives under `<dataset_root>/raw/kaggle-crop-yield/`.

## Tabular datasets

Multiple CSV files are supported (Crop, Yield, Weather, Rainfall, Soil,
Fertilizer, Production, Historical Yield, Season, Village, District, ...).
**No filenames are hardcoded** — `list_csvs()` / `csvs` discovers every CSV
under the dataset root automatically.

---

## Configuration

Configuration resolves, in precedence order:

1. **Environment variables** — `DM_` prefix, `__` separates nesting
   (`DM_DOWNLOAD__KAGGLE_HANDLE`, `DM_SCAN__WORKERS`, ...).
2. **YAML file** — `--config path.yaml` or `DM_CONFIG_FILE`.
3. **Defaults** — sensible built-in values (validated by pydantic).

Generate a template:

```bash
python training/dataset_manager/manage_dataset.py config-template dm.yaml
```

Key defaults:

| Setting | Default |
|---------|---------|
| `dataset_root` | `./datasets` |
| `download.kaggle_handle` | `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada` |
| `download.materialize` | `true` |
| `scan.workers` | `8` |
| `validation.expected_years` | `[2018, 2025]` |
| `metadata.store_type` | `sqlite` |
| `cache.default_ttl_seconds` | `86400` |
| `logging.level` | `INFO` |

---

## CLI reference

```
download           Download (or reuse) the primary Kaggle dataset
scan               Scan the dataset and build an inventory
validate           Validate structure, integrity and metadata
metadata           Generate metadata records for all files
inventory          Print the full file inventory
summary            Print a dataset summary
csvs               List discovered CSV files
images             List GeoTIFF files (--index NDVI/EVI, --resolution, --year)
register           Register the dataset in the registry
versions           List dataset version history
bump-version       Bump the version (major/minor/patch)
rollback           Roll back to a snapshotted version
cache-stats        Show cache statistics
info               Show environment and configuration info
config-template    Write an annotated YAML config template
```

Every subcommand supports `--json` for machine-readable output.

---

## Storage layout

```
<dataset_root>/
├── raw/                     # canonical copy of downloads
│   └── kaggle-crop-yield/   # materialised Kaggle dataset
├── processed/               # derived / cleaned datasets (future phases)
└── .cropfusion/             # internal state
    ├── metadata.db          # metadata records (SQLite)
    ├── registry.db          # registry + version history (SQLite)
    ├── cache.db             # scan/inventory cache (SQLite)
    └── cache/               # cached artifacts
```

---

## Design notes

* **Why SQLite for metadata** — incremental upserts, unique-path indexing,
  per-file point lookups (STAM) and zero extra dependencies. Parquet remains
  available for analytics via `export_metadata_parquet()`.
* **Lazy by default** — raster headers are read without touching pixel data;
  CSVs are profiled in streaming chunks.
* **Cache-aware scanning** — a cheap tree signature `(count, size, mtime)`
  avoids re-scanning unchanged directories.
* **Ports & adapters** — every capability implements an interface in
  `interfaces.py`; `DatasetManager` depends only on those ports.

---

## Tests

```bash
cd training/dataset_manager
pytest
```

The suite uses **mocked downloads** and synthetic GeoTIFFs — it never touches
the network or your real Kaggle cache.

## Documentation

* [Architecture](docs/ARCHITECTURE.md)
* [Installation](docs/INSTALL.md)
* [Usage](docs/USAGE.md)
* [Developer guide](docs/DEVELOPER.md)
