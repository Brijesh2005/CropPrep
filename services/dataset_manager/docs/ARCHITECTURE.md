# Dataset Manager — Architecture

## Overview

The Dataset Manager is a **hexagonal (ports & adapters) package** inside
`services.dataset_manager`. `DatasetManager` is the facade every other module
talks to; it depends only on the abstract interfaces in `interfaces.py`.

```
                    ┌────────────────────────────────────────────┐
                    │          DatasetManager (facade)           │
                    │  download · scan · validate · metadata      │
                    │  inventory · summary · load_csv · load_image│
                    │  registry · versions · cache                │
                    └───┬──────┬──────┬──────┬──────┬─────────────┘
                        │      │      │      │      │
        interfaces.py   ▼      ▼      ▼      ▼      ▼
                ┌─────────┐ ┌──────┐ ┌────────┐ ┌──────┐ ┌──────────┐
                │Scanner  │ │Validator│ │Metadata│ │Registry│ │  Cache   │
                └────┬────┘ └──┬───┘ └───┬────┘ └──┬───┘ └────┬─────┘
                     │         │         │         │          │
        adapters     ▼         ▼         ▼         ▼          ▼
                ┌─────────┐ ┌──────┐ ┌─────────────┐ ┌──────┐ ┌──────────┐
                │Dataset  │ │Dataset│ │SQLiteMetadata│ │SQLite│ │CacheManager│
                │Scanner  │ │Validator│ │   Store     │ │Registry│ │ (SQLite)  │
                └─────────┘ └──────┘ └─────────────┘ └──────┘ └──────────┘
                     │
                ┌────▼──────────────┐        ┌──────────────────────┐
                │  CSVLoader /      │        │  KaggleDownloader    │
                │  ImageLoader      │        │  (kagglehub)         │
                └───────────────────┘        └──────────────────────┘
```

## Data flow (pipeline)

```
 Kaggle dataset ──► downloader ──► raw/kaggle-crop-yield/
      │
      ├─► scanner  ──► DatasetInventory (CSV/GeoTIFF classification, cached)
      │
      ├─► metadata generator ──► SQLite metadata store (──► Parquet export)
      │
      ├─► validator ──► ValidationReport (JSON)
      │
      ├─► registry + version manager (status, checksum, semver)
      │
      └─► loader API (load_csv / load_image)   ◄── used by AI/GIS/backend
```

## Module responsibilities

| Module | Responsibility |
|--------|----------------|
| `manager.py` | Facade + orchestration + path enforcement (only reads inside root) |
| `downloader.py` | kagglehub download, existing-download detection, materialisation, integrity pre-flight |
| `scanner.py` | Recursive discovery, file classification, parallel scanning, cache-aware inventories |
| `validator.py` | Structure / duplicates / empty CSV / corrupt TIFF / CRS / metadata checks |
| `metadata.py` | Per-file metadata records; SQLite store (upserts, queries); Parquet export |
| `dataset_registry.py` | Dataset lifecycle, paths, status, checksum, provenance |
| `version_manager.py` | Semantic versioning (MAJOR.MINOR.PATCH), snapshots, rollback |
| `cache_manager.py` | SQLite key/value cache, TTL, prefix invalidation, capacity eviction |
| `csv_loader.py` | Discovery, schema inference, missing values, statistics, streaming reads |
| `image_loader.py` | Lazy GeoTIFF header metadata, previews, windowed reads (rasterio + light parser) |
| `config.py` | YAML + env + defaults with pydantic validation |
| `logger.py` | Structured JSON + rotating file logging |
| `exceptions.py` | Typed errors with stable codes (`DM-*`) |
| `utils.py` | Hashing, parallel map, classification helpers, streaming utilities |
| `interfaces.py` | Abstract ports implemented by every adapter |
| `models.py` | Shared dataclasses + enums (inventory, records, reports, ...) |
| `cli.py` / `manage_dataset.py` | Command-line adapter over the facade |

## Dependency rules

* `manager.py` depends only on `interfaces.py` + concrete factories.
* Adapters never import `manager.py` (no cycles).
* `utils.py` and `models.py` are dependency-free leaves.
* `metadata`/`registry`/`cache` share one SQLite helper (`_db.py`) — no
  duplicated connection logic.
* All read access is **gated** by the manager (`_resolve_within_root`), so no
  caller can escape the managed dataset root.

## Error handling

Every failure raises a subclass of `DatasetManagerError` carrying a stable
`code` (e.g. `DM-DL-001`, `DM-VALID-001`). The CLI maps any unhandled
exception to a non-zero exit and a JSON error body when `--json` is set.
