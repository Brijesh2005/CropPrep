# Usage Examples

## CLI

```bash
# Download (or reuse) the primary Kaggle dataset
python services/dataset_manager/manage_dataset.py download

# Force a fresh download
python services/dataset_manager/manage_dataset.py download --force

# Scan and cache the inventory
python services/dataset_manager/manage_dataset.py scan
python services/dataset_manager/manage_dataset.py scan --refresh

# Validate (writes validation_report.json under .cropfusion/)
python services/dataset_manager/manage_dataset.py validate
python services/dataset_manager/manage_dataset.py validate --json

# Generate metadata records
python services/dataset_manager/manage_dataset.py metadata

# Inspect the dataset
python services/dataset_manager/manage_dataset.py summary
python services/dataset_manager/manage_dataset.py inventory --json
python services/dataset_manager/manage_dataset.py csvs
python services/dataset_manager/manage_dataset.py images --index NDVI --year 2020

# Registry + versioning
python services/dataset_manager/manage_dataset.py versions
python services/dataset_manager/manage_dataset.py bump-version minor --message "added 2021 imagery"
python services/dataset_manager/manage_dataset.py rollback 1.0.0

# Diagnostics
python services/dataset_manager/manage_dataset.py info
python services/dataset_manager/manage_dataset.py cache-stats
```

## Python API

### Pipeline

```python
from services.dataset_manager import DatasetManager

manager = DatasetManager.from_config()

# 1. Download
manager.download()                 # reuses existing download when present
manager.download(force=True)       # re-download

# 2. Scan
inventory = manager.scan()         # cached on second call
inventory = manager.scan(refresh=True)

# 3. Validate
report = manager.validate()
print(report.passed, report.by_severity())
for issue in report.failing_issues():
    print(issue.severity, issue.code, issue.message)

# 4. Metadata
written = manager.generate_metadata()          # number of new records
print(manager.metadata_count())

# 5. Summary
print(manager.summary().to_dict())
```

### Reading data (the only access paths)

```python
# CSVs
csv_files = manager.list_csvs()
df = manager.load_csv(csv_files[0])
chunks = manager.load_csv(csv_files[0], chunksize=10_000)   # streaming
head = manager.preview_csv(csv_files[0], n_rows=5)

# GeoTIFFs
ndvi = manager.list_images(index_type="NDVI", year=2022)
meta = manager.image_metadata(ndvi[0])          # lazy, header only
patch = manager.load_image(ndvi[0], window=(0, 0, 3, 3))   # bounded window
```

### Metadata queries

```python
# Per-file record
record = manager.get_metadata(ndvi[0])
print(record.crs, record.pixel_size, record.sha256)

# Filtered queries
evi_2021 = manager.query_metadata(index_type="EVI", year=2021)
all_r10m = manager.query_metadata(resolution="R10m")

# Analytics export
manager.export_metadata_parquet("metadata.parquet")
```

### Registry + versioning

```python
manager.register()
print(manager.registry_entries())

manager.bump_version("minor", message="added new year")
print(manager.current_version())
manager.bump_version("patch")
print(manager.list_versions())
manager.rollback_version("0.1.0")
```

### Cache

```python
manager.cache_set("stam:features:village-42", {...}, ttl_seconds=3600)
value = manager.cache_get("stam:features:village-42")
manager.cache_invalidate("scan:")     # drop all scan inventories
manager.cache_clear()
print(manager.cache_stats())
```

### Custom configuration

```python
from services.dataset_manager import DatasetManager, Settings

settings = Settings(
    dataset_root="/data/cropfusion/datasets",
    download={"kaggle_handle": "owner/name", "materialize": True},
    scan={"workers": 16, "hash_files": True},
    logging={"level": "DEBUG", "dir": "/var/log/cropfusion"},
)
manager = DatasetManager(settings)
```

### Error handling

```python
from services.dataset_manager.exceptions import (
    DatasetManagerError, DownloadFailedError, DatasetNotFoundError,
)

try:
    manager.download()
except DownloadFailedError as exc:
    print(f"download failed [{exc.code}]: {exc}")
except DatasetManagerError as exc:
    print(f"generic dataset error [{exc.code}]: {exc}")
```

## Integration points for future phases

* **STAM / AI module** — use `query_metadata(year=..., index_type=...)`,
  `load_image(window=...)` and `load_csv()` to assemble observation samples.
* **GIS module** — read village/geometry CSVs via `list_csvs()`.
* **Backend API** — expose `registry_entries()`, `summary()`, `validate()`
  and `inventory()` over HTTP.
* **Frontend dashboards** — consume `summary()` / `registry_status()` for the
  Dataset Management dashboard.
