# Statistics, Reports & Extended Validation

R2.2 adds aggregate **statistics**, seven JSON **report** families and extended
**validation** checks (temporal / spatial / CRS / provider). Everything flows
through the Dataset Manager — no direct file access.

## Statistics

`DatasetManager.statistics()` combines tabular column statistics with the image
catalog's year / index / resolution counts:

```python
stats = manager.statistics()
stats.total_tabular_rows
stats.tabular            # {name: {column: stats}}
stats.tabular_row_counts # {name: n}
stats.images_by_year     # {2019: 2, 2020: 1}
stats.images_by_index    # {"NDVI": ..., "EVI": ...}
stats.images_by_resolution
stats.total_images
```

## Reports

`generate_reports(manager, report_dir)` writes one JSON file per family:

| Report | Contents |
| --- | --- |
| `inventory` | Every scanned file, categorised |
| `csv` | Per tabular dataset: rows, columns, size, missing values |
| `image` | NDVI / EVI counts by year + resolution, dataset location |
| `provider` | Registered providers: discovery, availability, health, capabilities |
| `spatial` | Spatial index metadata + locations |
| `temporal` | Index x year x resolution availability (persisted or derived) |
| `validation` | Full validation report (structure / integrity / metadata) |

```python
from training.dataset_manager.reports import generate_reports

paths = generate_reports(manager)   # writes <state_root>/reports/*_report.json
```

One failing report family is serialised as `{"kind": ..., "error": ...}` so it
never blocks the others.

## Extended validation

The R1.2 validator is extended with R2.2 checks (skipped when their dependency
is not wired, e.g. no spatial index):

| Code | Check |
| --- | --- |
| `V-TEMP-001` | Duplicate observation records (index/resolution/year/date) |
| `V-TEMP-002` | Expected year range not fully covered by imagery |
| `V-SPAT-001/002` | Latitude / longitude out of range |
| `V-SPAT-003` | Duplicate spatial locations sharing a name |
| `V-CRS-001` | Rasters use mixed coordinate systems |
| `V-META-004` | Metadata store contains duplicate observation records |
| `V-PROV-001` | A registered provider is unavailable |

```python
report = manager.validate()
report.passed
report.issues            # [ValidationIssue, ...]
for issue in report.issues:
    issue.code           # e.g. "V-PROV-001"
    issue.severity
    issue.category
    issue.message
    issue.detail         # structured payload (files, missing_years, ...)
```

## CLI

```bash
python -m training.dataset_manager statistics
python -m training.dataset_manager reports --dir ./reports
python -m training.dataset_manager validate
python -m training.dataset_manager search --query NDVI
```
