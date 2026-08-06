# Dataset Export — R2.3 Artifacts

`training/export/` writes the generated feature frame to portable training
artifacts. One normalised table → JSON, NDJSON, Parquet or PyTorch payloads,
plus a `manifest.json` describing everything written.

## Pipeline

```python
from training.feature_engineering import build_feature_frame
from training.export import ExportConfig, export_dataset

frame = build_feature_frame(corpus)
artifacts = export_dataset(
    frame,
    corpus=corpus,                      # attaches sample_id / year / season / quality_score
    config=ExportConfig(
        output_dir="data/out/datasets",
        formats=["json", "parquet", "torch"],
        prefix="cropfusion",
    ),
)
# data/out/datasets/cropfusion.{json,parquet,pt} + manifest.json
```

## Formats

| Format | Writer | Artifact | Payload |
| --- | --- | --- | --- |
| `json` | `JsonExporter` | `*.json` | array of JSON-safe records (`NaN` → `null`) |
| `jsonl` | `JsonExporter.export_jsonl` | `*.jsonl` | NDJSON stream, one record per line |
| `parquet` | `ParquetExporter` | `*.parquet` | pandas `to_parquet` (pyarrow engine) |
| `torch` | `TorchExporter` | `*.pt` | `{sample_id, features: tensor, feature_names, n_samples}` |

The torch payload stacks only **numeric** feature columns into a `float32`
tensor, so a `DataLoader` can consume it directly. Non-numeric (label / meta)
columns stay available in the JSON / Parquet artifacts.

## Metadata

With `include_meta=True` (default) and a corpus supplied, `attach_meta` adds:

- `sample_id` — the resolver sample id (stable identity across formats),
- `quality_score` — the per-cell STAM quality score,
- `year` / `season` / `location_id` — the sampling-cell keys.

This makes the artifacts self-describing: downstream consumers never need the
corpus object or the Dataset Manager.

## Configuration

`ExportConfig` loads via `load_export_config` (**env > YAML > defaults**):

| Env | Purpose |
| --- | --- |
| `EX_OUTPUT_DIR` | artifact directory |
| `EX_FORMATS` | JSON array, e.g. `["json","parquet","torch"]` |
| `EX_PREFIX` | file-name prefix |
| `EX_INCLUDE_META` / `EX_INCLUDE_QUALITY` | metadata toggles |
| `EX_WRITE_MANIFEST` | write `manifest.json` |

`save_export_template(path)` writes an annotated YAML template.

## Manifest

`manifest.json` records `generated_at`, `prefix`, `rows`, `columns` and a
`formats → path` map — the single source of truth for what a dataset release
contains (used by the MLOps registry / release export in R3).

## Error codes

| Code | Class | Meaning |
| --- | --- | --- |
| `EX-CONFIG-001` | `ExportConfigError` | Invalid export config / missing YAML |
| `EX-FORMAT-001` | `ExportFormatError` | Unsupported format requested |
| `EX-WRITE-001` | `ExportWriteError` | Artifact could not be written (incl. missing pyarrow/torch) |

Loggers live under `cropfusion.export.*`.

See `training/export/tests/test_export.py` (17 tests) for the contract.
