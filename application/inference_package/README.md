# Inference Package

The **inference package** is the immutable, versioned artifact set the
Prediction Platform serves predictions from. It is **consumed, never
generated** by `application/` — the Training Platform export pipeline produces
it and deployment ships it into this directory.

R1.4 prepares this directory as a contract: the expected file layout, the
validator that checks it (`application/inference/validation`), and the loader
that will read it (`application/inference/loaders`). **No inference is
implemented and no model is loaded in R1.4.**

## Expected files

| File | Kind | Required | Purpose |
| --- | --- | --- | --- |
| `metadata.db` | database | yes | SQLite store for inference metadata / history lookups |
| `historical_context.parquet` | data | yes | Long-run historical context (climatology / seasonality) |
| `location_index.parquet` | data | yes | Index of known locations for reverse geocoding |
| `feature_scalers.pkl` | serialized | yes | Fitted feature scalers used by preprocessing |
| `label_encoder.pkl` | serialized | yes | Fitted crop-label encoder |
| `model_config.yaml` | config | yes | Model architecture configuration |
| `dataset_version.json` | metadata | yes | Dataset version used to train the model |
| `model_version.json` | metadata | yes | Model version / checksum / status |
| `metrics.json` | metadata | yes | Evaluation metrics recorded at export time |
| `README.md` | docs | no | Human-readable package description |

The canonical list is defined once in
[`manifest.py`](manifest.py) and shared by the validator and the loader.

## Model weights

The trained weights are **not** part of the package; they live alongside it in
`application/models/`:

- `models/cropfusion.pt` — the current model (R1.4 expectation, not yet shipped)
- `models/cropfusion_v1.pt`, `models/cropfusion_v2.pt`, … — future pinned versions
- `models/cropfusion_latest.pt` — future "latest" symlink/copy

Version resolution is defined by `application/inference/versioning` and is not
implemented yet.

## Lifecycle

1. **Training** exports a model + context into a staging directory.
2. **Packaging** assembles the files above into a versioned package.
3. **Deployment** copies the package + weights into this directory (see
   `application/docker/Dockerfile.inference.standalone`).
4. **Validation** (`application/inference/validation`) checks the manifest.
5. **Serving** (future) loads via `application/inference/loaders`.

The `.gitignore` in this directory keeps generated artifacts out of version
control; only the manifest and this README are committed.
