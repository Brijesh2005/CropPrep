# CropFusion Inference Package Guide

The **inference package** is the immutable, versioned artifact set the
Prediction Platform serves predictions from. The Training Platform exports it;
`application/` consumes it. R1.4 fixes the contract.

## Files

| File | Kind | Purpose |
| --- | --- | --- |
| `metadata.db` | database | inference metadata / history lookups |
| `historical_context.parquet` | data | long-run climatology / seasonality |
| `location_index.parquet` | data | known locations for reverse geocoding |
| `feature_scalers.pkl` | serialized | fitted feature scalers |
| `label_encoder.pkl` | serialized | fitted crop-label encoder |
| `model_config.yaml` | config | model architecture configuration |
| `dataset_version.json` | metadata | dataset version used to train |
| `model_version.json` | metadata | model version / checksum / status |
| `metrics.json` | metadata | evaluation metrics at export time |
| `README.md` | docs | human-readable package description |

The canonical list lives once in
`application/inference_package/manifest.py` and is shared by:

- the validator (`application/inference/validation`),
- the loader (`application/inference/loaders`),
- this documentation.

## Who produces / consumes

| Step | Owner | Action |
| --- | --- | --- |
| Export | Training Platform | emit model + context into a staging dir |
| Package | Training Platform / CI | assemble the files into a versioned package |
| Ship | Deployment | copy package + weights into `application/` |
| Validate | `InferencePackageValidator` | check the manifest |
| Load | `ModelLoader` | read the artifacts into a `ModelPackage` |
| Serve | `InferenceEngine` | run the model |

`application/inference_package` never generates anything; its `.gitignore`
keeps the real artifacts out of version control.

## Lifecycle diagram

See [r1-4-inference-package-flow](../diagrams/r1-4-inference-package-flow.md).

## Validation

The validator returns a `shared.validation.ValidationResult`:

- every `required` artifact exists and has the expected kind;
- `model_version.json` parses as a `shared.versioning.ModelVersion`;
- weights referenced in the model config resolve under `application/models`.

R1.4 ships the port only; the check logic is a later-phase implementation.
