# Inference Package Generator (Phase R5)

`PackageBuilder` assembles a **versioned, self-validating, consumer-side
inference package**: the exported model artifacts, the resolved configuration,
input schema metadata, runtime-side code (the same `training.inference`
modules + a no-training-dependency adapter), reproducibility metadata, checksums
and a manifest — then hands it to the validator.

## Layout

```
<package_dir>/cropfusion-<model_version>/
├── cropfusion.pt  cropfusion.torchscript.pt  cropfusion.onnx
├── model_config.yaml · input_schema.json · metrics.json · metadata.json
├── requirements.txt · api.py · inference_adapter.py · README.md
├── manifest.json · checksums.json
```

- **Required artifacts** (14) — one per bundle format selected by
  `exporter.formats` (pytorch always; torchscript/onnx optional).
- **Manifest** — a semver-triple, the `source_repo` URL, a `description`, the
  artifact `checksums` (from `checksums.json`) and the `formats` list. Both
  `checksums.json` and `manifest.json` exclude themselves from their own
  checksum map (no self-referential hashing).
- **`metadata.json`** — model fingerprint + export formats for end-to-end
  verification (`InferenceReport.fingerprint`).

## Usage

```python
from training.inference import PackageBuilder, InferenceConfig
from training.inference import dataset_sources as sources

config = InferenceConfig(
    versioning={"package_name": "cropfusion", "major": 1, "minor": 0, "patch": 0},
    dataset_sources={"sources": {"train": "gs://bucket/...", "eval": "..."}},
)
builder = PackageBuilder(metadata={"framework": "pytorch", "framework_version": "2.13.0"})
result = builder.build(model, sample_batch, "artifacts/inference", config)

result.package_dir    # resolved versioned directory
result.manifest       # BuildManifest
result.reports_dir    # validation reports live here
result.artifact_checksums
```

`ModelExporter.export_bundle` does the model export; `source_repo` comes from
`versioning.source_repo` or is auto-detected via `git remote get-url origin`
at build time.

## Versioning

`next_version(existing_version)` yields a semver bump with `(major, minor,
patch)`-increment disambiguation (`(1,0,0)` → `1.0.1`, `(0,1,0)` →
`1.1.0`, `(0,0,1)` → `2.0.0`). An explicit `major/minor/patch` in
`InferenceConfig.versioning` wins; otherwise the package is numbered from
`1.0.0` and existing package directories at the same name are scanned to pick
the next free version. A stale fingerprint between a new build and a prior
package of the same version raises `VersionConflictError`.

## Validation

Every built package is validated before use — see
`training/inference/validate.py`: `integrity`, `manifest`, `compatibility` and
a `smoke_test` that builds a batch straight from the model's own config (no
external `sample_batch` needed). `package_validator` wires the four checks into
one `validate_package(package_dir)` entry point; the individual checks
(`validate_integrity`, `validate_manifest`, `validate_compatibility`,
`smoke_test`) stay public. Failures raise `PackageValidationError` with the
check name, the artifact key and a human-readable detail. A full validation run
is recorded in `validation_report.md` / `.json` (`integrity`, `manifest`,
`compatibility`, `smoke_test` results + version + fingerprint).
