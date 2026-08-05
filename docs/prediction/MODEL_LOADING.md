# CropFusion Model Loading Guide

How the inference-only runtime loads a trained model — and why R1.4 does not.

## The rule

The Prediction Platform **loads exported artifacts, never training code**. It
reads versioned weights (`application/models`) and the sidecar files in the
inference package (`application/inference_package`), then runs them. It does
not construct models from the Training Platform at serving time.

## Expected layout

```
application/models/
  cropfusion.pt              current model (expected, not yet shipped)
  cropfusion_v1.pt           future pinned version
  cropfusion_v2.pt           future pinned version
  cropfusion_latest.pt       future "latest" copy / symlink

application/inference_package/
  metadata.db                inference metadata store
  historical_context.parquet long-run context (climatology / seasonality)
  location_index.parquet     known locations for reverse geocoding
  feature_scalers.pkl        fitted feature scalers
  label_encoder.pkl          fitted crop-label encoder
  model_config.yaml          model architecture config
  dataset_version.json       dataset version used to train
  model_version.json         model version / checksum / status
  metrics.json               evaluation metrics at export time
```

## Loading sequence (future)

1. **Validate** — `InferencePackageValidator.validate(package_dir)` checks the
   manifest (`application/inference_package/manifest.py`).
2. **Resolve version** — `ModelVersionResolver.resolve(pinned)` returns the
   `shared.versioning.ModelVersion` to serve (`latest` default).
3. **Load** — `ModelLoader.load(package_dir, pinned=...)` returns a
   `ModelPackage` bundling weights path, model config, scaler, encoder, device.
4. **Warm up** — one forward pass at startup (config: `model.warmup`).
5. **Serve** — `InferenceEngine.predict` uses the loaded package.

## Why R1.4 does not load anything

- The exported artifacts do not exist yet (training export is not wired).
- Loading would embed torch + training model code at serving time, which is
  exactly what the inference-only split is meant to avoid.
- The ports are the deliverable: they fix *how* loading will happen so the
  later implementation has no design ambiguity.

## Version resolution

- Default: `cropfusion_latest.pt` / `cropfusion.pt`
- Pinned: `cropfusion_{version}.pt`
- Metadata source: `model_version.json` (semantic version + checksum + status)

Version types come from `shared.versioning.ModelVersion`.

See [r1-4-model-loading-flow](../diagrams/r1-4-model-loading-flow.md).
