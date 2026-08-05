# CropFusion Preprocessing Pipeline (Phase 4)

Converts **`AgriculturalObservation`** samples (produced by STAM) into
**AI-ready tensors** consumed directly by PyTorch DataLoaders — the input for
the Phase 5 multimodal AI architecture.

No AI component touches raw CSVs or GeoTIFFs: every sample passes through
STAM first, then through this pipeline.

---

## What it does

```
AgriculturalObservation (STAM)
        │
        ▼
Quality filtering (reject low-quality / invalid / unpaired)
        │
        ▼
Leakage-free split (temporal / spatial / group / stratified / random)
        │
        ▼
Preprocessor.fit(train)   ← scalers, encoders, image stats fitted on TRAIN ONLY
        │
        ▼
Preprocessor.transform(obs)  → AI-ready sample dict
   { observation_id, tabular, ndvi, evi, temporal_mask,
     crop_label, yield_label, metadata }
        │
        ▼
CropFusionDataset + build_dataloader → PyTorch batches
```

## Quick start

```python
from training.preprocessing import (
    Preprocessor, CropFusionDataset, build_dataloader, split_observations,
)

preprocessor = Preprocessor.from_config()            # or Preprocessor(config)
accepted, decisions = preprocessor.filter(observations)

train, val, test = split_observations(accepted, preprocessor.config.split)
preprocessor.fit(train, extractor=stam.get_patch)    # TRAIN ONLY

train_ds = CropFusionDataset.build(preprocessor, train, split="train", extractor=stam.get_patch)
val_ds   = CropFusionDataset.build(preprocessor, val,   split="val",   extractor=stam.get_patch)

train_loader = build_dataloader(train_ds, preprocessor.config, split="train", batch_size=32)
for batch in train_loader:
    # batch["tabular"] [B, F]
    # batch["ndvi"]    [B, T, 1, H, W]
    # batch["evi"]     [B, T, 1, H, W]
    # batch["temporal_mask"] [B, T]
    # batch["crop_label"] [B], batch["yield_label"] [B]
    ...
```

## Per-modality pipelines

| Pipeline | Output | Notes |
|----------|--------|-------|
| `TabularPipeline` | `[F]` float tensor | missing values, outliers, scaler (standard/minmax/robust), encoder (onehot/ordinal), constant removal, correlation drop |
| `ImagePipeline` | `[1, H, W]` per patch | NDVI/EVI normalize (minmax/standard/identity), NaN/invalid, clip, resize/pad |
| `TemporalPipeline` | `[T,1,H,W]`, `[T,1,H,W]`, `[T]` mask | sort, dedupe, truncate, pad, mask |
| `LabelPipeline` | crop label, yield label | label/onehot crop encoding, yield scaling |

## Data splitting

* `temporal` — whole years assigned (most recent → test). No temporal leakage.
* `spatial` / `group` — whole villages assigned. No spatial leakage.
* `stratified` — per-class ratio split.
* `random` — seeded shuffle.

## Augmentation (training only)

`ImageAugmentation` applies random flip / rotation / crop / brightness /
contrast / noise to `[T,1,H,W]` sequences — gated on the `train` split.

## Statistics

`DatasetStatistics.summarize(observations)` produces a `StatisticsReport`
(class distribution, yield distribution, sequence lengths, missing values,
feature stats, patch stats) persisted as JSON.

## Persistence

`Preprocessor.save(dir)` / `load(dir)` round-trips all fitted artifacts
(scalers, encoders, image stats, label encoder, temporal params).

## Tests

```bash
cd training/preprocessing
pytest
```

## Docs

* [Pipeline diagram](docs/PIPELINE.md)
* [Sequence diagram](docs/SEQUENCE.md)
* [Usage examples](docs/USAGE.md)
* [Configuration guide](docs/CONFIGURATION.md)
* [Developer guide](docs/DEVELOPER.md)
