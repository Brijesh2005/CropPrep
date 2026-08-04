# Preprocessing — Sequence Diagram

## Fit phase (train only)

```
 User          Preprocessor      TabularPipeline  ImagePipeline  TemporalPipeline  LabelPipeline   Extractors(STAM)
  │ fit(train, extractor)            │              │               │                │                │
  │─────────────────►│                │              │               │                │                │
  │                  │ tabular.fit   │              │               │                │                │
  │                  │──────────────►│              │               │                │                │
  │                  │ image.fit     │              │               │                │                │
  │                  │───────────────│─────────────►│               │                │                │
  │                  │               │  sample patches───────────────│────────────────│───────────────►│
  │                  │ temporal.fit  │              │               │                │                │
  │                  │───────────────│──────────────│──────────────►│                │                │
  │                  │ label.fit     │              │               │                │                │
  │                  │───────────────│──────────────│───────────────│───────────────►│                │
  │  fitted          │               │              │               │                │                │
  │◄─────────────────│               │              │               │                │                │
```

## Transform phase (per sample, lazy)

```
 Dataset.__getitem__(i)     Preprocessor.transform       Patch extractor      PyTorch
      │                          │                          │                  │
      │ transform(obs)           │                          │                  │
      │────────────────────────►│                          │                  │
      │                          │ tabular.transform(obs)  │                  │
      │                          │─────────────────────────│                  │
      │                          │ for each pair:          │                  │
      │                          │  extractor(ndvi_path, lon, lat, size)      │
      │                          │──────────────────────────────────────────►│
      │                          │  transform_patch(NDVI)  │                  │
      │                          │  extractor(evi_path,...)│                  │
      │                          │──────────────────────────────────────────►│
      │                          │ temporal.transform_sequence(...)          │
      │                          │ label.transform(obs)   │                  │
      │                          │ (augment if train)     │                  │
      │  sample dict             │                          │                  │
      │◄────────────────────────│                          │                  │
      │                          │                          │                  │
      ▼                          │                          │                  │
  collate_samples(batch) ───────────────────────────────────────────────────► batch tensors
```

## Key properties

* **Lazy** — patches are read only in `__getitem__`; nothing is precomputed.
* **Train-only fit** — scalers/encoders/image stats are computed once on the
  training split, then reused for val/test.
* **No leakage** — splits assign whole years / whole groups; fit never sees
  val/test statistics.
* **Deterministic** — seeded splits and fixed tensor shapes.
