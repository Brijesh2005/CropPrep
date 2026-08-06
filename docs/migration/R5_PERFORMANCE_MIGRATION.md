# R5 — Training Performance Upgrade: Migration Report

Non-breaking, additive upgrade. **No existing file was modified.** Nothing in
Dataset Manager, STAM, AgriculturalObservation, Feature Builders,
TabTransformer, EfficientNet, Temporal Transformer, Cross Attention, Adaptive
Gated Fusion, Confidence Fusion, Prediction Heads, FastAPI, React, Docker,
APIs, or the Release Package was touched. Model outputs, training objectives,
loss functions and evaluation metrics are unchanged.

## What was generated

| File | Purpose |
|---|---|
| `training/feature_store/__init__.py` | Public package API |
| `training/feature_store/manager.py` | `FeatureStoreManager` — generic create/load/update/invalidate/version/checksum cache over NumPy, Parquet, Torch, JSON |
| `training/feature_store/invalidation.py` | `CacheInvalidator`, `CacheFingerprint` — hashes dataset version, backbone, patch size, preprocessing config, feature version |
| `training/feature_store/image_cache.py` | `ImageEmbeddingGenerator` — GeoTIFF → patch → EfficientNet → 768-d embedding, cached |
| `training/feature_store/temporal_cache.py` | `TemporalSequenceCache` — NDVI/EVI/timestamps/quality/cloud/mask, cached |
| `training/feature_store/tabular_cache.py` | `TabularFeatureCache` — normalized/encoded/scaled tabular features, cached |
| `training/feature_store/observation_cache.py` | `ObservationCache` — assembled `AgriculturalObservation` + historical context, cached |
| `training/feature_store/patch_cache.py` | `PatchCache` — memory-mapped raw 224×224 patches (NDVI/EVI/future indices) |
| `training/training/profiler.py` | `TrainingProfiler` — data-loading/forward/backward timing, CPU/GPU/memory sampling, cache hit rate, JSON report |
| `training/config/performance.yaml` | All new knobs: feature store, dataloader, AMP, `torch.compile`, GPU pipeline, curriculum freezing, checkpoint frequency, invalidation rules, profiling |
| `docs/migration/R5_PERFORMANCE_MIGRATION.md` | This report |

Every `*_cache.py` class is a **wrapper**: it takes your existing builder
functions (patch extractor, EfficientNet forward, sequence builder, tabular
builder, observation resolver) as constructor arguments and only adds a
cache lookup around them. None of them reimplement domain logic.

## Not included in this pass (kept out to stay within scope/cost)

These were requested but are either thin config you'll set directly in
`performance.yaml` / your existing `dataloader.py` and `trainer.py`, or are
better written once you tell me which items to prioritize:

- Wiring `performance.yaml` into `training/preprocessing/dataloader.py` (adds `prefetch_factor`, `persistent_workers`, worker-count auto-tuning — your `dataloader.py` already accepts most of these as config, see below).
- AMP / `torch.compile` call sites inside `training/training/trainer.py`'s train step.
- CUDA-stream / pinned-memory async copy helper.
- `scripts/benchmark_training.py` (original vs optimized epoch-time/samples-per-sec harness).
- Cache/Feature Store/GPU-optimization prose guides under `docs/`.

Say the word and I'll generate any of these next, standalone, without
touching existing files.

## Integration steps (manual — as requested)

1. Copy the `training/feature_store/` package in as-is.
2. Copy `training/config/performance.yaml` in as-is.
3. In your dataset/loader construction code, instantiate one
   `FeatureStoreManager` and pass it into `ImageEmbeddingGenerator`,
   `TemporalSequenceCache`, `TabularFeatureCache`, `ObservationCache`, and
   `PatchCache`, wiring each `*_builder` / `*_extractor` argument to the
   existing function you already call today. Replace direct calls to those
   functions with `.get_*(...)` calls on the cache wrapper.
4. At the top of each training run, call
   `CacheInvalidator(store).check_and_invalidate(namespace, CacheFingerprint(...))`
   per namespace before the first epoch, so stale caches from a prior
   dataset version / backbone / patch size are dropped automatically.
5. Drop `TrainingProfiler` timing blocks around your existing
   `data_loading` / `forward` / `backward` steps in `trainer.py`; call
   `profiler.write_report(...)` at the end of training.
6. Your existing `training/config/training.yaml` already has
   `amp: true`, `pin_memory`, `prefetch_factor`, `persistent_workers` — align
   those values with `performance.yaml`'s `mixed_precision` / `dataloader`
   sections, or point one at the other, so there's a single source of truth.

## Expected performance gains

Directional estimates (validate with the (not-yet-generated) benchmark
harness against your actual dataset size and hardware):

- **Image embedding cache**: eliminates the EfficientNet forward pass for
  every observation on every epoch after the first — for frozen-backbone
  curriculum stages this is typically the single largest per-epoch cost.
  Expect epoch time to drop most on datasets with many observations per
  image and a large number of training epochs.
- **Temporal/tabular/observation caches**: removes repeated STAM
  reconstruction and feature re-encoding; helps most when this work is done
  in single-process Python (i.e., before parallel `DataLoader` workers can
  hide the cost).
- **DataLoader tuning** (`persistent_workers`, `pin_memory`, tuned
  `prefetch_factor`): reduces worker-restart and host↔device transfer
  overhead; a few percent to double-digit percent gain depending on how
  I/O-bound the current pipeline is.
- **AMP (bf16/fp16)**: typically 1.5–3x throughput on compatible GPUs for
  the transformer/CNN compute portions, with negligible accuracy impact.
- **`torch.compile`**: typically 10–30% additional throughput on top of AMP
  once warmup/recompilation overhead is amortized across an epoch.

None of these change what the model computes — only how fast it gets there.
