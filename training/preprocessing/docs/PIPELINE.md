# Preprocessing — Pipeline Diagram

```
                    ┌────────────────────────────────────────────┐
                    │       AgriculturalObservation (STAM)       │
                    └───────────────────┬────────────────────────┘
                                        │
              ┌─────────────────────────▼─────────────────────────┐
              │  validators.py  Quality filtering                 │
              │  min_quality_score · coordinates · labels · pairs │
              └─────────────────────────┬─────────────────────────┘
                                        │ (accepted only)
              ┌─────────────────────────▼─────────────────────────┐
              │  dataset.py split_observations (no leakage)       │
              │  temporal · spatial/group · stratified · random   │
              └─────────────────────────┬─────────────────────────┘
                                        │
                       ┌────────────────┴────────────────┐
                       │                                 │
              ┌────────▼─────────┐              ┌────────▼─────────┐
              │   fit on TRAIN   │              │   train/val/test │
              └────────┬─────────┘              │  observation sets│
                       │                        └────────┬─────────┘
                       ▼                                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Preprocessor (master_pipeline.py)           │
   │                                                                 │
   │  TabularPipeline ──► [F] tabular tensor                         │
   │  ImagePipeline   ──► per-patch [1,H,W] (extractor → patches)   │
   │  TemporalPipeline──► [T,1,H,W] + [T,1,H,W] + [T] mask          │
   │  LabelPipeline   ──► crop_label, yield_label                    │
   │  Augmentation    ──► train-only geometric/photometric           │
   └─────────────────────────────────────────┬───────────────────────┘
                                             │  AI-ready sample dict
                                             ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │  CropFusionDataset (dataset.py)   lazy __getitem__ per sample   │
   │  build_dataloader (dataloader.py) batch/workers/pin/prefetch    │
   └─────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
                                 PyTorch DataLoader batches
                                 (input to Phase 5 AI architecture)
```

## Module map

| Module | Role |
|--------|------|
| `master_pipeline.py` | `Preprocessor` facade: fit/transform/fit_transform/validate/summary/save/load |
| `tabular_pipeline.py` | `TabularPipeline` |
| `image_pipeline.py` | `ImagePipeline` (per-patch) |
| `temporal_pipeline.py` | `TemporalPipeline` (sequence assembly) |
| `label_pipeline.py` | `LabelPipeline` |
| `transforms.py` | scalers + encoders (persistable) |
| `augmentations.py` | `ImageAugmentation` (train only) |
| `validators.py` | `filter_observation(s)` quality gate |
| `dataset.py` | `CropFusionDataset` + `split_observations` |
| `dataloader.py` | `build_dataloader` + `collate_samples` |
| `statistics.py` | `DatasetStatistics` / `StatisticsReport` |
| `config.py` | `PreprocessingConfig` (env > YAML > defaults) |
| `utils.py` / `interfaces.py` / `exceptions.py` / `logger.py` | cross-cutting |
