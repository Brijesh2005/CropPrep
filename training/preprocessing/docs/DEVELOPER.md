# Preprocessing — Developer Guide

## Layout

```
ai/preprocessing/
├── __init__.py          # public API
├── master_pipeline.py   # Preprocessor facade
├── tabular_pipeline.py  # TabularPipeline
├── image_pipeline.py    # ImagePipeline
├── temporal_pipeline.py # TemporalPipeline
├── label_pipeline.py    # LabelPipeline
├── transforms.py        # scalers + encoders
├── augmentations.py     # ImageAugmentation
├── validators.py        # quality filtering
├── dataset.py           # CropFusionDataset + split_observations
├── dataloader.py        # build_dataloader + collate
├── statistics.py        # DatasetStatistics
├── config.py            # PreprocessingConfig
├── interfaces.py        # Transformer / Pipeline ports
├── utils.py             # tensor/padding helpers
├── exceptions.py        # PP-* errors
├── logger.py
├── README.md / docs/
└── tests/
```

## Conventions

| Thing | Convention |
|-------|------------|
| Modules | `snake_case.py` |
| Classes | `PascalCase` (pipelines end in `Pipeline`) |
| Public API verbs | `fit / transform / fit_transform / validate / summary / save / load` |
| Error codes | `PP-<AREA>-<NNN>` |
| Env vars | `PRE_<SECTION>__<FIELD>` |

## Data-access rule

The preprocessing package receives `AgriculturalObservation` objects from
STAM **only**. It never opens raw CSVs/GeoTIFFs. The only data reads are via
the injected **patch extractor** (`STAM.get_patch`) used by the image
pipeline.

## Adding a transform

1. Implement `Transformer` in `transforms.py` (fit/transform/to_dict/from_dict).
2. Register it in the config `scaler`/`categorical_encoding` options if user
   selectable.

## Adding a pipeline stage

1. Implement `Pipeline` (fit/transform/validate/summary/save/load).
2. Wire it into `Preprocessor` (`master_pipeline.py`) and the sample dict.
3. Add tests under `tests/`.

## Testing

```bash
cd ai/preprocessing
pytest
```

Observations come from a real STAM run over the synthetic dataset (see
`tests/conftest.py`), so the integration path is exercised without mocks for
the data chain.

## Performance notes

* **Lazy** patch extraction in `__getitem__` (no eager materialisation).
* Train-only fit; transforms reuse fitted parameters for val/test.
* `minmax` image normalization avoids patch reads at fit time.
* DataLoader supports workers/pin_memory/prefetch/persistent_workers.
