# Preprocessing — Usage Examples

## 1. End-to-end training-data pipeline

```python
from services.dataset_manager import DatasetManager
from services.spatial_alignment import STAM
from ai.preprocessing import (
    Preprocessor, CropFusionDataset, build_dataloader, split_observations,
)

# 1. Dataset Manager
manager = DatasetManager.from_config()
manager.download()
manager.generate_metadata()

# 2. STAM → observations
stam = STAM(manager)
stam.initialize()
observations = [
    stam.build_observation(74.802, 13.098, year=y, season="Kharif")
    for y in (2020, 2021)
    for lon, lat in [(74.801, 13.099), (74.803, 13.097)]
]

# 3. Preprocessing
preprocessor = Preprocessor.from_config("preprocessing.yaml")
accepted, decisions = preprocessor.filter(observations)

train, val, test = split_observations(accepted, preprocessor.config.split)
preprocessor.fit(train, extractor=stam.get_patch)       # TRAIN ONLY

train_ds = CropFusionDataset.build(preprocessor, train, split="train", extractor=stam.get_patch)
val_ds   = CropFusionDataset.build(preprocessor, val,   split="val",   extractor=stam.get_patch)

train_loader = build_dataloader(train_ds, preprocessor.config, split="train", batch_size=32)
```

## 2. Inspect one AI-ready sample

```python
sample = preprocessor.transform(train[0], extractor=stam.get_patch)
print(sample["tabular"].shape)      # [F]
print(sample["ndvi"].shape)         # [T, 1, H, W]
print(sample["temporal_mask"])      # [T] 1=real, 0=padded
print(sample["crop_label"], sample["yield_label"])
print(sample["metadata"])
```

## 3. Fit/transform/persist

```python
samples = preprocessor.fit_transform(train, extractor=stam.get_patch)
preprocessor.save("artifacts/preprocessor")          # scalers, encoders, ...
restored = Preprocessor.load("artifacts/preprocessor")
```

## 4. Statistics report

```python
report = DatasetStatistics.summarize(train, extractor=stam.get_patch, patch_size=32)
report.save("artifacts/stats")
print(report.class_distribution)      # {crop: count}
print(report.sequence_length_distribution)
```

Or via the dataset:
```python
report_dict = train_ds.statistics(output_dir="artifacts/stats")
```

## 5. Manual collate

```python
from ai.preprocessing import collate_samples
batch = collate_samples([sample, sample])
```

## 6. Config file (`preprocessing.yaml`)

```yaml
image: {size: 224, normalize: minmax, ndvi_range: [-1, 1], evi_range: [-1, 1]}
tabular:
  scaler: robust
  categorical_encoding: onehot
  exclude_columns: [crop, yield_kg, village, district, year, season]
temporal: {max_observations: 8, min_observations: 1, pad_value: 0.0}
label: {crop_encoding: label, yield_scaler: standard}
split: {strategy: temporal, test_years: [2024, 2025], val_years: [2023]}
dataloader: {batch_size: 32, workers: 4, pin_memory: true}
augmentation: {enabled: true, flip_horizontal: true, noise_std: 0.01}
quality: {min_quality_score: 40.0, require_crop_label: true}
```
