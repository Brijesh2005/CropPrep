"""Tests for the PyTorch dataset and dataloader."""

from __future__ import annotations

import numpy as np

from ai.preprocessing import (
    CropFusionDataset,
    Preprocessor,
    build_dataloader,
    collate_samples,
    split_observations,
)


def test_dataset_len_and_getitem(observations, extractor, preprocessing_config):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    dataset = CropFusionDataset.build(preprocessor, observations, split="train",
                                      extractor=extractor)
    assert len(dataset) == len(observations)
    sample = dataset[0]
    assert sample["observation_id"]
    assert sample["ndvi"].dim() == 4
    assert sample["temporal_mask"].dim() == 1


def test_torch_dataset_adapter(observations, extractor, preprocessing_config):
    import torch

    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    dataset = CropFusionDataset.build(preprocessor, observations, split="test",
                                      extractor=extractor)
    adapter = dataset.torch_dataset
    assert isinstance(adapter, torch.utils.data.Dataset)
    assert len(adapter) == len(observations)
    item = adapter[0]
    assert isinstance(item["tabular"], torch.Tensor)


def test_split_then_build(observations, extractor, preprocessing_config):
    train, val, test = split_observations(observations, preprocessing_config.split)
    preprocessor = Preprocessor(preprocessing_config).fit(train, extractor=extractor)
    train_ds = CropFusionDataset.build(preprocessor, train, split="train", extractor=extractor)
    test_ds = CropFusionDataset.build(preprocessor, test, split="test", extractor=extractor)
    assert len(train_ds) == len(train)
    assert len(test_ds) == len(test)


def test_dataloader_batch(observations, extractor, preprocessing_config):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    dataset = CropFusionDataset.build(preprocessor, observations, split="train",
                                      extractor=extractor)
    loader = build_dataloader(
        dataset, preprocessing_config, split="train",
        batch_size=2, workers=0,
    )
    batch = next(iter(loader))
    assert set(batch.keys()) == {
        "observation_id", "metadata", "tabular", "ndvi", "evi",
        "temporal_mask", "crop_label", "yield_label",
    }
    assert batch["tabular"].shape[0] == 2
    assert batch["ndvi"].shape[0] == 2
    assert batch["ndvi"].shape[1] == preprocessing_config.temporal.max_observations
    assert batch["crop_label"].shape[0] == 2
    assert batch["yield_label"].shape[0] == 2
    assert len(batch["metadata"]) == 2


def test_collate_manual():
    import torch

    sample = {
        "observation_id": "x",
        "tabular": torch.ones(3),
        "ndvi": torch.ones(8, 1, 4, 4),
        "evi": torch.zeros(8, 1, 4, 4),
        "temporal_mask": torch.ones(8),
        "crop_label": torch.tensor(0),
        "yield_label": torch.tensor(1.0),
        "metadata": {"year": 2020},
    }
    batch = collate_samples([sample, sample])
    assert batch["tabular"].shape == (2, 3)
    assert batch["crop_label"].shape == (2,)


def test_statistics(observations, extractor, preprocessing_config, tmp_path):
    preprocessor = Preprocessor(preprocessing_config).fit(observations, extractor=extractor)
    dataset = CropFusionDataset.build(preprocessor, observations, split="val",
                                      extractor=extractor)
    report = dataset.statistics(output_dir=tmp_path)
    assert report["total_observations"] == len(observations)
    assert report["class_distribution"]
    assert (tmp_path / "dataset_statistics.json").exists()
    assert report["sequence_length_distribution"]["count"] == len(observations)
