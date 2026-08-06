"""Integration: Phase 4 preprocessor -> DataLoader batch -> model forward.

Exercises the exact production path: STAM observations -> Preprocessor
(ordinal encoding) -> CropFusionDataset -> build_dataloader -> CropFusionModel.
"""

from __future__ import annotations

import pytest
import torch

from training.preprocessing import CropFusionDataset, build_dataloader
from training.models import ModelFactory


@pytest.fixture
def pipeline(preprocessor_ordinal, stam_chain):
    """A fitted ordinal preprocessor + its STAM extractor."""
    return preprocessor_ordinal, stam_chain


def test_dataloader_batch_feeds_model(pipeline):
    preprocessor, stam = pipeline

    model = ModelFactory.from_preprocessor(
        preprocessor,
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 16},
    )

    observations = stam.build_observation(74.801, 13.099, year=2020, season="Kharif")
    # a single observation, repeated through the dataset so we can batch it
    samples = [observations] * 4

    dataset = CropFusionDataset.build(
        preprocessor, samples, split="train", extractor=stam.get_patch
    )
    loader = build_dataloader(
        dataset, config=preprocessor.config, split="train",
        batch_size=2, shuffle=False,
    )
    batch = next(iter(loader))

    # batch contract
    assert batch["tabular"].shape[1] == model.config.tabular_feature_dim
    assert batch["ndvi"].shape[2] == 1
    assert batch["temporal_mask"].shape == batch["ndvi"].shape[:2]

    # forward + gradient
    model.train()
    out = model(batch)
    assert out.crop_logits.shape[0] == 2
    assert out.crop_logits.shape[1] == preprocessor.label.num_classes
    assert out.yield_pred.shape == (2, 1)
    assert out.shared_representation.shape[0] == 2
    assert set(out.gates) == {"image_gate", "tabular_gate", "fusion_gate"}
    (out.crop_logits.sum() + out.yield_pred.sum()).backward()
    assert sum(1 for p in model.parameters() if p.grad is not None) > 0


def test_ordinal_schema_derived(pipeline):
    preprocessor, _ = pipeline
    config = ModelFactory.build_config(
        preprocessor,
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 16},
    )
    assert config.tabular.numeric_dim == 1
    assert len(config.tabular.categorical_cardinalities) == 2
    assert config.heads.crop.num_classes == 2
    # model consumes the same F the preprocessor emits
    assert config.tabular_feature_dim == len(preprocessor.tabular.feature_names)


def test_temporal_mask_marks_padding(pipeline):
    preprocessor, stam = pipeline
    obs = stam.build_observation(74.801, 13.099, year=2020, season="Kharif")
    sample = preprocessor.transform(obs, extractor=stam.get_patch, augment=False)
    mask = sample["temporal_mask"]
    # 2020 has 3 observations; the rest is padding
    assert int(mask.sum().item()) == 3
    assert mask.shape[0] == preprocessor.config.temporal.max_observations
