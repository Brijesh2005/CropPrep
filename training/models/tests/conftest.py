"""Shared fixtures for the AI model test-suite.

A small, fast configuration (tiny timm backbone at 32x32) keeps every test
light while still exercising the full architecture. Batch sizes are >= 2
because the BatchNorm-based backbones require >1 sample per channel in train
mode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Make the repository root importable regardless of where pytest runs from.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.models import ModelConfig, ModelFactory


def small_config() -> ModelConfig:
    """A fast full-multimodal model config for tests."""
    return ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [4, 2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={
            "d_model": 64,
            "depth": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "embedding_dim": 64,
            "max_len": 8,
        },
        cross_attention={"num_heads": 4, "out_dim": 64},
        gated_fusion={"out_dim": 64, "hidden_dim": 64},
        shared_encoder={
            "d_model": 64,
            "depth": 2,
            "num_heads": 4,
            "ff_dim": 256,
            "out_dim": 128,
        },
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


@pytest.fixture(scope="session")
def config() -> ModelConfig:
    return small_config()


@pytest.fixture(scope="session")
def model(config: ModelConfig):
    """A built multimodal model (shared across tests)."""
    return ModelFactory.create(config)


@pytest.fixture(scope="session")
def batch(model) -> dict[str, torch.Tensor]:
    """A valid Phase-4-style batch for the shared model."""
    return model.sample_batch(batch_size=4, seq_len=4)


@pytest.fixture
def tabular_only_config() -> ModelConfig:
    return ModelConfig(
        tabular={"numeric_dim": 5, "categorical_cardinalities": [3, 3]},
        image_encoder={"backbone": None},
        shared_encoder={"d_model": 64, "depth": 1, "num_heads": 4, "ff_dim": 256,
                        "out_dim": 96},
        heads={"crop": {"num_classes": 4}, "yield_prediction": {}},
    )


@pytest.fixture
def image_only_config() -> ModelConfig:
    return ModelConfig(
        tabular={"numeric_dim": 0, "categorical_cardinalities": []},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 64, "depth": 1, "num_heads": 4, "ff_dim": 256,
                  "embedding_dim": 64, "max_len": 8},
        shared_encoder={"d_model": 64, "depth": 1, "num_heads": 4, "ff_dim": 256,
                        "out_dim": 96},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


# --------------------------------------------------------------------------- #
# Real STAM -> Preprocessor chain (shared with test_config / test_integration)
# --------------------------------------------------------------------------- #


@pytest.fixture
def stam_chain(tmp_path: Path):
    """A real initialized STAM over the synthetic dataset (patch size 16)."""
    from training.dataset_manager import DatasetManager, Settings
    from training.stam import STAM, StamConfig
    from training.stam.tests.conftest import _build_synthetic_dataset

    catalog = _build_synthetic_dataset(tmp_path)
    dataset_root = catalog.parent.parent
    manager = DatasetManager(
        Settings(
            dataset_root=dataset_root,
            catalog_name="kaggle-crop-yield",
            logging={"console": False, "level": "ERROR"},
        )
    )
    manager.generate_metadata(force=True)
    stam = STAM(
        manager,
        StamConfig(
            patch={"size": 16},
            tabular={"table": "crop_yield.csv",
                     "village_column": "village",
                     "district_column": "district",
                     "year_column": "year",
                     "season_column": "season",
                     "crop_column": "crop",
                     "yield_column": "yield_kg"},
            admin={"boundaries": ["raw/kaggle-crop-yield/boundaries.geojson"],
                   "name_column": "name", "level_column": "level"},
            image={"resolution": "R10m", "require_pairs": True},
        ),
    )
    stam.initialize()
    return stam


@pytest.fixture
def stam_observations(stam_chain):
    """A small observation set across points and years (2020 + 2021 Kharif)."""
    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_chain.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_chain.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    return obs


def _build_preprocessor(stam_chain, observations, *, ordinal: bool) -> Any:
    from training.preprocessing import PreprocessingConfig, Preprocessor

    config = PreprocessingConfig(
        image={"size": 16, "normalize": "minmax"},
        temporal={"max_observations": 8, "min_observations": 1},
        tabular={
            "scaler": "standard",
            "categorical_encoding": "ordinal" if ordinal else "onehot",
            "numeric_features": ["rainfall_mm"],
            "categorical_features": ["village", "district"],
            "exclude_columns": ["crop", "yield_kg", "year", "season"],
        },
        quality={"min_quality_score": 0.0},
    )
    preprocessor = Preprocessor(config)
    accepted, _ = preprocessor.filter(observations)
    preprocessor.fit(accepted, extractor=stam_chain.get_patch)
    return preprocessor


@pytest.fixture
def preprocessor_ordinal(stam_chain, stam_observations):
    return _build_preprocessor(stam_chain, stam_observations, ordinal=True)


@pytest.fixture
def preprocessor_onehot(stam_chain, stam_observations):
    return _build_preprocessor(stam_chain, stam_observations, ordinal=False)
