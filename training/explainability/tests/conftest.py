"""Shared fixtures for the explainability test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.explainability.config import ExplainabilityConfig
from training.models import ModelConfig, ModelFactory


def _tabular_config() -> ModelConfig:
    return ModelConfig(
        tabular={"numeric_dim": 4, "categorical_cardinalities": [2]},
        image_encoder={"backbone": None},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


def _full_config() -> ModelConfig:
    return ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                  "embedding_dim": 32, "max_len": 6},
        cross_attention={"num_heads": 2, "out_dim": 32},
        gated_fusion={"out_dim": 32, "hidden_dim": 32},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                        "out_dim": 48},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


def _make_sample(seed: int, feature_dim: int = 5, seq_len: int = 4) -> dict:
    torch.manual_seed(seed)
    return {
        "tabular": torch.randn(feature_dim),
        "crop_label": torch.tensor(seed % 3),
        "yield_label": torch.randn(1),
        "ndvi": torch.randn(seq_len, 1, 32, 32) * 0.1,
        "evi": torch.randn(seq_len, 1, 32, 32) * 0.1,
        "temporal_mask": torch.cat(
            [torch.ones(seq_len - 1), torch.zeros(1)]
        ),
    }


@pytest.fixture
def explainability_config() -> ExplainabilityConfig:
    return ExplainabilityConfig(
        shap={"background_size": 4, "max_samples": 32},
        uncertainty={"mc_dropout_samples": 0},
    )


@pytest.fixture
def tabular_model():
    model = ModelFactory.create(_tabular_config())
    model.eval()  # deterministic (dropout off) for explanation tests
    return model


@pytest.fixture
def full_model():
    model = ModelFactory.create(_full_config())
    model.eval()
    return model


@pytest.fixture
def samples():
    """Samples whose tabular width (5) matches the tabular-only model."""
    return [_make_sample(i, feature_dim=5) for i in range(8)]


@pytest.fixture
def sample(samples):
    return samples[0]


@pytest.fixture
def full_samples():
    """Samples whose tabular width (4) matches the full multimodal model."""
    return [_make_sample(i, feature_dim=4) for i in range(8)]


@pytest.fixture
def full_sample(full_samples):
    return full_samples[0]


# --------------------------------------------------------------------------- #
# Real STAM -> Preprocessor chain (facade integration tests)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def stam_chain(tmp_path_factory):
    from training.dataset_manager import DatasetManager, Settings
    from training.stam import STAM, StamConfig
    from training.stam.tests.conftest import _build_synthetic_dataset

    tmp_path = tmp_path_factory.mktemp("mxai")
    catalog = _build_synthetic_dataset(tmp_path)
    dataset_root = catalog.parent.parent
    manager = DatasetManager(
        Settings(dataset_root=dataset_root, catalog_name="kaggle-crop-yield",
                 logging={"console": False, "level": "ERROR"})
    )
    manager.generate_metadata(force=True)
    stam = STAM(
        manager,
        StamConfig(
            patch={"size": 16},
            tabular={"table": "crop_yield.csv", "village_column": "village",
                     "district_column": "district", "year_column": "year",
                     "season_column": "season", "crop_column": "crop",
                     "yield_column": "yield_kg"},
            admin={"boundaries": ["raw/kaggle-crop-yield/boundaries.geojson"],
                   "name_column": "name", "level_column": "level"},
            image={"resolution": "R10m", "require_pairs": True},
        ),
    )
    stam.initialize()
    return stam


@pytest.fixture(scope="session")
def observations(stam_chain):
    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_chain.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_chain.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    return obs


@pytest.fixture(scope="session")
def preprocessor(stam_chain, observations):
    from training.preprocessing import PreprocessingConfig, Preprocessor

    config = PreprocessingConfig(
        image={"size": 16, "normalize": "minmax"},
        temporal={"max_observations": 8, "min_observations": 1},
        tabular={"scaler": "standard", "categorical_encoding": "ordinal",
                 "numeric_features": ["rainfall_mm"],
                 "categorical_features": ["village", "district"],
                 "exclude_columns": ["crop", "yield_kg", "year", "season"]},
        quality={"min_quality_score": 0.0},
    )
    pre = Preprocessor(config)
    accepted, _ = pre.filter(observations)
    pre.fit(accepted, extractor=stam_chain.get_patch)
    return pre


@pytest.fixture(scope="session")
def derived_model_config(preprocessor):
    return ModelFactory.build_config(
        preprocessor,
        temporal={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                  "embedding_dim": 32, "max_len": 8},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 2, "ff_dim": 128,
                        "out_dim": 48},
    )
