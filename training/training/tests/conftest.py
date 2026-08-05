"""Shared fixtures for the training test-suite.

Fast tests use a tiny tabular-only model so no timm backbone is instantiated.
Integration tests (experiment / ablation / cross-validation) build a real STAM
chain over the synthetic dataset (patch size 16) and a fitted preprocessor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.models import ModelConfig, ModelFactory
from training.training.config import TrainingConfig


def small_tabular_config() -> ModelConfig:
    """A fast tabular-only model config."""
    return ModelConfig(
        tabular={"numeric_dim": 4, "categorical_cardinalities": [3]},
        image_encoder={"backbone": None},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


def small_full_config() -> ModelConfig:
    """A fast full multimodal model config (tiny timm backbone)."""
    return ModelConfig(
        tabular={"numeric_dim": 3, "categorical_cardinalities": [2]},
        image_encoder={"backbone": "mobilenetv3_small_050", "input_size": 32},
        temporal={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                  "embedding_dim": 32, "max_len": 6},
        cross_attention={"num_heads": 4, "out_dim": 32},
        gated_fusion={"out_dim": 32, "hidden_dim": 32},
        shared_encoder={"d_model": 32, "depth": 1, "num_heads": 4, "ff_dim": 128,
                        "out_dim": 48},
        heads={"crop": {"num_classes": 3}, "yield_prediction": {}},
    )


@pytest.fixture
def tabular_config() -> ModelConfig:
    return small_tabular_config()


@pytest.fixture
def full_config() -> ModelConfig:
    return small_full_config()


@pytest.fixture
def tabular_model(tabular_config):
    return ModelFactory.create(tabular_config)


def make_fake_loader(n: int = 16, batch_size: int = 8, feature_dim: int = 5,
                     num_classes: int = 3):
    """A loader yielding Phase-4-style batches (tabular-only)."""
    import random

    class _FakeLoader:
        def __init__(self) -> None:
            self.batches = []
            for _ in range(n // batch_size):
                self.batches.append(
                    {
                        "tabular": torch.randn(batch_size, feature_dim),
                        "crop_label": torch.randint(0, num_classes, (batch_size,)),
                        "yield_label": torch.randn(batch_size, 1),
                    }
                )
            self.n = n

        def __len__(self) -> int:
            return len(self.batches)

        def __iter__(self):
            return iter(self.batches)

    return _FakeLoader()


@pytest.fixture
def fake_loader():
    return make_fake_loader()


@pytest.fixture
def train_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        name="test",
        general={"device": "cpu", "seed": 42},
        train={"epochs": 2, "early_stopping_patience": 3},
        checkpoint={"directory": str(tmp_path / "ckpt")},
        logging={"console": False},
    )


# --------------------------------------------------------------------------- #
# Real STAM -> Preprocessor chain (integration tests)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def stam_chain(tmp_path_factory):
    """A real initialized STAM over the synthetic dataset (patch size 16)."""
    from training.dataset_manager import DatasetManager, Settings
    from training.stam import STAM, StamConfig
    from training.stam.tests.conftest import _build_synthetic_dataset

    tmp_path = tmp_path_factory.mktemp("stam")
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


@pytest.fixture(scope="session")
def stam_observations(stam_chain):
    obs = []
    for lon, lat in [(74.801, 13.099), (74.802, 13.098), (74.803, 13.097)]:
        obs.append(stam_chain.build_observation(lon, lat, year=2020, season="Kharif"))
    obs.append(stam_chain.build_observation(74.802, 13.098, year=2021, season="Kharif"))
    return obs


@pytest.fixture(scope="session")
def preprocessor(stam_chain, stam_observations):
    """A fitted preprocessor over the synthetic observations (random split)."""
    from training.preprocessing import PreprocessingConfig, Preprocessor

    config = PreprocessingConfig(
        image={"size": 16, "normalize": "minmax"},
        temporal={"max_observations": 8, "min_observations": 1},
        split={"strategy": "random", "train_ratio": 0.5, "val_ratio": 0.25,
               "test_ratio": 0.25},
        tabular={
            "scaler": "standard",
            "categorical_encoding": "ordinal",
            "numeric_features": ["rainfall_mm"],
            "categorical_features": ["village", "district"],
            "exclude_columns": ["crop", "yield_kg", "year", "season"],
        },
        quality={"min_quality_score": 0.0},
    )
    preprocessor = Preprocessor(config)
    accepted, _ = preprocessor.filter(stam_observations)
    preprocessor.fit(accepted, extractor=stam_chain.get_patch)
    return preprocessor


@pytest.fixture
def derived_model_config(preprocessor):
    from training.models import ModelFactory

    return ModelFactory.build_config(
        preprocessor,
        temporal={"d_model": 64, "depth": 1, "num_heads": 4, "ff_dim": 256,
                  "embedding_dim": 64, "max_len": 8},
        shared_encoder={"d_model": 64, "depth": 1, "num_heads": 4, "ff_dim": 256,
                        "out_dim": 96},
    )


@pytest.fixture
def integration_training_config(tmp_path):
    return TrainingConfig(
        name="integration",
        general={"device": "cpu", "seed": 42,
                 "output_dir": str(tmp_path / "out")},
        train={"epochs": 1, "early_stopping_patience": 3},
        checkpoint={"directory": str(tmp_path / "ckpt"), "save_best": True},
        logging={"console": False},
        visualization={"enabled": True, "directory": str(tmp_path / "viz")},
        benchmark={"enabled": True, "iterations": 2, "warmup_iterations": 1,
                   "batch_size": 2},
    )
