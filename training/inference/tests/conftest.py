"""Shared fixtures for the inference test-suite (Phase R5)."""

from __future__ import annotations

import pandas as pd
import pytest
from types import SimpleNamespace

from training.models import ModelConfig, ModelFactory
from training.preprocessing.label_pipeline import LabelPipeline
from training.preprocessing.tabular_pipeline import TabularPipeline
from training.preprocessing.transforms import LabelEncoder


def small_config() -> ModelConfig:
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


def fake_preprocessor():
    """A Preprocessor-shaped object exposing ``tabular`` / ``label``."""
    tabular = TabularPipeline()
    tabular.numeric_features = ["rainfall_mm", "temperature", "soil_moisture"]
    tabular.categorical_features = ["soil_type"]
    tabular.feature_names = [
        "rainfall_mm", "temperature", "soil_moisture", "soil_type"
    ]
    tabular.fitted = True
    tabular._categorical_columns = ["soil_type"]

    label = LabelPipeline()
    label.crop_encoder = LabelEncoder().fit(["maize", "wheat", "rice"])
    label.fitted = True
    return SimpleNamespace(tabular=tabular, label=label)


@pytest.fixture(scope="module")
def model() -> ModelFactory:
    model = ModelFactory.create(small_config())
    model.eval()
    return model


@pytest.fixture
def preprocessor():
    return fake_preprocessor()


def make_dataset_sources(tmp_path, metadata_bytes: bytes = b"SQLITE"):
    """Write fake staged dataset artefacts and return a ``DatasetSources``."""
    from training.inference import DatasetSources

    src = tmp_path / "sources"
    src.mkdir(exist_ok=True)
    (src / "metadata.db").write_bytes(metadata_bytes)
    pd.DataFrame({"season": ["Kharif"]}).to_parquet(
        src / "historical_context.parquet"
    )
    pd.DataFrame({"lon": [74.8], "lat": [13.0]}).to_parquet(
        src / "location_index.parquet"
    )
    return DatasetSources(
        metadata_db=src / "metadata.db",
        historical_context=src / "historical_context.parquet",
        location_index=src / "location_index.parquet",
    )
