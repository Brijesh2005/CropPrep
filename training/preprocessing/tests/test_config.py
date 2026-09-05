"""Unit tests for preprocessing configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from training.preprocessing.config import (
    PreprocessingConfig,
    load_preprocessing_config,
    save_preprocessing_template,
)
from training.preprocessing.exceptions import ConfigurationError


def test_defaults():
    config = load_preprocessing_config(env={})
    assert config.image.size == 128
    assert config.tabular.scaler == "standard"
    assert config.temporal.max_observations == 8
    assert config.split.strategy == "temporal"
    assert config.augmentation.enabled is False
    assert config.dataloader.batch_size == 32


def test_env_overrides():
    config = load_preprocessing_config(
        env={
            "PRE_IMAGE__SIZE": "224",
            "PRE_TABULAR__SCALER": "minmax",
            "PRE_SPLIT__STRATEGY": "stratified",
            "PRE_DATALOADER__BATCH_SIZE": "64",
            "PRE_AUGMENTATION__ENABLED": "true",
        }
    )
    assert config.image.size == 224
    assert config.tabular.scaler == "minmax"
    assert config.split.strategy == "stratified"
    assert config.dataloader.batch_size == 64
    assert config.augmentation.enabled is True


def test_template_roundtrip(tmp_path: Path):
    out = tmp_path / "pre.yaml"
    save_preprocessing_template(out)
    config = load_preprocessing_config(out, env={})
    assert config.image.size == 128
    assert config.split.strategy == "temporal"


def test_missing_config_raises():
    with pytest.raises(ConfigurationError):
        load_preprocessing_config("missing.yaml", env={})


def test_unknown_key_raises(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_preprocessing_config(config, env={})


def test_invalid_strategy_raises(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("split:\n  strategy: banana\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_preprocessing_config(config, env={})


def test_quality_config_defaults():
    config = PreprocessingConfig()
    assert config.quality.min_quality_score == 40.0
    assert config.quality.require_valid_coordinates is True


def test_repo_preprocessing_config():
    # The repository's accuracy-improved preprocessing.yaml must stay loadable
    # with the settings the training pipeline relies on (224px imagery,
    # ordinal categorical encoding, augmentation enabled).
    config_path = (
        Path(__file__).resolve().parents[3] / "training" / "config" / "preprocessing.yaml"
    )
    config = load_preprocessing_config(config_path, env={})
    assert config.image.size == 224
    assert config.tabular.categorical_encoding == "ordinal"
    # R5.2.9: the repository config now declares the full frozen-v1 base fields
    # PLUS the DK-grid environmental features (28 numeric + 4 categorical).
    expected_numeric = [
        "lat", "lon", "spatial_match_distance_km", "year", "annual_rainfall_mm",
        "dewpoint_c", "elevation", "temperature_c", "relative_humidity_pct",
        "slope", "ndvi", "evi", "ndwi", "ndre", "savi", "s2_obs_count",
        "soil_clay_pct", "soil_sand_pct", "soil_organic_carbon", "soil_ph",
        "soil_moisture", "kharif_ndvi", "kharif_evi", "kharif_ndwi",
        "rabi_ndvi", "rabi_evi", "rabi_ndwi", "env_match_distance_m",
    ]
    assert config.tabular.numeric_features == expected_numeric
    assert config.tabular.categorical_features == [
        "season", "is_cropland", "land_cover_class", "soil_type_class",
    ]
    assert config.augmentation.enabled is True
