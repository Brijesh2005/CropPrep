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
