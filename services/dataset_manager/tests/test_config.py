"""Unit tests for configuration loading, merging and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.dataset_manager.config import (
    DEFAULT_KAGGLE_HANDLE,
    Settings,
    load_settings,
    save_settings_template,
)
from services.dataset_manager.exceptions import InvalidConfigurationError


def test_defaults():
    settings = load_settings(env={})
    assert settings.download.kaggle_handle == DEFAULT_KAGGLE_HANDLE
    assert settings.scan.workers >= 1
    assert settings.validation.expected_years == (2018, 2025)
    assert settings.metadata.store_type == "sqlite"
    assert settings.cache.default_ttl_seconds == 86400


def test_derived_paths(tmp_path: Path):
    settings = Settings(dataset_root=tmp_path / "d")
    assert settings.raw_root == tmp_path / "d" / "raw"
    assert settings.catalog_root == tmp_path / "d" / "raw" / "kaggle-crop-yield"
    assert settings.state_root == tmp_path / "d" / ".cropfusion"
    assert settings.metadata_db_path() == settings.state_root / "metadata.db"


def test_env_overrides():
    env = {
        "DM_DATASET_ROOT": "/data/datasets",
        "DM_DOWNLOAD__KAGGLE_HANDLE": "owner/other",
        "DM_SCAN__WORKERS": "16",
        "DM_CACHE__ENABLED": "false",
        "DM_LOGGING__LEVEL": "DEBUG",
    }
    settings = load_settings(env=env)
    assert settings.dataset_root == Path("/data/datasets")
    assert settings.download.kaggle_handle == "owner/other"
    assert settings.scan.workers == 16
    assert settings.cache.enabled is False
    assert settings.logging.level == "DEBUG"


def test_yaml_overrides_then_env_wins(tmp_path: Path):
    config = tmp_path / "dm.yaml"
    config.write_text(
        "dataset_root: /from/yaml\ndownload:\n  kaggle_handle: a/b\n",
        encoding="utf-8",
    )
    settings = load_settings(config, env={"DM_DOWNLOAD__KAGGLE_HANDLE": "c/d"})
    assert settings.dataset_root == Path("/from/yaml")
    assert settings.download.kaggle_handle == "c/d"  # env wins over yaml


def test_missing_config_file_raises():
    with pytest.raises(InvalidConfigurationError):
        load_settings("definitely/missing.yaml", env={})


def test_malformed_yaml_raises(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("download: [unclosed", encoding="utf-8")
    with pytest.raises(InvalidConfigurationError):
        load_settings(config, env={})


def test_unknown_keys_rejected(tmp_path: Path):
    config = tmp_path / "unknown.yaml"
    config.write_text("nonsense_field: 1\n", encoding="utf-8")
    with pytest.raises(InvalidConfigurationError):
        load_settings(config, env={})


def test_invalid_values_rejected(tmp_path: Path):
    config = tmp_path / "invalid.yaml"
    config.write_text("scan:\n  workers: -5\n", encoding="utf-8")
    with pytest.raises(InvalidConfigurationError):
        load_settings(config, env={})


def test_config_template_roundtrip(tmp_path: Path):
    out = tmp_path / "template.yaml"
    save_settings_template(out)
    assert out.exists()
    settings = load_settings(out, env={})
    assert settings.download.kaggle_handle == DEFAULT_KAGGLE_HANDLE
