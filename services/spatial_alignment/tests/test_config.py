"""Unit tests for STAM configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.spatial_alignment.config import (
    DEFAULT_SEASONS,
    StamConfig,
    load_stam_config,
    save_stam_config_template,
)
from services.spatial_alignment.exceptions import StamConfigurationError


def test_defaults():
    config = load_stam_config(env={})
    assert config.patch.size == 128
    assert config.spatial.max_search_radius_km == 5.0
    assert config.image.resolution == "R10m"
    assert config.image.index_types == ["NDVI", "EVI"]
    assert config.temporal.tolerance_days == 15
    assert [s.name for s in config.seasons] == ["Kharif", "Rabi", "Summer"]


def test_season_defaults():
    config = StamConfig()
    kharif = config.seasons[0]
    assert kharif.start_month == 6 and kharif.end_month == 10
    rabi = config.seasons[1]
    assert rabi.crosses_year_boundary is True


def test_env_overrides():
    config = load_stam_config(
        env={
            "ST_PATCH__SIZE": "224",
            "ST_SPATIAL__MAX_SEARCH_RADIUS_KM": "10.0",
            "ST_TEMPORAL__DEFAULT_SEASON": "Rabi",
            "ST_IMAGE__RESOLUTION": "R20m",
            "ST_CACHE__ENABLED": "false",
        }
    )
    assert config.patch.size == 224
    assert config.spatial.max_search_radius_km == 10.0
    assert config.temporal.default_season == "Rabi"
    assert config.image.resolution == "R20m"
    assert config.cache.enabled is False


def test_yaml_roundtrip(tmp_path: Path):
    out = tmp_path / "stam.yaml"
    save_stam_config_template(out)
    assert out.exists()
    config = load_stam_config(out, env={})
    assert config.patch.size == 128
    assert config.image.resolution == "R10m"


def test_missing_config_file_raises():
    with pytest.raises(StamConfigurationError):
        load_stam_config("missing.yaml", env={})


def test_unknown_keys_rejected(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("nonsense: 1\n", encoding="utf-8")
    with pytest.raises(StamConfigurationError):
        load_stam_config(config, env={})


def test_invalid_patch_size_rejected(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text("patch:\n  size: 2\n", encoding="utf-8")
    with pytest.raises(StamConfigurationError):
        load_stam_config(config, env={})


def test_custom_season_definition():
    config = StamConfig(
        seasons=[{"name": "Mono", "start_month": 5, "end_month": 9}]
    )
    assert config.seasons[0].name == "Mono"
    assert config.season_names() == ["Mono"]


def test_default_seasons_matches_constant():
    assert len(DEFAULT_SEASONS) == 3
