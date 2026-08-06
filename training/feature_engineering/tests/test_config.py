"""Tests for the feature-engineering configuration loader."""

from __future__ import annotations

import yaml

from training.feature_engineering.config import (
    FeatureEngineeringConfig,
    load_feature_engineering_config,
    save_feature_engineering_template,
)
from training.feature_engineering.exceptions import FeatureConfigError


def test_defaults():
    config = FeatureEngineeringConfig()
    assert config.tabular.enabled is True
    assert config.image.extract_patch_stats is False
    assert config.temporal.enabled is True
    assert config.prefixes is True


def test_load_from_yaml(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(
        yaml.safe_dump(
            {"image": {"extract_patch_stats": True, "max_dates": 4}, "prefixes": False}
        ),
        encoding="utf-8",
    )
    config = load_feature_engineering_config(path)
    assert config.image.extract_patch_stats is True
    assert config.image.max_dates == 4
    assert config.prefixes is False


def test_load_from_env_overrides(tmp_path):
    path = tmp_path / "features.yaml"
    path.write_text(yaml.safe_dump({"image": {"max_dates": 2}}), encoding="utf-8")
    config = load_feature_engineering_config(
        path, env={"FE_IMAGE__MAX_DATES": "6", "FE_PREFIXES": "false"}
    )
    assert config.image.max_dates == 6
    assert config.prefixes is False


def test_save_template_roundtrip(tmp_path):
    path = save_feature_engineering_template(tmp_path / "template.yaml")
    config = load_feature_engineering_config(path)
    assert config.tabular.enabled is True


def test_missing_file_raises(tmp_path):
    try:
        load_feature_engineering_config(tmp_path / "missing.yaml")
    except FeatureConfigError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("expected FeatureConfigError")


def test_invalid_value_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"image": {"max_dates": "nope"}}), encoding="utf-8")
    try:
        load_feature_engineering_config(path)
    except FeatureConfigError:
        pass
    else:
        raise AssertionError("expected FeatureConfigError")
