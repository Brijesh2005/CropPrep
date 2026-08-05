"""Tests for the shared configuration loader."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from shared.config import (
    apply_case_insensitive,
    deep_merge,
    load_yaml_config,
    normalise_key,
    parse_env,
)
from shared.exceptions import ConfigurationError


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_root: str = "datasets"
    scan: dict = {}


def test_deep_merge_recursive() -> None:
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = deep_merge(base, {"a": {"y": 9, "z": 4}, "b": 8})
    assert merged == {"a": {"x": 1, "y": 9, "z": 4}, "b": 8}
    # base is not mutated
    assert base["a"]["y"] == 2


def test_normalise_key() -> None:
    assert normalise_key("DataRoot") == "dataroot"
    assert normalise_key("download-root") == "download_root"


def test_parse_env_nested_and_typed() -> None:
    env = {
        "DM_SCAN__WORKERS": "16",
        "DM_SCAN__HASH_FILES": "true",
        "DM_DATASET_ROOT": "/data",
        "DM_LOG__LEVEL": "DEBUG",
        "DM_EXPECTED_YEARS": "[2018, 2025]",
    }
    parsed = parse_env(env, prefix="DM_")
    assert parsed["dataset_root"] == "/data"
    assert parsed["scan"]["workers"] == 16
    assert parsed["scan"]["hash_files"] is True
    assert parsed["log"]["level"] == "DEBUG"
    assert parsed["expected_years"] == [2018, 2025]


def test_parse_env_ignores_other_prefixes() -> None:
    parsed = parse_env({"OTHER_X": "1", "DM_OK": "2"}, prefix="DM_")
    assert parsed == {"ok": 2}


def test_apply_case_insensitive() -> None:
    assert apply_case_insensitive({"Dataset_Root": "/x"}, _Settings) == {
        "dataset_root": "/x"
    }


def test_load_yaml_config_merges_env(tmp_path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("dataset_root: from-yaml\nscan:\n  workers: 4\n", encoding="utf-8")
    env = {"DM_SCAN__WORKERS": "16"}
    merged = load_yaml_config(config, env=env, prefix="DM_")
    assert merged["dataset_root"] == "from-yaml"
    assert merged["scan"]["workers"] == 16


def test_load_yaml_config_missing_file_raises(tmp_path) -> None:
    with pytest_raises(ConfigurationError):
        load_yaml_config(tmp_path / "nope.yaml", env={}, prefix="DM_")


def pytest_raises(exc_type):
    """Small helper so this module does not depend on pytest internals."""
    import pytest

    return pytest.raises(exc_type)
