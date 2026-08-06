"""Config tests for the release runtime (Phase R6)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from training.runtime import (
    RuntimeConfig,
    load_runtime_config,
    save_runtime_template,
)
from training.runtime.exceptions import RuntimeConfigurationError


def test_defaults():
    config = RuntimeConfig()
    assert config.name == "cropfusion_runtime"
    assert config.general.releases_root == "releases"
    assert config.model.backend == "auto"
    assert config.model.warmup_steps == 1
    assert config.hot_reload.enabled is False
    assert config.hot_reload.poll_interval_seconds == 30.0
    assert config.validation.strict is True


def test_extra_forbid():
    with pytest.raises(ValidationError):
        RuntimeConfig(general={"unknown_key": 1})


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_MODEL__BACKEND", "onnx")
    monkeypatch.setenv("RT_MODEL__WARMUP_STEPS", "3")
    monkeypatch.setenv("RT_HOT_RELOAD__ENABLED", "true")
    config = load_runtime_config()
    assert config.model.backend == "onnx"
    assert config.model.warmup_steps == 3
    assert config.hot_reload.enabled is True


def test_env_json_list(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_MODEL__ONNX_PROVIDERS", '["CPUExecutionProvider"]')
    config = load_runtime_config()
    assert config.model.onnx_providers == ["CPUExecutionProvider"]


def test_yaml_file(tmp_path):
    path = tmp_path / "runtime.yaml"
    save_runtime_template(path)
    config = load_runtime_config(path)
    assert config.model.backend == "auto"
    assert config.general.releases_root == "releases"


def test_yaml_beats_defaults(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("model:\n  backend: torchscript\n", encoding="utf-8")
    config = load_runtime_config(path)
    assert config.model.backend == "torchscript"


def test_missing_config_file(tmp_path):
    with pytest.raises(RuntimeConfigurationError):
        load_runtime_config(tmp_path / "does_not_exist.yaml")


def test_malformed_yaml(tmp_path):
    path = tmp_path / "runtime.yaml"
    path.write_text("model: [unclosed\n", encoding="utf-8")
    with pytest.raises(RuntimeConfigurationError):
        load_runtime_config(path)


def test_template_roundtrip(tmp_path):
    path = tmp_path / "runtime.yaml"
    saved = save_runtime_template(path)
    assert saved.exists()
    assert load_runtime_config(saved).to_yaml()


def test_to_yaml_contains_sections():
    yaml_text = RuntimeConfig().to_yaml()
    for section in ("general", "model", "preprocess", "metadata", "cache",
                    "memory", "health", "hot_reload", "validation"):
        assert section in yaml_text
