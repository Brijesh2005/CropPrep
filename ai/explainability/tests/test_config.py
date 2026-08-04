"""Configuration tests: validation, YAML round-trip, env overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.explainability import (
    ExplainabilityConfig,
    load_explainability_config,
    save_explainability_template,
)
from ai.explainability.exceptions import ExplainabilityConfigurationError


def test_defaults_valid():
    cfg = ExplainabilityConfig()
    assert cfg.shap.method == "kernel"
    assert cfg.cam.method == "gradcam++"
    assert cfg.uncertainty.bins == 10


def test_rejects_bad_cam_method():
    with pytest.raises(Exception):
        ExplainabilityConfig(cam={"method": "magic"})


def test_yaml_round_trip(tmp_path: Path):
    path = tmp_path / "mxai.yaml"
    cfg = ExplainabilityConfig(name="rt", shap={"max_samples": 128})
    cfg.save(path)
    loaded = load_explainability_config(path)
    assert loaded.name == "rt"
    assert loaded.shap.max_samples == 128


def test_template_written(tmp_path: Path):
    path = save_explainability_template(tmp_path / "template.yaml")
    assert path.exists()
    loaded = load_explainability_config(path)
    assert loaded.shap.background_size == 50


def test_env_overrides(tmp_path: Path):
    path = save_explainability_template(tmp_path / "template.yaml")
    cfg = load_explainability_config(
        env={
            "MXAI_CONFIG_FILE": str(path),
            "MXAI_CAM__METHOD": "layercam",
            "MXAI_UNCERTAINTY__MC_DROPOUT_SAMPLES": "20",
        }
    )
    assert cfg.cam.method == "layercam"
    assert cfg.uncertainty.mc_dropout_samples == 20


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(ExplainabilityConfigurationError):
        load_explainability_config(tmp_path / "nope.yaml")
