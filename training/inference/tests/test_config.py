"""Inference configuration tests (Phase R5)."""

from __future__ import annotations

import pytest

from training.inference.config import (
    InferenceConfig,
    load_inference_config,
    save_inference_template,
)
from training.inference.exceptions import InferenceConfigurationError


def test_defaults():
    config = InferenceConfig()
    assert config.name == "cropfusion_inference"
    assert config.general.package_name == "cropfusion"
    assert config.general.version == "1.0.0"
    assert config.exporter.formats == ["pytorch", "torchscript", "onnx"]
    assert config.exporter.onnx_opset == 17
    assert config.validation.strict is True


def test_env_override():
    config = load_inference_config(
        env={"INF_NAME": "r5", "INF_EXPORTER__FORMATS": '["pytorch", "onnx"]'}
    )
    assert config.name == "r5"
    assert config.exporter.formats == ["pytorch", "onnx"]


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "inference.yaml"
    save_inference_template(path)
    config = load_inference_config(path)
    assert config.name == "cropfusion_inference"


def test_missing_file_raises():
    with pytest.raises(InferenceConfigurationError):
        load_inference_config("does_not_exist.yaml")


def test_invalid_raises():
    with pytest.raises(InferenceConfigurationError):
        load_inference_config(env={"INF_EXPORTER__ONNX_OPSET": "5"})
