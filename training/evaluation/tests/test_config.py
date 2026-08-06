"""Evaluation configuration tests (Phase R5)."""

from __future__ import annotations

import pytest

from training.evaluation.config import (
    EvaluationConfig,
    load_evaluation_config,
    save_evaluation_template,
)
from training.evaluation.exceptions import EvaluationConfigurationError


def test_defaults():
    config = EvaluationConfig()
    assert config.name == "cropfusion_evaluation"
    assert config.general.device == "auto"
    assert config.metrics.top_k == 3
    assert config.ablation.compare_metric == "crop/f1"
    assert config.error_analysis.failure_relative_error == 0.3


def test_env_override():
    config = load_evaluation_config(env={"EVAL_NAME": "r5", "EVAL_METRICS__TOP_K": "5"})
    assert config.name == "r5"
    assert config.metrics.top_k == 5


def test_yaml_roundtrip(tmp_path):
    path = tmp_path / "eval.yaml"
    save_evaluation_template(path)
    config = load_evaluation_config(path)
    assert config.name == "cropfusion_evaluation"


def test_missing_file_raises():
    with pytest.raises(EvaluationConfigurationError):
        load_evaluation_config("does_not_exist.yaml")


def test_invalid_value_raises():
    with pytest.raises(EvaluationConfigurationError):
        load_evaluation_config(env={"EVAL_METRICS__AVERAGE": "bogus"})
