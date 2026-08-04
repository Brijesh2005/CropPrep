"""Uncertainty / confidence estimation tests."""

from __future__ import annotations

import numpy as np

from ai.explainability import UncertaintyEstimator
from ai.explainability.config import ExplainabilityConfig


def _batch(sample):
    return {k: v.unsqueeze(0) for k, v in sample.items() if hasattr(v, "unsqueeze") and v.dim() > 0}


def test_crop_confidence_and_entropy(tabular_model, sample):
    estimator = UncertaintyEstimator(tabular_model)
    logits = tabular_model(_batch(sample)).crop_logits
    conf = estimator.crop_confidence(logits)
    ent = estimator.entropy(logits)
    assert 0.0 <= conf <= 1.0
    assert ent >= 0.0


def test_mc_dropout(tabular_model, sample):
    config = ExplainabilityConfig(uncertainty={"mc_dropout_samples": 5})
    estimator = UncertaintyEstimator(tabular_model, config.uncertainty)
    result = estimator.mc_dropout(_batch(sample), samples=5)
    assert "crop_conf" in result
    assert "yield_std" in result
    assert result["yield_std"] >= 0.0


def test_calibration(tabular_model):
    estimator = UncertaintyEstimator(tabular_model)
    confidences = np.asarray([0.9, 0.9, 0.5, 0.5, 0.2, 0.2])
    correct = np.asarray([1, 0, 1, 0, 1, 0])
    calibration = estimator.calibration(confidences, correct, bins=5)
    assert 0.0 <= calibration["ece"] <= 1.0
    assert len(calibration["bins"]) > 0


def test_confidence_distribution(tabular_model):
    estimator = UncertaintyEstimator(tabular_model)
    dist = estimator.confidence_distribution(np.asarray([0.5, 0.7, 0.9]), bins=5)
    assert sum(dist["counts"]) == 3
