"""Counterfactual and integrated-gradients tests."""

from __future__ import annotations

import numpy as np

from training.explainability import CounterfactualEngine
from training.explainability.integrated_gradients import (
    SharedEmbeddingIntegratedGradients,
    TabularIntegratedGradients,
)
from training.explainability.config import ExplainabilityConfig


def test_tabular_perturbation(tabular_model, samples):
    engine = CounterfactualEngine(
        tabular_model, ExplainabilityConfig().counterfactual,
        feature_names=[f"f{i}" for i in range(5)],
    )
    result = engine.explain(
        samples[0],
        perturbations=[
            {"feature": "f0", "delta": 0.5, "mode": "add", "label": "f0 +0.5"},
            {"feature": 1, "delta": 0.9, "mode": "multiply", "label": "f1 *0.9"},
            {"feature": 2, "delta": 1.0, "mode": "set", "label": "f2 = 1.0"},
        ],
    )
    assert result["original"]["crop_class"] is not None
    assert len(result["counterfactuals"]) == 3
    for cf in result["counterfactuals"]:
        assert "crop_changed" in cf
        assert "yield_delta" in cf


def test_image_and_mask_perturbations(full_model, full_sample):
    engine = CounterfactualEngine(full_model, ExplainabilityConfig().counterfactual)
    result = engine.explain(
        full_sample,
        perturbations=[
            {"image": "ndvi", "factor": 0.7, "label": "NDVI -30%"},
            {"mask": 0, "label": "drop obs 0"},
        ],
    )
    assert len(result["counterfactuals"]) == 2


def test_unknown_feature_raises(tabular_model, samples):
    engine = CounterfactualEngine(
        tabular_model, ExplainabilityConfig().counterfactual, feature_names=["a", "b"]
    )
    result = engine.explain(samples[0], perturbations=[{"feature": "zzz", "delta": 1.0}])
    assert result["counterfactuals"][0].get("error") is not None


def test_tabular_integrated_gradients(tabular_model, sample):
    config = ExplainabilityConfig(integrated_gradients={"steps": 10})
    ig = TabularIntegratedGradients(tabular_model, config.integrated_gradients)
    attributions = ig.attribute(sample, kind="crop")
    assert np.asarray(attributions).shape == sample["tabular"].shape
    assert np.isfinite(attributions).all()


def test_shared_embedding_integrated_gradients(tabular_model, sample):
    config = ExplainabilityConfig(integrated_gradients={"steps": 10})
    ig = SharedEmbeddingIntegratedGradients(tabular_model, config.integrated_gradients)
    attributions = ig.attribute(sample, kind="crop")
    assert attributions.ndim == 1
    assert np.isfinite(attributions).all()
