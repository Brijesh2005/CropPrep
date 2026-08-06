"""Evaluator tests (Phase R5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evaluation.config import EvaluationConfig
from training.evaluation.evaluator import MultimodalEvaluator


def test_evaluate_full_multimodal(full_model, fake_loader):
    config = EvaluationConfig(general={"device": "cpu", "collect_embeddings": True})
    outcome = MultimodalEvaluator(full_model, config).evaluate(fake_loader)

    assert outcome.num_samples == 16
    assert set(outcome.metrics) == {"crop", "yield"}
    assert "accuracy" in outcome.metrics["crop"]
    assert "rmse" in outcome.metrics["yield"]
    assert outcome.latency_ms["mean"] > 0
    assert outcome.embeddings is not None
    assert outcome.embeddings.shape == (16, 48)
    assert outcome.pr_curves["crop"]


def test_gates_captured(full_model, fake_loader):
    config = EvaluationConfig(general={"device": "cpu"})
    outcome = MultimodalEvaluator(full_model, config).evaluate(fake_loader)
    assert outcome.gates is not None
    assert set(outcome.gates) >= {"image_gate", "tabular_gate", "fusion_gate"}
    for values in outcome.gates.values():
        assert values.shape == (16,)
        assert np.all((values >= 0) & (values <= 1))


def test_predictions_contract(full_model, fake_loader):
    outcome = MultimodalEvaluator(full_model, EvaluationConfig()).evaluate(fake_loader)
    crop = outcome.predictions["crop"]
    assert crop["targets"].shape == (16,)
    assert crop["preds"].shape == (16,)
    assert crop["probs"].shape == (16, 3)
    yield_ = outcome.predictions["yield"]
    assert yield_["targets"].shape == (16,)
    assert yield_["preds"].shape == (16, 1)


def test_outcome_to_dict_serializable(full_model, fake_loader):
    outcome = MultimodalEvaluator(full_model, EvaluationConfig()).evaluate(fake_loader)
    payload = outcome.to_dict()
    assert payload["num_samples"] == 16
    assert "metrics" in payload
