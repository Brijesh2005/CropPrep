"""Metric tests: known-value checks for classification + regression."""

from __future__ import annotations

import torch

from training.training import (
    MetricsTracker,
    compute_classification_metrics,
    compute_regression_metrics,
)
from training.training.config import MetricsConfig


def test_regression_known_values():
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0]])
    result = compute_regression_metrics(preds, targets)
    assert abs(result["mse"] - 0.0) < 1e-6
    assert abs(result["rmse"] - 0.0) < 1e-6
    assert abs(result["r2"] - 1.0) < 1e-6
    assert abs(result["mape"] - 0.0) < 1e-6


def test_regression_mape_guards_zero_targets():
    preds = torch.tensor([[0.5], [1.0]])
    targets = torch.tensor([[0.0], [2.0]])
    result = compute_regression_metrics(preds, targets)
    assert torch.isfinite(torch.tensor(result["mape"]))


def test_classification_accuracy():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 0])
    result = compute_classification_metrics(logits, targets, MetricsConfig(top_k=2))
    assert result["accuracy"] == 1.0
    assert result["f1"] == 1.0
    assert result["confusion_matrix"] is not None


def test_top_k_accuracy():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0], [1.0, 2.0, 0.0]])
    targets = torch.tensor([0, 2, 1])  # each true class is the top-2 hit
    result = compute_classification_metrics(logits, targets, MetricsConfig(top_k=2))
    assert result["top2_accuracy"] == 1.0  # correct class is in top-2 everywhere


def test_metrics_tracker():
    tracker = MetricsTracker({"crop": "classification", "yield": "regression"})
    tracker.update("crop", torch.tensor([[2.0, 0.0]]), torch.tensor([0]))
    tracker.update("yield", torch.tensor([[1.0]]), torch.tensor([[1.0]]))
    result = tracker.result()
    assert "crop/accuracy" in result
    assert "yield/rmse" in result
