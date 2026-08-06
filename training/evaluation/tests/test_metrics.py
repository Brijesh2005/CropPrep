"""Metrics tests (Phase R5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from training.evaluation.config import MetricsConfig
from training.evaluation.exceptions import MetricComputationError
from training.evaluation.metrics import (
    EvaluationAccumulator,
    compute_classification_metrics,
    compute_pr_curves,
    compute_regression_metrics,
)


class TestClassificationMetrics:
    def test_perfect_classifier(self):
        logits = torch.tensor([[2.0, 0.0]] * 6 + [[0.0, 2.0]] * 6)
        targets = torch.tensor([0] * 6 + [1] * 6)
        metrics = compute_classification_metrics(logits, targets)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["balanced_accuracy"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)
        assert metrics["confusion_matrix"] == [[6, 0], [0, 6]]
        assert metrics["support"] == 12
        assert len(metrics["per_class"]) == 2

    def test_top_k_accuracy(self):
        logits = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [0.0, 1.0, 3.0]])
        targets = torch.tensor([1, 1, 1])
        metrics = compute_classification_metrics(logits, targets, MetricsConfig(top_k=3))
        assert metrics["top3_accuracy"] == pytest.approx(1.0)

    def test_auprc_and_roc_auc_present(self):
        rng = torch.Generator().manual_seed(0)
        logits = torch.randn(64, 3, generator=rng)
        targets = torch.randint(0, 3, (64,), generator=rng)
        metrics = compute_classification_metrics(logits, targets)
        assert metrics["roc_auc"] is not None
        assert metrics["auprc"] is not None


class TestRegressionMetrics:
    def test_perfect_regression(self):
        preds = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(preds, targets)
        assert metrics["rmse"] == pytest.approx(0.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["bias"] == pytest.approx(0.0)
        assert metrics["median_absolute_error"] == pytest.approx(0.0)
        assert metrics["within_tolerance"] == pytest.approx(1.0)
        assert len(metrics["error_histogram"]["counts"]) == 10

    def test_bias_signed(self):
        preds = torch.tensor([2.0, 3.0, 4.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(preds, targets)
        assert metrics["bias"] == pytest.approx(-1.0)

    def test_empty_raises(self):
        with pytest.raises(MetricComputationError):
            compute_regression_metrics(torch.empty(0), torch.empty(0))


class TestPrCurves:
    def test_curve_length(self):
        logits = torch.randn(32, 3)
        targets = torch.randint(0, 3, (32,))
        curves = compute_pr_curves(logits, targets)
        assert len(curves) == 3
        assert "precision" in curves[0] and "recall" in curves[0]


class TestEvaluationAccumulator:
    def test_classification_reduction(self):
        acc = EvaluationAccumulator(MetricsConfig())
        for _ in range(2):
            acc.update(
                torch.randn(4, 3), None, torch.randint(0, 3, (4,))
            )
        assert acc.empty is False
        result = acc.result("classification")
        assert "accuracy" in result
        preds = acc.predictions("classification")
        assert preds["targets"].shape == (8,)
        assert preds["preds"].shape == (8,)

    def test_regression_reduction(self):
        acc = EvaluationAccumulator(MetricsConfig())
        for _ in range(2):
            acc.update(None, torch.randn(4, 1), torch.randn(4, 1))
        result = acc.result("regression")
        assert "rmse" in result

    def test_reset(self):
        acc = EvaluationAccumulator()
        acc.update(None, torch.randn(2, 1), torch.randn(2, 1))
        acc.reset()
        assert acc.empty is True
