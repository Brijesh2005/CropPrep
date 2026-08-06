"""Comparison table tests (Phase R5)."""

from __future__ import annotations

import pytest
import torch

from training.evaluation.comparison import (
    build_classification_comparison,
    build_multimodal_comparison,
    build_regression_comparison,
    render_comparison_markdown,
    render_markdown_table,
)
from training.evaluation.config import ComparisonConfig
from training.evaluation.exceptions import ComparisonError
from training.evaluation.evaluator import EvaluationOutcome


class TestBuilders:
    def test_classification_comparison(self):
        logits = torch.randn(20, 3)
        targets = torch.randint(0, 3, (20,))
        table = build_classification_comparison(logits, targets)
        assert table["rows"]
        assert "accuracy" in table["summary"]
        assert len(table["rows"]) == 3

    def test_classification_top_k(self):
        logits = torch.randn(30, 5)
        targets = torch.randint(0, 5, (30,))
        table = build_classification_comparison(
            logits, targets, ComparisonConfig(top_k_classes=2)
        )
        assert len(table["rows"]) == 2

    def test_regression_comparison(self):
        table = build_regression_comparison(torch.randn(10, 1), torch.randn(10, 1))
        assert "rmse" in table["summary"]
        assert any(row["metric"] == "bias" for row in table["rows"])

    def test_multimodal_comparison(self):
        outcome = EvaluationOutcome(
            metrics={"crop": {"accuracy": 0.8, "f1": 0.7},
                     "yield": {"rmse": 1.2}},
            latency_ms={"mean": 5.0},
            num_samples=10,
        )
        table = build_multimodal_comparison({"model_a": outcome, "model_b": outcome})
        assert table["columns"] == [
            "samples", "crop/accuracy", "crop/balanced_accuracy", "crop/f1",
            "crop/roc_auc", "crop/auprc", "yield/rmse", "yield/mae", "yield/r2",
            "yield/median_absolute_error", "latency_ms",
        ]
        assert table["rows"]["model_a"]["crop/accuracy"] == 0.8
        assert table["best"]["crop/accuracy"] in {"model_a", "model_b"}

    def test_empty_raises(self):
        with pytest.raises(ComparisonError):
            build_multimodal_comparison({})


class TestRendering:
    def test_markdown_table(self):
        md = render_markdown_table(["a", "b"], [[1, 2.0], [3, 4.5]])
        assert md.startswith("| a")
        assert "---" in md

    def test_comparison_markdown(self):
        outcome = EvaluationOutcome(
            metrics={"crop": {"accuracy": 0.9}}, num_samples=8
        )
        table = build_multimodal_comparison({"m": outcome})
        md = render_comparison_markdown(table)
        assert "| model |" in md
