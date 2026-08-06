"""Error analysis tests (Phase R5)."""

from __future__ import annotations

import numpy as np
import pytest

from training.evaluation.config import EvaluationConfig
from training.evaluation.error_analysis import ErrorAnalysis, ErrorAnalysisReport
from training.evaluation.evaluator import EvaluationOutcome


def _outcome() -> EvaluationOutcome:
    outcome = EvaluationOutcome(num_samples=8)
    outcome.metrics = {
        "crop": {"accuracy": 0.75, "confusion_matrix": [[2, 1], [1, 4]]},
        "yield": {"rmse": 1.0},
    }
    outcome.predictions = {
        "crop": {
            "targets": np.array([0, 1, 0, 1, 1, 0, 1, 1]),
            "preds": np.array([0, 1, 1, 1, 1, 0, 0, 1]),
            "probs": np.full((8, 2), 0.5),
        },
        "yield": {
            "targets": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            "preds": np.array([1.2, 1.8, 3.5, 4.2, 4.0, 6.5, 6.0, 9.0]),
        },
    }
    outcome.gates = {
        "image_gate": np.linspace(0.0, 0.7, 8),
        "tabular_gate": np.linspace(0.3, 1.0, 8),
        "fusion_gate": np.full(8, 0.5),
    }
    return outcome


class TestErrorAnalysis:
    def test_classification_report(self):
        report = ErrorAnalysis(EvaluationConfig()).analyze(_outcome())
        crop = report.task_reports["crop"]
        assert crop["error_rate"] == pytest.approx(0.25)
        assert crop["num_samples"] == 8
        assert len(crop["per_class"]) == 2
        assert crop["per_class"][0]["false_positives"] == 1
        assert crop["misclassified"]
        assert crop["top_confusions"]

    def test_regression_report(self):
        report = ErrorAnalysis(EvaluationConfig()).analyze(_outcome())
        yield_ = report.task_reports["yield"]
        assert "median_absolute_error" in yield_
        assert yield_["worst_predictions"]
        assert "outliers" in yield_

    def test_group_breakdown(self):
        metadata = [
            {"village": "a" if i % 2 == 0 else "b"} for i in range(8)
        ]
        report = ErrorAnalysis(EvaluationConfig()).analyze(_outcome(), metadata)
        groups = report.task_reports["crop"]["group_breakdown"]["village"]
        assert set(groups) == {"a", "b"}
        assert report.sample_metadata_keys == ["village"]

    def test_group_breakdown_length_mismatch(self):
        with pytest.raises(Exception):
            ErrorAnalysis(EvaluationConfig()).analyze(_outcome(), [{}])


class TestFusionGateAnalysis:
    def test_gate_analysis(self):
        analysis = ErrorAnalysis(EvaluationConfig()).analyze_gates(_outcome())
        assert analysis is not None
        assert analysis["task"] == "crop"
        assert analysis["num_samples"] == 8
        assert analysis["num_errors"] == 2
        assert "image_gate" in analysis["gates"]
        assert set(analysis["gates"]["image_gate"]) == {
            "overall", "correct", "error"
        }

    def test_no_gates_returns_none(self):
        outcome = _outcome()
        outcome.gates = None
        assert ErrorAnalysis(EvaluationConfig()).analyze_gates(outcome) is None

    def test_analyze_populates_fusion(self):
        report = ErrorAnalysis(EvaluationConfig()).analyze(_outcome())
        assert report.fusion_analysis is not None
