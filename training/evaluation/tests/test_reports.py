"""Report generation tests (Phase R5)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from training.evaluation.reports import (
    evaluation_report_markdown,
    fusion_gate_figure,
    generate_ablation_reports,
    generate_comparison_report,
    generate_error_analysis_reports,
    generate_evaluation_reports,
)
from training.evaluation.ablation import AblationStudyReport
from training.evaluation.error_analysis import ErrorAnalysisReport
from training.evaluation.evaluator import EvaluationOutcome
from training.evaluation.config import EvaluationConfig


def _outcome() -> EvaluationOutcome:
    outcome = EvaluationOutcome(num_samples=8)
    outcome.metrics = {
        "crop": {
            "accuracy": 0.75,
            "f1": 0.7,
            "confusion_matrix": [[2, 1], [1, 4]],
        },
        "yield": {"rmse": 1.0, "mae": 0.8},
    }
    outcome.latency_ms = {"mean": 3.0, "p50": 2.9, "p95": 3.5}
    outcome.pr_curves = {
        "crop": [{"class": 0, "precision": [1.0], "recall": [1.0]}]
    }
    outcome.per_class_tables = {
        "crop": [
            {"class": 0, "precision": 0.8, "recall": 0.7, "f1": 0.74, "support": 4},
            {"class": 1, "precision": 0.8, "recall": 0.9, "f1": 0.85, "support": 4},
        ]
    }
    outcome.predictions = {
        "crop": {
            "targets": np.array([0, 1, 0, 1]),
            "preds": np.array([0, 1, 1, 1]),
        }
    }
    return outcome


def test_evaluation_markdown():
    md = evaluation_report_markdown(_outcome(), EvaluationConfig())
    assert "# Evaluation Report" in md
    assert "## Task: crop" in md
    assert "| metric | value |" in md


def test_generate_evaluation_reports(tmp_path):
    outcome = _outcome()
    paths = generate_evaluation_reports(
        outcome, EvaluationConfig(), directory=str(tmp_path)
    )
    assert (tmp_path / "evaluation_report.md").exists()
    assert (tmp_path / "evaluation_report.json").exists()
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "per_class_comparison.csv").exists()
    payload = json.loads((tmp_path / "evaluation_report.json").read_text(encoding="utf-8"))
    assert payload["num_samples"] == 8


def test_generate_ablation_reports(tmp_path):
    report = AblationStudyReport(
        base_name="cropfusion_v1",
        compare_metric="crop/f1",
        compare_mode="max",
        best_variant="without_confidence_fusion",
        results={
            "without_confidence_fusion": {
                "parameter_count": 100, "parameter_delta": -5,
                "inference_ms": 2.0, "speedup_vs_full": 1.5,
                "metrics": {},
            }
        },
        comparison={
            "columns": ["crop/accuracy"],
            "rows": {
                "without_confidence_fusion": {"crop/accuracy": 0.9},
            },
        },
    )
    paths = generate_ablation_reports(report, directory=str(tmp_path))
    assert paths["ablation_markdown"].exists()
    assert "## Variant comparison" in paths["ablation_markdown"].read_text(encoding="utf-8")


def test_generate_error_analysis_reports(tmp_path):
    report = ErrorAnalysisReport(
        task_reports={},
        fusion_analysis={
            "task": "crop",
            "num_samples": 8,
            "num_errors": 2,
            "gates": {
                "image_gate": {"overall": 0.4, "correct": 0.5, "error": 0.2},
                "tabular_gate": {"overall": 0.6, "correct": 0.5, "error": 0.8},
            },
        },
    )
    paths = generate_error_analysis_reports(report, directory=str(tmp_path))
    md = paths["error_analysis_markdown"].read_text(encoding="utf-8")
    assert "## Fusion gate analysis" in md
    assert "gate" in md and "overall" in md and "correct" in md
    assert "image_gate" in md
    assert (tmp_path / "fusion_gates.png").exists()


def test_generate_comparison_report(tmp_path):
    table = {
        "columns": ["crop/accuracy", "latency_ms"],
        "rows": {"a": {"crop/accuracy": 0.8, "latency_ms": 3.0}},
        "best": {"crop/accuracy": "a", "latency_ms": "a"},
    }
    paths = generate_comparison_report(table, directory=str(tmp_path))
    assert "| model |" in paths["comparison_markdown"].read_text(encoding="utf-8")
    assert paths["comparison_csv"].exists()
