"""Error analysis for the CropFusion model (Phase R5).

Turns an :class:`EvaluationOutcome` into actionable failure diagnostics:

* classification — confusion-matrix summary, per-class error rates, the top
  false-positive / false-negative pairs and misclassified samples;
* regression — residual distribution, outliers (|error| above a percentile
  threshold) and failure cases (relative error above a threshold), plus the
  worst predictions;
* group breakdowns — error rates grouped by optional sample metadata (village
  / district / season / year) so systematic weaknesses become visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ErrorAnalysisConfig
from .evaluator import EvaluationOutcome
from .exceptions import ErrorAnalysisError


@dataclass
class ErrorAnalysisReport:
    """Structured error-analysis result."""

    task_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    sample_metadata_keys: list[str] = field(default_factory=list)
    fusion_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_reports": self.task_reports,
            "metadata_keys": self.sample_metadata_keys,
            "fusion_analysis": self.fusion_analysis,
        }


class ErrorAnalysis:
    """Analyse the failures of one evaluation outcome.

    Args:
        config: Validated :class:`EvaluationConfig` (``None`` = defaults).
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        outcome: EvaluationOutcome,
        sample_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> ErrorAnalysisReport:
        """Analyse ``outcome`` (optionally with per-sample metadata)."""
        report = ErrorAnalysisReport()
        for task in outcome.metrics:
            predictions = outcome.predictions.get(task)
            if predictions is None or not predictions:
                continue
            if task == "crop":
                report.task_reports[task] = self._analyze_classification(
                    predictions, outcome
                )
            else:
                report.task_reports[task] = self._analyze_regression(
                    predictions, outcome
                )
            if sample_metadata is not None:
                report.task_reports[task]["group_breakdown"] = (
                    self._group_breakdown(
                        predictions, sample_metadata, task=task
                    )
                )
        report.sample_metadata_keys = (
            list(sample_metadata[0].keys()) if sample_metadata else []
        )
        report.fusion_analysis = self.analyze_gates(outcome)
        return report

    # ------------------------------------------------------------------ #
    # Fusion gate analysis (explainability tie-in)
    # ------------------------------------------------------------------ #

    def analyze_gates(
        self, outcome: EvaluationOutcome, task: str | None = None
    ) -> dict[str, Any] | None:
        """Compare fusion gate values on correct vs failing samples.

        Uses the per-sample ``image_gate`` / ``tabular_gate`` /
        ``fusion_gate`` collected by the evaluator to answer "does the model
        lean on the wrong modality exactly when it fails?".

        Args:
            outcome: The evaluation outcome (needs ``gates`` populated).
            task: Task to bucket errors on (defaults to the first task).
        """
        gates = outcome.gates
        if not gates:
            return None
        tasks = list(outcome.metrics)
        if not tasks:
            return None
        task = task or tasks[0]
        predictions = outcome.predictions.get(task)
        if predictions is None or "targets" not in predictions:
            return None
        targets = predictions["targets"]
        preds = predictions.get("preds")
        n = int(targets.size)
        if n == 0:
            return None

        if preds is not None and preds.size == n:
            correct = np.asarray(preds).reshape(-1) == np.asarray(targets).reshape(-1)
        else:
            abs_errors = np.abs(
                np.asarray(targets, dtype=np.float64).reshape(-1)
                - np.asarray(preds, dtype=np.float64).reshape(-1)
                if preds is not None
                else np.zeros(n)
            )
            threshold = float(np.median(abs_errors))
            correct = abs_errors <= threshold

        buckets = {
            "overall": np.ones(n, dtype=bool),
            "correct": correct,
            "error": ~correct,
        }
        per_gate: dict[str, dict[str, float | None]] = {}
        for gate_name, values in gates.items():
            values = np.asarray(values).reshape(-1)
            if values.size != n:
                continue
            per_gate[gate_name] = {
                bucket: (
                    float(values[selected].mean()) if selected.any() else None
                )
                for bucket, selected in buckets.items()
            }
        return {
            "task": task,
            "num_samples": n,
            "num_errors": int((~correct).sum()),
            "gates": per_gate,
        }

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #

    def _analyze_classification(
        self, predictions: dict[str, np.ndarray], outcome: EvaluationOutcome
    ) -> dict[str, Any]:
        targets = predictions["targets"].astype(np.int64)
        preds = predictions["preds"].astype(np.int64)
        probs = predictions.get("probs")
        num_classes = int(probs.shape[1]) if probs is not None else int(targets.max()) + 1

        correct = preds == targets
        num = int(targets.size)
        error_rate = float(1.0 - correct.mean()) if num else None

        # Per-class support, errors and FPR / FNR-style counts.
        per_class: list[dict[str, Any]] = []
        for cls in range(num_classes):
            is_cls = targets == cls
            support = int(is_cls.sum())
            errors = int((is_cls & ~correct).sum())
            # False positives for this class: predicted cls but different true.
            false_positives = int(((preds == cls) & (targets != cls)).sum())
            per_class.append(
                {
                    "class": cls,
                    "support": support,
                    "errors": errors,
                    "error_rate": float(errors / support) if support else None,
                    "false_positives": false_positives,
                }
            )

        # Misclassified samples (with confidence when probabilities exist).
        misclassified: list[dict[str, Any]] = []
        top_k = self.config.error_analysis.top_k_errors
        for idx in np.where(~correct)[0]:
            entry: dict[str, Any] = {
                "index": int(idx),
                "true": int(targets[idx]),
                "pred": int(preds[idx]),
            }
            if probs is not None:
                entry["true_prob"] = float(probs[idx, targets[idx]])
                entry["pred_prob"] = float(probs[idx, preds[idx]])
            misclassified.append(entry)
        misclassified.sort(key=lambda e: e.get("pred_prob", 1.0), reverse=True)
        misclassified = misclassified[:top_k]

        # Top FP / FN confusion pairs (true -> pred -> count).
        pairs: dict[tuple[int, int], int] = {}
        for true, pred, is_correct in zip(targets, preds, correct):
            if not is_correct:
                pairs[(int(true), int(pred))] = pairs.get((int(true), int(pred)), 0) + 1
        top_pairs = sorted(
            (
                {"true": t, "pred": p, "count": c}
                for (t, p), c in pairs.items()
            ),
            key=lambda e: e["count"],
            reverse=True,
        )[:top_k]

        return {
            "num_samples": num,
            "error_rate": error_rate,
            "per_class": per_class,
            "misclassified": misclassified,
            "top_confusions": top_pairs,
            "confusion_matrix": outcome.metrics.get("crop", {}).get(
                "confusion_matrix"
            ),
        }

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #

    def _analyze_regression(
        self, predictions: dict[str, np.ndarray], outcome: EvaluationOutcome
    ) -> dict[str, Any]:
        targets = predictions["targets"].astype(np.float64).reshape(-1)
        preds = predictions["preds"].astype(np.float64).reshape(-1)
        errors = targets - preds
        abs_errors = np.abs(errors)
        n = int(targets.size)

        threshold = float(
            np.percentile(abs_errors, self.config.error_analysis.outlier_percentile * 100)
        )
        rel = np.abs(errors) / np.maximum(np.abs(targets), 1e-8)
        failure_threshold = self.config.error_analysis.failure_relative_error

        top_k = self.config.error_analysis.top_k_errors
        worst = np.argsort(abs_errors)[::-1][:top_k]
        worst_rows = [
            {
                "index": int(idx),
                "target": float(targets[idx]),
                "pred": float(preds[idx]),
                "error": float(errors[idx]),
                "abs_error": float(abs_errors[idx]),
            }
            for idx in worst
        ]

        outlier_idx = np.where(abs_errors >= threshold)[0]
        failure_idx = np.where(rel >= failure_threshold)[0]

        return {
            "num_samples": n,
            "mean_signed_error": float(errors.mean()),
            "mean_absolute_error": float(abs_errors.mean()),
            "median_absolute_error": float(np.median(abs_errors)),
            "max_absolute_error": float(abs_errors.max()),
            "outlier_threshold": threshold,
            "num_outliers": int(outlier_idx.size),
            "outlier_fraction": float(outlier_idx.size / n) if n else None,
            "outliers": [
                {
                    "index": int(idx),
                    "target": float(targets[idx]),
                    "pred": float(preds[idx]),
                    "abs_error": float(abs_errors[idx]),
                }
                for idx in outlier_idx[:top_k]
            ],
            "failure_threshold": failure_threshold,
            "num_failures": int(failure_idx.size),
            "failure_fraction": float(failure_idx.size / n) if n else None,
            "failures": [
                {
                    "index": int(idx),
                    "target": float(targets[idx]),
                    "pred": float(preds[idx]),
                    "relative_error": float(rel[idx]),
                }
                for idx in failure_idx[:top_k]
            ],
            "worst_predictions": worst_rows,
        }

    # ------------------------------------------------------------------ #
    # Group breakdown
    # ------------------------------------------------------------------ #

    def _group_breakdown(
        self,
        predictions: dict[str, np.ndarray],
        sample_metadata: Sequence[Mapping[str, Any]],
        *,
        task: str,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Error rates grouped by each metadata key present on the samples."""
        targets = predictions["targets"]
        preds = predictions.get("preds")
        n = int(targets.size)
        if preds is None or n == 0:
            return {}
        if len(sample_metadata) != n:
            raise ErrorAnalysisError(
                f"sample_metadata length {len(sample_metadata)} does not match "
                f"samples {n}"
            )

        keys = set(sample_metadata[0].keys())
        breakdown: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for key in keys:
            groups: dict[str, dict[str, int]] = {}
            for meta, target, pred in zip(sample_metadata, targets, preds):
                label = str(meta.get(key, "unknown"))
                bucket = groups.setdefault(label, {"total": 0, "errors": 0})
                bucket["total"] += 1
                if int(pred) != int(target):
                    bucket["errors"] += 1
            breakdown[key] = {
                label: {
                    "total": counts["total"],
                    "errors": counts["errors"],
                    "error_rate": (
                        counts["errors"] / counts["total"]
                        if counts["total"]
                        else None
                    ),
                }
                for label, counts in groups.items()
            }
        return breakdown
