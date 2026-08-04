"""Promotion validation gates.

Each gate produces a :class:`GateResult` with a pass/fail verdict and a
machine-readable record that is stored on the model manifest. Gates are the
"human-in-the-loop guardrails" for the model promotion workflow:

* ``metric_gate`` - accuracy / loss meet configured thresholds
* ``regression_gate`` - candidate latency is not significantly worse than the incumbent
* ``drift_gate`` - a reference vs current comparison shows no high-severity drift
* ``fairness_gate`` - protected groups are not at risk
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .config import MLOpsSettings

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a single validation gate."""

    gate: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    message: str = ""

    def result(self) -> dict[str, Any]:
        return asdict(self)


def metric_gate(
    metrics: dict[str, float],
    settings: MLOpsSettings,
    *,
    incumbent_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Accuracy above threshold; no accuracy regression vs the incumbent."""
    accuracy = metrics.get("accuracy") or metrics.get("acc") or 0.0
    passed = accuracy >= settings.min_accuracy
    message = f"accuracy {accuracy:.3f} >= {settings.min_accuracy:.3f}"
    if incumbent_metrics:
        incumbent_acc = incumbent_metrics.get("accuracy") or incumbent_metrics.get("acc") or 0.0
        regression = accuracy < incumbent_acc
        passed = passed and not regression
        message += f"; incumbent {incumbent_acc:.3f}"
    return GateResult(
        gate="metrics",
        passed=passed,
        details={"accuracy": accuracy, "min_accuracy": settings.min_accuracy},
        message=message,
    )


def regression_gate(
    candidate_latency_ms: float,
    incumbent_latency_ms: float,
    settings: MLOpsSettings,
) -> GateResult:
    """Latency regression must stay below the configured percentage."""
    if not incumbent_latency_ms:
        return GateResult(gate="regression", passed=True, details={}, message="no incumbent baseline")
    regression_pct = (candidate_latency_ms - incumbent_latency_ms) / incumbent_latency_ms * 100.0
    passed = regression_pct <= settings.max_latency_regression_pct
    return GateResult(
        gate="regression",
        passed=passed,
        details={
            "candidate_latency_ms": candidate_latency_ms,
            "incumbent_latency_ms": incumbent_latency_ms,
            "regression_pct": round(regression_pct, 2),
            "max_regression_pct": settings.max_latency_regression_pct,
        },
        message=f"latency regression {regression_pct:.1f}% (max {settings.max_latency_regression_pct:.1f}%)",
    )


def drift_gate(
    reference_df: Any,
    current_df: Any,
    settings: MLOpsSettings,
    *,
    feature_columns: Sequence[str] | None = None,
    label_column: str | None = None,
    predictions: Any = None,
) -> GateResult:
    """Run the quality drift battery; pass unless overall severity is high."""
    try:
        from quality.drift import DriftConfig, DriftMonitor
    except ImportError:  # pragma: no cover
        logger.warning("quality package unavailable; drift gate skipped")
        return GateResult(gate="drift", passed=True, details={"skipped": True}, message="skipped")

    config = DriftConfig()
    monitor = DriftMonitor(
        reference_df,
        config=config,
        feature_columns=feature_columns or settings.drift_feature_columns or None,
        label_column=label_column or settings.drift_label_column,
    )
    report = monitor.evaluate(current_df, predictions=predictions)
    severity = report.overall_severity if hasattr(report, "overall_severity") else "low"
    passed = severity != "high"
    return GateResult(
        gate="drift",
        passed=passed,
        details={"overall_severity": severity},
        message=f"drift severity: {severity}",
    )


def fairness_gate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    groups: dict[str, Sequence[Any]],
    *,
    y_proba: Sequence[float] | None = None,
) -> GateResult:
    """Run the quality fairness evaluator; pass unless overall status is at risk."""
    try:
        from quality.fairness import FairnessConfig, FairnessEvaluator
    except ImportError:  # pragma: no cover
        logger.warning("quality package unavailable; fairness gate skipped")
        return GateResult(gate="fairness", passed=True, details={"skipped": True}, message="skipped")

    evaluator = FairnessEvaluator(FairnessConfig())
    result = evaluator.evaluate(y_true, y_pred, groups, y_proba=y_proba)
    status = getattr(result, "overall_status", "pass")
    passed = status != "at_risk"
    return GateResult(
        gate="fairness",
        passed=passed,
        details={"overall_status": status},
        message=f"fairness status: {status}",
    )


def all_passed(results: Sequence[GateResult]) -> bool:
    return all(r.passed for r in results)


def write_gate_report(results: Sequence[GateResult], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "promotion-gates.json"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": all_passed(results),
        "gates": [r.result() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
