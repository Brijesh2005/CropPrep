"""Comparison tables for evaluation (Phase R5).

Builds the human- and machine-readable tables that answer the "how does the
model do per class / per modality / per variant" questions:

* per-class classification comparison (precision / recall / F1 / support),
* regression error-profile comparison,
* a multimodal comparison that rows up multiple models or variants side by
  side for the final evaluation report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .config import ComparisonConfig
from .exceptions import ComparisonError
from .evaluator import EvaluationOutcome
from .metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


def build_classification_comparison(
    logits: Any,
    targets: Any,
    config: ComparisonConfig | None = None,
) -> dict[str, Any]:
    """Per-class comparison table for one classification evaluation."""
    config = config or ComparisonConfig()
    metrics = compute_classification_metrics(logits, targets)
    rows = sorted(
        metrics.get("per_class", []),
        key=lambda row: row.get(config.sort_by, 0.0) if config.sort_by != "support" else row.get("support", 0),
        reverse=True,
    )
    if config.top_k_classes and len(rows) > config.top_k_classes:
        rows = rows[: config.top_k_classes]
    return {
        "rows": rows,
        "summary": {
            "accuracy": metrics.get("accuracy"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "auprc": metrics.get("auprc"),
        },
    }


def build_regression_comparison(
    preds: Any,
    targets: Any,
    config: ComparisonConfig | None = None,
) -> dict[str, Any]:
    """Error-profile table for one regression evaluation."""
    metrics = compute_regression_metrics(preds, targets)
    rows = [
        {"metric": key, "value": value}
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and key not in {"support"}
    ]
    return {"rows": rows, "summary": metrics}


def build_multimodal_comparison(
    outcomes: Mapping[str, EvaluationOutcome],
    *,
    metric_keys: Sequence[str] = (
        "crop/accuracy",
        "crop/balanced_accuracy",
        "crop/f1",
        "crop/roc_auc",
        "crop/auprc",
        "yield/rmse",
        "yield/mae",
        "yield/r2",
        "yield/median_absolute_error",
        "latency_ms",
    ),
) -> dict[str, Any]:
    """Row up multiple model / variant evaluations for comparison.

    Args:
        outcomes: Mapping of label -> :class:`EvaluationOutcome`.
        metric_keys: Slash-separated ``<task>/<metric>`` keys to tabulate
            (``latency_ms`` selects the latency statistic).

    Returns:
        A table dict with ``columns``, ``rows`` (label -> value) and ``best``
        (label with the best value per column, where higher is better unless
        the metric is an error metric).
    """
    if not outcomes:
        raise ComparisonError("multimodal comparison requires at least one outcome")

    rows: dict[str, dict[str, Any]] = {}
    for label, outcome in outcomes.items():
        row: dict[str, Any] = {"samples": outcome.num_samples}
        for key in metric_keys:
            row[key] = _lookup(outcome, key)
        rows[label] = row

    columns = [*(["samples"] if metric_keys else []), *metric_keys]
    best: dict[str, str | None] = {}
    for key in metric_keys:
        lower_is_better = key.endswith(("rmse", "mae", "bias", "latency_ms"))
        candidates = [
            (label, row.get(key)) for label, row in rows.items()
            if row.get(key) is not None
        ]
        if not candidates:
            best[key] = None
            continue
        best[key] = (
            min(candidates, key=lambda pair: float(pair[1]))
            if lower_is_better
            else max(candidates, key=lambda pair: float(pair[1]))
        )[0]
    return {"columns": columns, "rows": rows, "best": best}


def _lookup(outcome: EvaluationOutcome, key: str) -> Any:
    if key == "latency_ms":
        return outcome.latency_ms.get("mean")
    if "/" not in key:
        return None
    task, metric = key.split("/", 1)
    metrics = outcome.metrics.get(task, {})
    return metrics.get(metric)


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #


def render_markdown_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]]
) -> str:
    """Render a markdown table with aligned columns."""
    if not rows:
        return "| " + " | ".join(str(h) for h in headers) + " |\n"
    str_rows = [[_fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in str_rows))
        for i in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(
            str(headers[i]).ljust(widths[i]) for i in range(len(headers))
        ) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    for row in str_rows:
        lines.append(
            "| " + " | ".join(
                row[i].ljust(widths[i]) for i in range(len(headers))
            ) + " |"
        )
    return "\n".join(lines) + "\n"


def render_comparison_markdown(table: Mapping[str, Any]) -> str:
    """Render a multimodal comparison table as markdown."""
    columns = table["columns"]
    rows = table["rows"]
    data = [
        [label, *[row.get(col, "") for col in columns]]
        for label, row in rows.items()
    ]
    headers = ["model"] + list(columns)
    return render_markdown_table(headers, data)


@dataclass
class ComparisonTable:
    """A named comparison table plus its markdown rendering."""

    name: str
    headers: list[str]
    rows: list[list[Any]] = field(default_factory=list)

    def to_markdown(self) -> str:
        return render_markdown_table(self.headers, self.rows)

    def to_csv(self) -> str:
        lines = [",".join(str(h) for h in self.headers)]
        for row in self.rows:
            lines.append(",".join(_fmt(cell) for cell in row))
        return "\n".join(lines) + "\n"


def _fmt(cell: Any) -> str:
    if isinstance(cell, float):
        return f"{cell:.4f}"
    return "None" if cell is None else str(cell)
