"""Evaluation report generation (Phase R5).

Turns the evaluation / ablation / error-analysis outcomes into the artefacts
the Evaluation report consumes:

* ``evaluation_report.md`` + ``evaluation_report.json`` — per-task extended
  metrics, per-class tables and latency;
* ``confusion_matrix.png``, ``pr_curves.png``, ``error_histogram.png`` —
  figures when matplotlib is available;
* ``ablation_report.md`` + ``ablation_report.json`` — variant comparison
  (metrics / parameters / inference speed);
* ``error_analysis.md`` + ``error_analysis.json`` — failure diagnostics;
* ``comparison.csv`` / ``comparison.md`` — the multimodal comparison table.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .ablation import AblationStudyReport
from .comparison import render_comparison_markdown, render_markdown_table
from .config import EvaluationConfig
from .error_analysis import ErrorAnalysisReport
from .evaluator import EvaluationOutcome

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib optional
    plt = None


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


# --------------------------------------------------------------------------- #
# Evaluation report
# --------------------------------------------------------------------------- #


def evaluation_report_markdown(
    outcome: EvaluationOutcome, config: EvaluationConfig
) -> str:
    """Markdown body: overview + per-task metric and per-class tables."""
    lines = [
        "# Evaluation Report",
        "",
        f"- **evaluation name**: {config.name}",
        f"- **generated**: {datetime.now(timezone.utc).isoformat()}",
        f"- **samples**: {outcome.num_samples}",
        f"- **latency ms (mean / p50 / p95)**: "
        f"{_fmt(outcome.latency_ms.get('mean'))} / "
        f"{_fmt(outcome.latency_ms.get('p50'))} / "
        f"{_fmt(outcome.latency_ms.get('p95'))}",
        "",
    ]
    for task, metrics in outcome.metrics.items():
        lines.append(f"## Task: {task}")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, (int, float)) and key not in {
                "per_class", "confusion_matrix",
            }:
                lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")
        per_class = outcome.per_class_tables.get(task)
        if per_class:
            lines.append(f"### Per-class ({task})")
            lines.append("")
            lines.append(
                render_markdown_table(
                    ["class", "precision", "recall", "f1", "support"],
                    [
                        [row["class"], row["precision"], row["recall"],
                         row["f1"], row["support"]]
                        for row in per_class
                    ],
                )
            )
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Ablation report
# --------------------------------------------------------------------------- #


def ablation_report_markdown(report: AblationStudyReport) -> str:
    """Markdown body: per-variant comparison table."""
    lines = [
        "# Ablation Study Report",
        "",
        f"- **base model**: {report.base_name}",
        f"- **compare metric**: {report.compare_metric} "
        f"({'higher is better' if report.compare_mode == 'max' else 'lower is better'})",
        f"- **best variant**: {report.best_variant or 'n/a'}",
        "",
        "## Variant comparison",
        "",
        "| variant | parameters | param delta | inference ms | speedup vs full |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in report.results:
        data = report.results[name]
        lines.append(
            f"| {name} | {data['parameter_count']} | "
            f"{_fmt(data['parameter_delta'])} | "
            f"{_fmt(data['inference_ms'])} | "
            f"{_fmt(data['speedup_vs_full'])} |"
        )
    lines.append("")
    lines.append("## Metric comparison")
    lines.append("")
    lines.append(render_comparison_markdown(report.comparison))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Error-analysis report
# --------------------------------------------------------------------------- #


def error_analysis_markdown(report: ErrorAnalysisReport) -> str:
    """Markdown body: failure diagnostics per task."""
    lines = ["# Error Analysis Report", ""]
    if not report.task_reports:
        lines.append("_No errors recorded._")
        lines.append("")
    for task, data in report.task_reports.items():
        lines.append(f"## Task: {task}")
        lines.append("")
        if task == "crop":
            lines.append(f"- **error rate**: {_fmt(data.get('error_rate'))}")
            lines.append(f"- **samples**: {data.get('num_samples')}")
            lines.append("")
            per_class = data.get("per_class", [])
            if per_class:
                lines.append("### Per-class error rates")
                lines.append("")
                lines.append(
                    render_markdown_table(
                        ["class", "support", "errors", "error rate", "false positives"],
                        [
                            [row["class"], row["support"], row["errors"],
                             _fmt(row["error_rate"]), row["false_positives"]]
                            for row in per_class
                        ],
                    )
                )
                lines.append("")
            confusions = data.get("top_confusions", [])
            if confusions:
                lines.append("### Top confusion pairs")
                lines.append("")
                lines.append(
                    render_markdown_table(
                        ["true", "pred", "count"],
                        [[row["true"], row["pred"], row["count"]] for row in confusions],
                    )
                )
                lines.append("")
        else:
            for key in (
                "num_samples",
                "mean_signed_error",
                "mean_absolute_error",
                "median_absolute_error",
                "max_absolute_error",
                "num_outliers",
                "outlier_fraction",
                "num_failures",
                "failure_fraction",
            ):
                lines.append(f"- **{key}**: {_fmt(data.get(key))}")
            lines.append("")
        group = data.get("group_breakdown")
        if group:
            lines.append("### Group breakdown")
            lines.append("")
            for key, buckets in group.items():
                lines.append(f"**{key}**")
                lines.append("")
                lines.append(
                    render_markdown_table(
                        ["group", "total", "errors", "error rate"],
                        [
                            [label, bucket["total"], bucket["errors"],
                             _fmt(bucket["error_rate"])]
                            for label, bucket in buckets.items()
                        ],
                    )
                )
                lines.append("")
    fusion = report.fusion_analysis
    if fusion:
        lines.append("## Fusion gate analysis")
        lines.append("")
        lines.append(
            f"- **task**: {fusion.get('task')} — {fusion.get('num_errors')} "
            f"errors across {fusion.get('num_samples')} samples"
        )
        lines.append("")
        rows = [
            [
                gate,
                _fmt(values.get("overall")),
                _fmt(values.get("correct")),
                _fmt(values.get("error")),
            ]
            for gate, values in fusion.get("gates", {}).items()
        ]
        if rows:
            lines.append(
                render_markdown_table(
                    ["gate", "overall", "correct", "error"], rows
                )
            )
            lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def confusion_matrix_figure(
    matrix: list[list[Any]], path: str | Path
) -> Path | None:
    """Render a confusion-matrix heatmap PNG."""
    if plt is None:
        return None
    data = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, data.shape[0] * 0.6), max(4, data.shape[1] * 0.6)))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    fig.colorbar(im)
    out = Path(path)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def pr_curves_figure(
    curves: list[Mapping[str, Any]], path: str | Path
) -> Path | None:
    """Render one-vs-rest PR curves per class."""
    if plt is None or not curves:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for curve in curves:
        ax.plot(curve["recall"], curve["precision"], label=f"class {curve['class']}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(fontsize=8)
    out = Path(path)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def error_histogram_figure(
    histogram: Mapping[str, Any], path: str | Path
) -> Path | None:
    """Render the prediction-error histogram."""
    if plt is None or not histogram:
        return None
    counts = histogram["counts"]
    edges = histogram["edges"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(edges[:-1], counts, width=np.diff(edges))
    ax.set_xlabel("prediction error")
    ax.set_ylabel("count")
    ax.set_title("Prediction-error distribution")
    out = Path(path)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def fusion_gate_figure(
    fusion_analysis: Mapping[str, Any], path: str | Path
) -> Path | None:
    """Render mean fusion-gate values for overall / correct / error samples."""
    if plt is None or not fusion_analysis:
        return None
    gates = fusion_analysis.get("gates") or {}
    if not gates:
        return None
    names = sorted(gates)
    buckets = ("overall", "correct", "error")
    fig, ax = plt.subplots(figsize=(max(5, len(names) * 1.6), 4))
    width = 0.8 / len(buckets)
    for offset, bucket in enumerate(buckets):
        values = [
            gates[name].get(bucket) if gates[name].get(bucket) is not None else 0.0
            for name in names
        ]
        positions = np.arange(len(names)) + offset * width
        ax.bar(
            positions, values, width, label=bucket,
            color=["#9db4c0", "#2e7d32", "#c62828"][offset],
        )
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("mean gate value")
    ax.set_title("Fusion gates vs. outcome (correct / error)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    out = Path(path)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def generate_evaluation_reports(
    outcome: EvaluationOutcome,
    config: EvaluationConfig,
    *,
    directory: str | Path,
) -> dict[str, Path]:
    """Write evaluation reports + figures into ``directory``."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    md_path = out_dir / "evaluation_report.md"
    md_path.write_text(evaluation_report_markdown(outcome, config), encoding="utf-8")
    paths["evaluation_markdown"] = md_path
    paths["evaluation_json"] = _write_json(
        out_dir / "evaluation_report.json", outcome.to_dict()
    )

    crop_metrics = outcome.metrics.get("crop", {})
    if crop_metrics.get("confusion_matrix"):
        fig = confusion_matrix_figure(
            crop_metrics["confusion_matrix"], out_dir / "confusion_matrix.png"
        )
        if fig:
            paths["confusion_matrix_png"] = fig
    if outcome.pr_curves.get("crop"):
        fig = pr_curves_figure(
            outcome.pr_curves["crop"], out_dir / "pr_curves.png"
        )
        if fig:
            paths["pr_curves_png"] = fig
    yield_metrics = outcome.metrics.get("yield", {})
    if yield_metrics.get("error_histogram"):
        fig = error_histogram_figure(
            yield_metrics["error_histogram"], out_dir / "error_histogram.png"
        )
        if fig:
            paths["error_histogram_png"] = fig

    per_class = outcome.per_class_tables.get("crop")
    if per_class:
        rows = [[
            row["class"], row["precision"], row["recall"], row["f1"], row["support"]
        ] for row in per_class]
        csv_path = out_dir / "per_class_comparison.csv"
        csv_path.write_text(
            "class,precision,recall,f1,support\n"
            + "\n".join(",".join(_fmt(cell) for cell in row) for row in rows)
            + "\n",
            encoding="utf-8",
        )
        paths["per_class_csv"] = csv_path
    return paths


def generate_ablation_reports(
    report: AblationStudyReport,
    *,
    directory: str | Path,
) -> dict[str, Path]:
    """Write the ablation study report (markdown + JSON)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "ablation_report.md"
    md_path.write_text(ablation_report_markdown(report), encoding="utf-8")
    return {
        "ablation_markdown": md_path,
        "ablation_json": _write_json(
            out_dir / "ablation_report.json", report.to_dict()
        ),
    }


def generate_error_analysis_reports(
    report: ErrorAnalysisReport,
    *,
    directory: str | Path,
) -> dict[str, Path]:
    """Write the error-analysis report (markdown + JSON)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "error_analysis.md"
    md_path.write_text(error_analysis_markdown(report), encoding="utf-8")
    paths = {
        "error_analysis_markdown": md_path,
        "error_analysis_json": _write_json(
            out_dir / "error_analysis.json", report.to_dict()
        ),
    }
    if report.fusion_analysis:
        fig = fusion_gate_figure(
            report.fusion_analysis, out_dir / "fusion_gates.png"
        )
        if fig:
            paths["fusion_gates_png"] = fig
    return paths


def generate_comparison_report(
    comparison: Mapping[str, Any],
    *,
    directory: str | Path,
) -> dict[str, Path]:
    """Write the multimodal comparison table (markdown + CSV)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison.md"
    md_path.write_text(render_comparison_markdown(comparison), encoding="utf-8")

    columns = comparison["columns"]
    rows = [
        [label, *[row.get(col, "") for col in columns]]
        for label, row in comparison["rows"].items()
    ]
    csv_path = out_dir / "comparison.csv"
    csv_path.write_text(
        "model," + ",".join(columns) + "\n"
        + "\n".join(
            ",".join(_fmt(cell) for cell in row) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return {"comparison_markdown": md_path, "comparison_csv": csv_path}


__all__ = [
    "generate_evaluation_reports",
    "generate_ablation_reports",
    "generate_error_analysis_reports",
    "generate_comparison_report",
    "evaluation_report_markdown",
    "ablation_report_markdown",
    "error_analysis_markdown",
    "confusion_matrix_figure",
    "pr_curves_figure",
    "error_histogram_figure",
    "fusion_gate_figure",
]
