"""End-of-run reports for the CropFusion training engine.

:func:`generate_reports` turns a finished :class:`~ai.training.trainer.
TrainingResult` (and the resolved :class:`~ai.training.config.TrainingConfig`)
into five artefacts:

* ``training_report.md`` — run overview: config, epoch budget, duration and
  best-metric summary.
* ``validation_report.md`` — per-epoch validation metrics table.
* ``metrics_report.md`` — final classification + regression metric tables.
* ``checkpoint_report.md`` — checkpoint policy and the saved artifact list.
* ``learning_curve.csv`` — per-epoch scalar metrics (loss / accuracy / LR) for
  external plotting and the dashboard.

All values come from the in-memory training history — nothing is re-read from
disk inside the loop.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import TrainingConfig
from .trainer import TrainingResult

REPORT_TYPES: tuple[str, ...] = (
    "training",
    "validation",
    "metrics",
    "checkpoint",
    "learning_curve",
)

_REPORT_FILENAMES: dict[str, str] = {
    "training": "training_report.md",
    "validation": "validation_report.md",
    "metrics": "metrics_report.md",
    "checkpoint": "checkpoint_report.md",
    "learning_curve": "learning_curve.csv",
}


def default_reports_dir(config: TrainingConfig) -> Path:
    """``<output_dir>/reports`` unless ``general.reports_dir`` is set."""
    if config.general.reports_dir:
        return Path(config.general.reports_dir)
    return Path(config.general.output_dir) / "reports"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _scalar_rows(history: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Flatten each epoch log into scalars only (CSV-safe)."""
    rows: list[dict[str, Any]] = []
    seen: list[str] = []
    for log in history:
        row: dict[str, Any] = {}
        for key, value in (log or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
                if key not in seen:
                    seen.append(key)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Individual report bodies
# --------------------------------------------------------------------------- #


def training_report(config: TrainingConfig, result: TrainingResult) -> str:
    """Markdown: run overview + best-metric summary."""
    best = result.best_metrics or {}
    lines = [
        "# Training Report",
        "",
        f"- **run name**: {config.name}",
        f"- **generated**: {datetime.now(timezone.utc).isoformat()}",
        f"- **epochs completed**: {result.epochs}",
        f"- **optimizer steps**: {result.steps}",
        f"- **duration (s)**: {_fmt(result.duration_seconds)}",
        f"- **stopped early**: {result.stopped_early}",
        f"- **best epoch**: {result.best_epoch if result.best_epoch is not None else 'n/a'}",
        f"- **best checkpoint**: {result.best_path or 'n/a'}",
        "",
        "## Best metrics",
        "",
    ]
    if best:
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in sorted(best):
            lines.append(f"| {key} | {_fmt(best[key])} |")
    else:
        lines.append("_No best metrics recorded._")
    lines.append("")
    return "\n".join(lines)


def validation_report(config: TrainingConfig, result: TrainingResult) -> str:
    """Markdown: per-epoch validation metrics table."""
    val_cols: list[str] = []
    for log in result.history:
        for key, value in (log or {}).items():
            if key.startswith("val_") or "/" in key and key.split("/")[0] in (
                "val",
                "crop",
                "yield",
            ):
                if isinstance(value, (int, float)) and key not in val_cols:
                    val_cols.append(key)
    lines = ["# Validation Report", ""]
    if not val_cols:
        lines.append("_No validation metrics recorded (no validation loader)._")
        lines.append("")
        return "\n".join(lines)
    lines.append("| epoch | " + " | ".join(val_cols) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(val_cols)) + " |")
    for log in result.history:
        epoch = log.get("epoch", "")
        values = [_fmt(log.get(key, "")) for key in val_cols]
        lines.append(f"| {epoch} | " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def metrics_report(config: TrainingConfig, result: TrainingResult) -> str:
    """Markdown: final classification + regression metric tables."""
    if not result.history:
        return "# Metrics Report\n\n_No epochs recorded._\n"
    last = result.history[-1]
    lines = [
        "# Metrics Report",
        "",
        f"Metrics from epoch **{last.get('epoch', 'n/a')}**.",
        "",
    ]
    classification = [k for k in last if k.startswith("crop/")]
    regression = [k for k in last if k.startswith("yield/")]
    if classification:
        lines.append("## Classification (crop)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in sorted(classification):
            lines.append(f"| {key} | {_fmt(last[key])} |")
        lines.append("")
    if regression:
        lines.append("## Regression (yield)")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("| --- | --- |")
        for key in sorted(regression):
            lines.append(f"| {key} | {_fmt(last[key])} |")
        lines.append("")
    if not classification and not regression:
        lines.append("_No crop / yield metrics recorded._")
        lines.append("")
    return "\n".join(lines)


def checkpoint_report(config: TrainingConfig, result: TrainingResult) -> str:
    """Markdown: checkpoint policy + saved artifact list."""
    ckpt = config.checkpoint
    lines = [
        "# Checkpoint Report",
        "",
        f"- **directory**: {ckpt.directory}",
        f"- **save best**: {ckpt.save_best}",
        f"- **save latest**: {ckpt.save_latest}",
        f"- **save periodic**: {ckpt.save_periodic if ckpt.save_periodic else 'disabled'}",
        f"- **keep last**: {ckpt.keep_last if ckpt.keep_last else 'all'}",
        f"- **resume**: {ckpt.resume}",
        f"- **best checkpoint**: {result.best_path or 'n/a'}",
        "",
    ]
    directory = Path(ckpt.directory)
    if directory.exists():
        artifacts = sorted(p.name for p in directory.glob("*.pt"))
        if artifacts:
            lines.append("## Artifacts")
            lines.append("")
            for name in artifacts:
                lines.append(f"- `{name}`")
            lines.append("")
    else:
        lines.append("_No checkpoint directory found._")
        lines.append("")
    return "\n".join(lines)


def learning_curve_csv(result: TrainingResult) -> str:
    """CSV: per-epoch scalar metrics (loss / accuracy / LR)."""
    rows = _scalar_rows(result.history)
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([_fmt(row.get(key, "")) for key in columns])
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def generate_reports(
    config: TrainingConfig,
    result: TrainingResult,
    *,
    directory: str | Path | None = None,
) -> dict[str, Path]:
    """Write all five reports and return ``{report_type: path}``.

    Args:
        config: The resolved training config.
        result: The finished :class:`TrainingResult`.
        directory: Output directory (defaults to
            :func:`default_reports_dir`).

    Returns:
        Mapping of report type -> written file path.
    """
    out_dir = Path(directory) if directory is not None else default_reports_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    bodies: dict[str, str] = {
        "training": training_report(config, result),
        "validation": validation_report(config, result),
        "metrics": metrics_report(config, result),
        "checkpoint": checkpoint_report(config, result),
        "learning_curve": learning_curve_csv(result),
    }
    paths: dict[str, Path] = {}
    for report_type in REPORT_TYPES:
        path = out_dir / _REPORT_FILENAMES[report_type]
        path.write_text(bodies[report_type], encoding="utf-8")
        paths[report_type] = path
    return paths


__all__ = [
    "REPORT_TYPES",
    "default_reports_dir",
    "generate_reports",
    "training_report",
    "validation_report",
    "metrics_report",
    "checkpoint_report",
    "learning_curve_csv",
]
