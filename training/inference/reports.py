"""Report generation for the inference package (Phase R5).

Writes the Export report (formats + metadata + checksums) and the Inference
Package report (artifact inventory, versions, validation summary) used by the
R5 deliverables.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .package_builder import BuildReport


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def export_report_markdown(paths: Mapping[str, Path]) -> str:
    """Markdown: model export bundle summary."""
    lines = [
        "# Model Export Report",
        "",
        f"- **generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Artifacts",
        "",
        "| key | path |",
        "| --- | --- |",
    ]
    for key in sorted(paths):
        lines.append(f"| {key} | `{paths[key]}` |")
    lines.append("")
    return "\n".join(lines)


def inference_package_report_markdown(report: BuildReport) -> str:
    """Markdown: inference package inventory + validation summary."""
    manifest = report.manifest
    lines = [
        "# Inference Package Report",
        "",
        f"- **package name**: {manifest.get('package_name')}",
        f"- **package version**: {manifest.get('package_version')}",
        f"- **model version**: {manifest.get('model_version')}",
        f"- **dataset version**: {manifest.get('dataset_version')}",
        f"- **generated**: {manifest.get('generated_at')}",
        f"- **git commit**: {manifest.get('git_commit') or 'n/a'}",
        f"- **output dir**: `{report.output_dir}`",
        "",
        "## Artifacts",
        "",
        "| artifact | sha256 |",
        "| --- | --- |",
    ]
    for name in sorted(manifest.get("files", {})):
        lines.append(f"| `{name}` | `{manifest['files'][name]}` |")
    lines.append("")
    if report.validation:
        validation = report.validation
        if isinstance(validation, dict) and "checks" in validation:
            lines.append("## Validation")
            lines.append("")
            lines.append(f"- **valid**: {validation.get('valid')}")
            for check, ok in validation.get("checks", {}).items():
                lines.append(f"- **{check}**: {'ok' if ok else 'FAILED'}")
            for error in validation.get("errors", []):
                lines.append(f"- error: {error}")
            lines.append("")
        else:
            validation = validation.to_dict() if hasattr(validation, "to_dict") else validation
            lines.append("## Validation")
            lines.append("")
            lines.append(f"- **valid**: {validation.get('valid')}")
            for check, ok in validation.get("checks", {}).items():
                lines.append(f"- **{check}**: {'ok' if ok else 'FAILED'}")
            for error in validation.get("errors", []):
                lines.append(f"- error: {error}")
            lines.append("")
    return "\n".join(lines)


def generate_export_reports(
    paths: Mapping[str, Path], *, directory: str | Path
) -> dict[str, Path]:
    """Write the export report (markdown + JSON)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "export_report.md"
    md.write_text(export_report_markdown(paths), encoding="utf-8")
    js = out_dir / "export_report.json"
    js.write_text(
        json.dumps({k: str(v) for k, v in paths.items()}, indent=2),
        encoding="utf-8",
    )
    return {"export_markdown": md, "export_json": js}


def generate_inference_package_reports(
    report: BuildReport, *, directory: str | Path
) -> dict[str, Path]:
    """Write the inference package report (markdown + JSON)."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "inference_package_report.md"
    md.write_text(inference_package_report_markdown(report), encoding="utf-8")
    js = out_dir / "inference_package_report.json"
    js.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return {"package_markdown": md, "package_json": js}


__all__ = [
    "generate_export_reports",
    "generate_inference_package_reports",
    "export_report_markdown",
    "inference_package_report_markdown",
]
