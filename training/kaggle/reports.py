"""Reports for the Kaggle Training Platform (R2.1).

Pure reporting helpers that turn the environment / workspace / configuration
managers into JSON-serialisable report dicts:

* environment      — combined capability report,
* gpu              — GPU / CUDA details,
* dependency       — installed package versions,
* storage          — disk + cache usage,
* workspace        — layout, cache and checkpoint summary,
* configuration    — resolved config registry + effective settings.

All reports are plain dicts; :func:`write_reports` persists each one as JSON
under an output directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import PathsConfig, WorkspaceLayout


def environment_report(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Combined environment capability report."""
    report = dict(environment)
    report["generated_at"] = _now()
    return report


def gpu_report(environment: Mapping[str, Any]) -> dict[str, Any]:
    """GPU / CUDA section of the environment report."""
    gpu = dict(environment.get("gpu", {}))
    gpu["generated_at"] = _now()
    return gpu


def dependency_report(environment: Mapping[str, Any]) -> dict[str, Any]:
    """Installed dependency versions (sorted by name)."""
    deps = dict(environment.get("dependencies", {}))
    return {
        "generated_at": _now(),
        "count": len(deps),
        "installed": {
            name: info.get("version") for name, info in sorted(deps.items())
        },
    }


def storage_report(
    workspace: Any,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Disk + cache usage for the workspace outputs volume."""
    layout = workspace.layout
    from .environment.system import detect_disk

    disk = detect_disk(layout.outputs)
    cache_stats = workspace.cache.stats()
    return {
        "generated_at": _now(),
        "outputs_path": str(layout.outputs),
        "disk": disk,
        "cache": cache_stats,
    }


def workspace_report(workspace: Any) -> dict[str, Any]:
    """Workspace layout + checkpoint/cache summary."""
    report = workspace.report()
    report["generated_at"] = _now()
    return report


def configuration_report(
    paths: PathsConfig,
    layout: WorkspaceLayout,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Effective config registry + workspace layout."""
    return {
        "generated_at": _now(),
        "workspace": layout.as_dict(),
        "config_registry": paths.config.snapshot(),
        "environment_requirements": paths.environment.model_dump(),
        "extra": dict(extra or {}),
    }


def write_reports(
    reports: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist each report as ``<name>.json``; returns name → file path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for name, report in reports.items():
        target = out / f"{name}.json"
        target.write_text(
            json.dumps(report, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        written[name] = str(target)
    return written


def now() -> str:
    """UTC timestamp used across all reports."""
    return datetime.now(timezone.utc).isoformat()


def _now() -> str:  # legacy alias
    return now()
