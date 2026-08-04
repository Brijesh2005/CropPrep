"""Report generation for benchmarks and model releases.

* :func:`write_benchmark_report` - renders an OptimizationBenchmark result into
  Markdown + JSON under the reports directory.
* :func:`write_release_report` - summarizes the promotion decision, gate
  results and registry state for a release package.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MLOpsSettings
from .gates import GateResult
from .registry import ModelRegistry


def write_benchmark_report(
    benchmark_result: dict[str, Any],
    settings: MLOpsSettings,
    *,
    model_name: str,
    version: str,
) -> Path:
    """Persist a benchmark result as JSON + Markdown."""
    out_dir = Path(settings.reports_dir) / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"{model_name}-{version}-{stamp}.json"
    json_path.write_text(json.dumps(benchmark_result, indent=2, default=str), encoding="utf-8")

    md_path = out_dir / f"{model_name}-{version}-{stamp}.md"
    variants = benchmark_result.get("variants", [])
    rows = "\n".join(
        f"| {v.get('mode', '?')} | {v.get('mean_latency_ms', 0):.2f} ms | "
        f"{v.get('p95_latency_ms', 0):.2f} ms | {v.get('speedup', 1.0):.2f}x | "
        f"{v.get('throughput_qps', 0):.1f} qps |"
        for v in variants
    )
    md_path.write_text(
        f"# Benchmark - {model_name}@{version}\n\n"
        f"Generated {datetime.now(timezone.utc).isoformat()}\n\n"
        "| Mode | Mean latency | p95 | Speedup | Throughput |\n"
        "|---|---:|---:|---:|---:|\n" + rows + "\n",
        encoding="utf-8",
    )
    return json_path


def write_release_report(
    settings: MLOpsSettings,
    *,
    model_name: str,
    version: str,
    target: str,
    gates: list[GateResult],
    registry: ModelRegistry,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a model release report (Markdown + JSON)."""
    out_dir = Path(settings.reports_dir) / "releases"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    record = registry.get(model_name, version)
    payload: dict[str, Any] = {
        "model": model_name,
        "version": version,
        "target": target,
        "promoted_at": record.manifest.promoted_at,
        "promoted_by": record.manifest.promoted_by,
        "metrics": record.manifest.metrics,
        "hyperparameters": record.manifest.hyperparameters,
        "git_commit": record.manifest.git_commit,
        "checkpoint_path": record.manifest.checkpoint_path,
        "gates": [g.result() for g in gates],
        "overall_pass": all(g.passed for g in gates),
        **(extra or {}),
    }
    json_path = out_dir / f"{model_name}-{version}-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    gate_lines = "\n".join(
        f"- [{'x' if g.passed else ' '}] **{g.gate}**: {g.message}"
        for g in gates
    )
    md_path = out_dir / f"{model_name}-{version}-{stamp}.md"
    md_path.write_text(
        f"# Model release - {model_name}@{version} -> {target}\n\n"
        f"- **Promoted**: {record.manifest.promoted_at}\n"
        f"- **Metrics**: {json.dumps(record.manifest.metrics, default=str)}\n"
        f"- **Checkpoint**: `{record.manifest.checkpoint_path}`\n\n"
        f"## Validation gates\n\n{gate_lines}\n",
        encoding="utf-8",
    )
    return md_path
