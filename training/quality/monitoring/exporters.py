"""Exporters that push ML-QA verdicts into the Prometheus registry.

Each exporter reads an already-serialised report (drift JSON, fairness JSON)
and updates the matching gauges so a Grafana panel can render
"drift severity by dimension" and "fairness status by attribute" without the
backend needing to import the quality packages.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SEVERITY_GAUGE_RE = re.compile(r"^(low|moderate|high)$", re.IGNORECASE)


def export_drift_to_prometheus(
    report_path: str | Path,
    metrics: Any,
) -> int:
    """Update drift gauges from a ``drift_report.json``; returns dimension count."""
    data = _read_json(report_path)
    dimensions = [
        ("features", _feature_max_severity(data)),
        ("label", _dimension_severity(data, "label")),
        ("predictions", _dimension_severity(data, "predictions")),
        ("spatial", _dimension_severity(data, "spatial")),
        ("temporal", _dimension_severity(data, "temporal")),
    ]
    count = 0
    for dimension, severity in dimensions:
        if severity is None:
            continue
        metrics.set_drift(dimension, severity)
        count += 1
    return count


def export_fairness_to_prometheus(
    report_path: str | Path,
    metrics: Any,
) -> int:
    """Update fairness gauges from a ``fairness_report.json``; returns attribute count."""
    data = _read_json(report_path)
    count = 0
    for attribute in data.get("summary", {}).get("attributes", []):
        metrics.set_fairness(attribute, data.get("overall_status", "at_risk"))
        count += 1
    if "summary" not in data or not data["summary"].get("attributes"):
        metrics.set_fairness("all", data.get("overall_status", "at_risk"))
        count += 1
    for verdict in data.get("verdicts", []):
        if verdict.get("metric") == "disparate_impact":
            attribute = (data.get("summary", {}).get("attributes") or ["all"])[0]
            metrics.set_disparate_impact(attribute, verdict.get("value", 1.0))
    return count


def export_system_to_prometheus(
    metrics_snapshot: dict[str, Any],
    metrics: Any,
) -> None:
    """Synchronise a ``MetricsRegistry.snapshot()`` into the Prometheus registry."""
    by_path = metrics_snapshot.get("by_path", {})
    for path, entry in by_path.items():
        for _ in range(int(entry.get("requests", 0))):
            metrics.record_request(path, "GET", 200 if not entry.get("errors") else 500, 0.0)
    metrics.model_ready.set(1)


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _dimension_severity(data: dict[str, Any], key: str) -> str | None:
    dimension = data.get(key) or data.get(f"{key}s")
    if not dimension:
        return None
    severity = dimension.get("severity")
    return severity if SEVERITY_GAUGE_RE.match(str(severity)) else "low"


def _feature_max_severity(data: dict[str, Any]) -> str:
    severities = [f.get("severity", "low") for f in data.get("features", [])]
    rank = {"low": 0, "moderate": 1, "high": 2}
    if not severities:
        return "low"
    worst = max(severities, key=lambda s: rank.get(str(s), 0))
    return worst if rank.get(str(worst), 0) > 0 else "low"
