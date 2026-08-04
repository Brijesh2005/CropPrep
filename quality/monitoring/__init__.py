"""Monitoring assets: Prometheus exporters, Grafana dashboards, HTML dashboard.

The production-grade Prometheus metrics registry itself lives in the backend
(``app.services.prometheus``) so it is scraped from ``/metrics`` at runtime.
This package supplies the ML-QA exporters that feed drift / fairness verdicts
into that registry, plus visualisation dashboards.
"""

from __future__ import annotations

from .dashboard import render_performance_dashboard
from .exporters import (
    export_drift_to_prometheus,
    export_fairness_to_prometheus,
    export_system_to_prometheus,
)

__all__ = [
    "render_performance_dashboard",
    "export_drift_to_prometheus",
    "export_fairness_to_prometheus",
    "export_system_to_prometheus",
]
