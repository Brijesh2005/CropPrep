"""Monitoring module service — performance metrics."""

from __future__ import annotations

from typing import Any

from app.services.metrics import MetricsRegistry


class MonitoringService:
    """Collects and exposes runtime performance metrics."""

    def __init__(self, metrics: MetricsRegistry) -> None:
        self._metrics = metrics

    def metrics(self) -> dict[str, Any]:
        return self._metrics.snapshot()
