"""Monitoring module exceptions."""

from __future__ import annotations

from app.core.exceptions import ServiceUnavailableError

__all__ = ["ServiceUnavailableError"]


class MetricsUnavailableError(ServiceUnavailableError):
    code = "B-MONITOR-100"
