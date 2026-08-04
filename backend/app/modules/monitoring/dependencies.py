"""Monitoring module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_container
from app.modules.monitoring.service import MonitoringService


def get_monitoring_service(container: Any = Depends(get_container)) -> MonitoringService:
    return container.services.resolve("monitoring_service")
