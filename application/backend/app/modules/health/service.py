"""Health module service — liveness / readiness probes."""

from __future__ import annotations

from typing import Any

from app.modules.health.schemas import HealthResponse


class HealthService:
    """Reports liveness and readiness of the backend components."""

    def __init__(self, settings: Any, registry: Any, dataset_service: Any, database: Any) -> None:
        self._settings = settings
        self._registry = registry
        self._dataset = dataset_service
        self._database = database

    def live(self) -> HealthResponse:
        """Liveness — the process is up."""
        return HealthResponse(status="ok", version=self._settings.version)

    async def ready(self) -> HealthResponse:
        """Readiness — core dependencies are available."""
        checks = {
            "model": self._registry.is_ready(),
            "dataset": self._dataset is not None,
            "database": self._database is not None,
        }
        status = "ok" if all(checks.values()) else "degraded"
        return HealthResponse(status=status, version=self._settings.version, checks=checks)

    def full(self) -> HealthResponse:
        """Full health — component detail."""
        checks = {
            "model": self._registry.version_info(),
            "database_url": self._database.settings.url if self._database else None,
        }
        return HealthResponse(status="ok", version=self._settings.version, checks=checks)
