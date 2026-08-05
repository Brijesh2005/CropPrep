"""Configuration module repository (config lives in memory)."""

from __future__ import annotations

from typing import Any


class ConfigurationRepository:
    """Reads the runtime configuration (non-secret)."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    def get(self) -> dict[str, Any]:
        return self._settings.model_dump()
