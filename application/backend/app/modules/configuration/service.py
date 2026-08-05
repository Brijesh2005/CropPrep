"""Configuration module service — safe, non-secret config view."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings

_SECRET_KEYS = {"secret_key", "password", "token", "redis_url"}


class ConfigurationService:
    """Exposes a non-sensitive subset of the runtime configuration."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def summary(self) -> dict[str, Any]:
        data = self._settings.model_dump()
        return {k: v for k, v in data.items() if not any(
            secret in k.lower() for secret in _SECRET_KEYS
        )}

    def public(self) -> dict[str, Any]:
        return {
            "app_name": self._settings.app_name,
            "environment": self._settings.environment,
            "version": self._settings.version,
            "api_prefix": self._settings.api_prefix,
            "application_mode": self._settings.application_mode,
            "database": {"url": self._settings.database.url},
            "inference": self._settings.inference.model_dump(),
        }
