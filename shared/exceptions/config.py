"""Configuration related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class ConfigurationError(CropFusionError):
    """Raised when configuration is malformed, unknown or out of range."""

    code = "CF-CONFIG-001"


class InvalidEnvironmentError(ConfigurationError):
    """Raised when the runtime environment (env name) is unknown."""

    code = "CF-CONFIG-002"


class MissingSettingError(ConfigurationError):
    """Raised when a required configuration setting is absent."""

    code = "CF-CONFIG-003"


class SettingsTemplateError(ConfigurationError):
    """Raised when a settings template cannot be written or loaded."""

    code = "CF-CONFIG-004"
