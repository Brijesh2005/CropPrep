"""Registry and versioning related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class RegistryError(CropFusionError):
    """Raised when a registry cannot fulfil a request."""

    code = "CF-REGISTRY-001"


class VersionError(CropFusionError):
    """Raised when a semantic version is invalid or a version operation fails."""

    code = "CF-VERSION-001"


class InvalidVersionError(VersionError):
    """Raised when a version string is not valid semantic versioning."""

    code = "CF-VERSION-002"


class VersionNotFoundError(VersionError):
    """Raised when a requested version does not exist."""

    code = "CF-VERSION-003"
