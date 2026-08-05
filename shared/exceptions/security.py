"""Security and audit related shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class SecurityError(CropFusionError):
    """Base class for security-layer failures."""

    code = "CF-SEC-001"


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""

    code = "CF-SEC-002"


class AuthorizationError(SecurityError):
    """Raised when an authenticated principal lacks permission."""

    code = "CF-SEC-003"


class TokenError(SecurityError):
    """Raised when a token is invalid, expired or malformed."""

    code = "CF-SEC-004"
