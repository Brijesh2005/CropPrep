"""Admin module exceptions."""

from __future__ import annotations

from app.core.exceptions import AuthorizationError, ServiceUnavailableError

__all__ = ["AuthorizationError", "ServiceUnavailableError"]


class RetrainingUnavailableError(ServiceUnavailableError):
    code = "B-ADMIN-100"
