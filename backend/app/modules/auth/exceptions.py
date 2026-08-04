"""Auth module exceptions."""

from __future__ import annotations

from app.core.exceptions import AuthenticationError, ConflictError, TokenError

__all__ = ["AuthenticationError", "ConflictError", "TokenError"]


class EmailAlreadyRegisteredError(ConflictError):
    code = "B-AUTH-100"
