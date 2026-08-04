"""Users module exceptions."""

from __future__ import annotations

from app.core.exceptions import NotFoundError

__all__ = ["UserNotFoundError"]


class UserNotFoundError(NotFoundError):
    code = "B-USERS-100"
