"""Auth module models.

The auth module operates on the shared :class:`app.models.user.User` model;
this file re-exports it so the module is self-describing and extractable.
"""

from __future__ import annotations

from app.models.user import User

__all__ = ["User"]
