"""Application-level dependencies (container, database, security)."""

from __future__ import annotations

from app.dependencies.container import get_container
from app.dependencies.database import get_repository, get_session
from app.dependencies.security import (
    get_current_user,
    get_current_user_optional,
    oauth2_scheme,
    require_role,
)

__all__ = [
    "get_container",
    "get_session",
    "get_repository",
    "get_current_user",
    "get_current_user_optional",
    "require_role",
    "oauth2_scheme",
]
