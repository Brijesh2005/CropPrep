"""Auth module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.container import get_container
from app.dependencies.database import get_session
from app.modules.auth.service import AuthService


def get_auth_service(
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> AuthService:
    """Build an :class:`AuthService` bound to the request's session + cache."""
    settings = container.config.resolve("settings")
    cache = container.services.resolve("cache")
    return AuthService(session, settings.security, cache)
