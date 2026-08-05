"""Users module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.container import get_container
from app.dependencies.database import get_session
from app.modules.users.service import UserService


def get_user_service(
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> UserService:
    settings = container.config.resolve("settings")
    return UserService(session, settings.security)
