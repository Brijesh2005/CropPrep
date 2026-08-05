"""Security dependencies: current user, RBAC."""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.exceptions import AuthenticationError, AuthorizationError, TokenError
from app.dependencies.container import get_container
from app.dependencies.database import get_session
from app.repositories.user import UserRepository

#: OAuth2 password-flow bearer scheme (token URL = login endpoint).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def _resolve_current_user(
    token: str | None, session: AsyncSession, container: Any
) -> Any:
    """Decode the access token and load the matching active user."""
    settings = container.config.resolve("settings")
    if not token:
        raise AuthenticationError("missing bearer token")
    payload = security.decode_token(token, settings.security, expected="access")
    repository = UserRepository(session)
    user = await repository.get_by_id(int(payload["sub"]))
    if user is None:
        raise TokenError("user no longer exists")
    if not user.is_active:
        raise AuthorizationError("user account is disabled")
    return user


async def get_current_user(
    token: str | None = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> Any:
    """Resolve the authenticated user from a Bearer access token."""
    return await _resolve_current_user(token, session, container)


async def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme),
    session: AsyncSession = Depends(get_session),
    container: Any = Depends(get_container),
) -> Any | None:
    """Resolve the current user, or ``None`` for anonymous requests."""
    if not token:
        return None
    try:
        return await _resolve_current_user(token, session, container)
    except (AuthenticationError, TokenError, AuthorizationError):
        return None


def require_role(role: str):
    """Build a dependency enforcing that the current user holds ``role``."""

    async def _dependency(user: Any = Depends(get_current_user)) -> Any:
        if not security.RBAC.can(role, getattr(user, "role", None)):
            raise AuthorizationError(
                f"requires role {role!r} or higher", detail=getattr(user, "role", None)
            )
        return user

    return _dependency
