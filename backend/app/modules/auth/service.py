"""Auth service — registration, login, logout, token refresh."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import SecuritySettings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.modules.auth.exceptions import AuthenticationError, EmailAlreadyRegisteredError
from app.modules.auth.models import User
from app.modules.auth.repository import UserRepository
from app.services.cache import Cache


class AuthService:
    """Handles user registration and token issuance."""

    def __init__(
        self,
        session: AsyncSession,
        security_settings: SecuritySettings,
        cache: Cache | None = None,
    ) -> None:
        self._repository = UserRepository(session)
        self._security = security_settings
        self._cache = cache

    async def register(
        self, *, email: str, password: str, full_name: str | None
    ) -> User:
        email = email.strip().lower()
        if await self._repository.exists(email):
            raise EmailAlreadyRegisteredError("email is already registered", detail=email)
        user = User(
            email=email,
            hashed_password=hash_password(password, self._security.password_scheme),
            full_name=full_name,
            role="user",
        )
        await self._repository.add(user)
        await self._repository.commit()
        await self._repository.refresh(user)
        return user

    async def login(self, *, email: str, password: str) -> dict[str, Any]:
        user = await self._repository.get_by_email(email.strip().lower())
        if user is None or not verify_password(
            password, user.hashed_password, self._security.password_scheme
        ):
            raise AuthenticationError("invalid email or password")
        if not user.is_active:
            raise AuthenticationError("user account is disabled")
        return self._issue_tokens(user)

    async def refresh(self, refresh_token: str) -> dict[str, Any]:
        payload = decode_token(refresh_token, self._security, expected="refresh")
        user = await self._repository.get_by_id(int(payload["sub"]))
        if user is None or not user.is_active:
            raise AuthenticationError("user no longer active")
        return self._issue_tokens(user)

    async def logout(self, access_token: str) -> None:
        # Best-effort revocation via the cache blacklist (stateless JWT).
        if self._cache is not None:
            payload = decode_token(access_token, self._security, expected="access")
            jti = payload.get("jti", access_token)
            await self._cache.set(f"revoked:{jti}", True, ttl=3600)

    def _issue_tokens(self, user: User) -> dict[str, Any]:
        access = create_access_token(
            str(user.id), role=user.role, settings=self._security
        )
        refresh = create_refresh_token(
            str(user.id), role=user.role, settings=self._security
        )
        return {
            "access_token": access,
            "refresh_token": refresh,
            "token_type": "bearer",
            "expires_in": self._security.access_token_expire_minutes * 60,
            "user": user.to_dict(),
        }
