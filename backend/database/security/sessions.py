"""User session lifecycle service.

Persistence lives in :class:`~database.models.session.UserSession`; a Redis
cache entry mirrors each active session for fast validation and TTL refresh.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import SecuritySettings
from app.models.user import User
from database.models.session import UserSession
from database.repositories import UserSessionRepository
from database.services.redis_store import RedisStore


class SessionService:
    """Create / validate / revoke user sessions (DB + Redis)."""

    def __init__(
        self,
        repository: UserSessionRepository,
        store: RedisStore,
        settings: SecuritySettings,
    ) -> None:
        self._repo = repository
        self._store = store
        self._settings = settings

    @property
    def ttl_hours(self) -> int:
        return self._settings.session_ttl_hours

    def _cache_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def create(
        self,
        user: User,
        *,
        refresh_token_id: int | None = None,
        ip: str | None = None,
        user_agent: str | None = None,
    ) -> Any:
        session_id = secrets.token_urlsafe(24)
        now = datetime.now(timezone.utc)
        session = await self._repo.save(
            UserSession(
                user_id=user.id,
                session_id=session_id,
                refresh_token_id=refresh_token_id,
                ip_address=ip,
                user_agent=user_agent,
                created_at=now,
                expires_at=now + timedelta(hours=self.ttl_hours),
                last_seen_at=now,
                is_active=True,
            )
        )
        await self._cache(session_id, {"user_id": user.id, "revoked": False})
        return session

    async def _cache(self, session_id: str, payload: dict) -> None:
        await self._store.set(
            self._cache_key(session_id), payload, ttl=self.ttl_hours * 3600
        )

    async def get(self, session_id: str) -> Any | None:
        return await self._repo.get_by_session_id(session_id)

    async def touch(self, session_id: str) -> Any | None:
        session = await self._repo.get_by_session_id(session_id)
        if session is None or not session.is_active or session.revoked_at is not None:
            return None
        await self._repo.touch(session, ttl_hours=self.ttl_hours)
        await self._cache(session_id, {"user_id": session.user_id, "revoked": False})
        return session

    async def revoke(self, session_id: str) -> bool:
        session = await self._repo.get_by_session_id(session_id)
        if session is None:
            return False
        await self._repo.revoke(session)
        await self._store.delete(self._cache_key(session_id))
        return True

    async def revoke_all(self, user_id: int, *, except_session_id: str | None = None) -> int:
        count = await self._repo.revoke_all(user_id, except_session_id=except_session_id)
        await self._store.delete(self._cache_key(except_session_id)) if except_session_id else None
        return count

    async def list_active(self, user_id: int) -> list[Any]:
        return await self._repo.list_active(user_id)

    async def enforce_max_sessions(self, user_id: int, *, keep_session_id: str | None = None) -> None:
        """Revoke the oldest sessions when the user exceeds the configured cap."""
        max_sessions = self._settings.max_sessions_per_user
        active = await self._repo.list_active(user_id)
        excess = len(active) - max_sessions
        if excess <= 0:
            return
        for session in active[::-1]:
            if excess <= 0:
                break
            if keep_session_id is not None and session.session_id == keep_session_id:
                continue
            await self._repo.revoke(session)
            await self._store.delete(self._cache_key(session.session_id))
            excess -= 1
