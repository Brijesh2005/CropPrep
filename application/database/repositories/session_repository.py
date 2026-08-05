"""User session repository."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from database.models.session import UserSession
from database.repositories.base import DataRepository


class UserSessionRepository(DataRepository[UserSession]):
    model = UserSession

    async def get_by_session_id(self, session_id: str) -> UserSession | None:
        result = await self.session.execute(
            select(UserSession).where(UserSession.session_id == session_id)
        )
        return result.scalar_one_or_none()

    async def get_by_refresh_token_id(self, refresh_token_id: int) -> UserSession | None:
        result = await self.session.execute(
            select(UserSession).where(UserSession.refresh_token_id == refresh_token_id)
        )
        return result.scalar_one_or_none()

    async def list_active(self, user_id: int) -> list[UserSession]:
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.is_active.is_(True),
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
            .order_by(UserSession.created_at.desc())
        )
        return list(result.scalars().all())

    async def touch(self, session: UserSession, *, ttl_hours: int) -> None:
        session.last_seen_at = datetime.now(timezone.utc)
        session.expires_at = session.last_seen_at + timedelta(hours=ttl_hours)
        await self.session.flush()

    async def revoke(self, session: UserSession) -> None:
        session.is_active = False
        session.revoked_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def revoke_all(self, user_id: int, *, except_session_id: str | None = None) -> int:
        now = datetime.now(timezone.utc)
        stmt = (
            select(UserSession)
            .where(
                UserSession.user_id == user_id,
                UserSession.is_active.is_(True),
                UserSession.revoked_at.is_(None),
            )
        )
        if except_session_id:
            stmt = stmt.where(UserSession.session_id != except_session_id)
        result = await self.session.execute(stmt)
        rows = list(result.scalars().all())
        for row in rows:
            row.is_active = False
            row.revoked_at = now
        await self.session.flush()
        return len(rows)

    async def count_active(self, user_id: int) -> int:
        return await self.count_rows(
            UserSession.user_id == user_id,
            UserSession.is_active.is_(True),
            UserSession.revoked_at.is_(None),
            UserSession.expires_at > datetime.now(timezone.utc),
        )
