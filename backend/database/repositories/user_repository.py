"""User repository (extends the Phase 8 user repo)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from app.models.user import User
from app.repositories.user import UserRepository as BaseUserRepository
from database.repositories.base import DataRepository


class UserRepository(BaseUserRepository, DataRepository[User]):
    """Enterprise user repository with auth/profile helpers."""

    model = User

    async def get_by_email(self, email: str) -> User | None:
        normalized = email.strip().lower()
        result = await self.session.execute(select(User).where(User.email == normalized))
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone: str) -> User | None:
        result = await self.session.execute(select(User).where(User.phone == phone))
        return result.scalar_one_or_none()

    async def exists(self, email: str) -> bool:
        return (await self.get_by_email(email)) is not None

    async def record_login_success(self, user: User, ip: str | None = None) -> None:
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def record_login_failure(self, user: User) -> int:
        """Increment the failed counter; return the new counter value."""
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        await self.session.flush()
        return user.failed_login_attempts

    async def lock_account(self, user: User, until: datetime) -> None:
        user.locked_until = until
        await self.session.flush()

    async def mark_email_verified(self, user: User) -> None:
        user.is_email_verified = True
        user.email_verified_at = datetime.now(timezone.utc)
        await self.session.flush()

    async def set_password(self, user: User, hashed: str, *, must_change: bool = False) -> None:
        user.hashed_password = hashed
        user.must_change_password = must_change
        user.failed_login_attempts = 0
        user.locked_until = None
        await self.session.flush()

    async def update_profile(
        self, user: User, *, full_name: str | None = None, phone: str | None = None
    ) -> User:
        if full_name is not None:
            user.full_name = full_name
        if phone is not None:
            user.phone = phone
        await self.session.flush()
        return user
