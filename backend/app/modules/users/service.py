"""Users module service."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import SecuritySettings
from app.core.security import hash_password
from app.modules.users.exceptions import UserNotFoundError
from app.modules.users.models import User
from app.modules.users.repository import UserRepository


class UserService:
    """Read / update user profiles."""

    def __init__(self, session: AsyncSession, security_settings: SecuritySettings) -> None:
        self._repository = UserRepository(session)
        self._security = security_settings

    async def get(self, user_id: int) -> User:
        user = await self._repository.get_by_id(user_id)
        if user is None:
            raise UserNotFoundError("user not found", detail=user_id)
        return user

    async def update_profile(
        self, user_id: int, *, full_name: str | None = None, password: str | None = None
    ) -> User:
        user = await self.get(user_id)
        if full_name is not None:
            user.full_name = full_name
        if password is not None:
            user.hashed_password = hash_password(
                password, self._security.password_scheme
            )
        await self._repository.commit()
        await self._repository.refresh(user)
        return user
