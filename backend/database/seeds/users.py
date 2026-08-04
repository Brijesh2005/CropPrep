"""User seeding: super admin + demo accounts (idempotent)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from database.repositories import UserRepository
from database.security import PasswordService


class UserSeeder:
    """Create the bootstrap admin and demo users."""

    def __init__(self, session: AsyncSession, password_service: PasswordService) -> None:
        self._session = session
        self._users = UserRepository(session)
        self._passwords = password_service

    async def seed(self) -> None:
        await self._ensure_user(
            email="admin@cropfusion.local",
            password="AdminPass123!",
            full_name="CropFusion Super Admin",
            role="super_admin",
        )
        await self._ensure_user(
            email="dataset@cropfusion.local",
            password="DatasetPass123!",
            full_name="Dataset Manager",
            role="dataset_manager",
        )
        await self._ensure_user(
            email="farmer@cropfusion.local",
            password="FarmerPass123!",
            full_name="Demo Farmer",
            role="user",
        )
        await self._ensure_user(
            email="researcher@cropfusion.local",
            password="Researcher123!",
            full_name="Demo Researcher",
            role="analyst",
        )

    async def _ensure_user(
        self, *, email: str, password: str, full_name: str, role: str
    ) -> None:
        existing = await self._users.get_by_email(email)
        if existing is not None:
            return
        await self._users.add(
            User(
                email=email,
                hashed_password=self._passwords.hash(password),
                full_name=full_name,
                role=role,
                is_active=True,
                is_email_verified=True,
            )
        )
        await self._session.commit()
