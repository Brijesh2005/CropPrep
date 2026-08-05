"""Idempotent seed runner."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from database.seeds.boundaries import BoundarySeeder
from database.seeds.catalog import CatalogSeeder
from database.seeds.roles import RoleSeeder
from database.seeds.users import UserSeeder
from database.security import PasswordService


class SeedRunner:
    """Orchestrates the bootstrap seeds (roles, users, catalogs, boundaries)."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        password_service: PasswordService | None = None,
        include_boundaries: bool = True,
        csv_path: str | None = None,
    ) -> None:
        self._session = session
        self._password_service = password_service or PasswordService()
        self._include_boundaries = include_boundaries
        self._csv_path = csv_path

    async def run(self) -> dict[str, int]:
        from database.policies.catalog import PERMISSIONS, SYSTEM_ROLES

        await RoleSeeder(self._session).seed()
        await UserSeeder(self._session, self._password_service).seed()
        await CatalogSeeder(self._session).seed()
        boundary_count = 0
        if self._include_boundaries:
            boundaries = BoundarySeeder(
                self._session,
                csv_path=Path(self._csv_path) if self._csv_path else None,
            )
            await boundaries.seed()
            counts = await boundaries.boundary_counts()
            boundary_count = sum(counts.values())
        return {
            "roles": len(SYSTEM_ROLES),
            "permissions": len(PERMISSIONS),
            "boundaries": boundary_count,
        }


async def seed_database(
    session: AsyncSession,
    *,
    include_boundaries: bool = True,
    csv_path: str | None = None,
    password_service: PasswordService | None = None,
) -> dict[str, int]:
    """Convenience entry point used by the app startup hook and CLI."""
    runner = SeedRunner(
        session,
        password_service=password_service,
        include_boundaries=include_boundaries,
        csv_path=csv_path,
    )
    return await runner.run()
