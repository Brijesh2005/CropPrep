"""Role and permission seeding (idempotent)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from database.models.access import Permission, Role
from database.policies.catalog import PERMISSIONS, ROLE_PERMISSIONS, SYSTEM_ROLES
from database.repositories import PermissionRepository, RoleRepository


class RoleSeeder:
    """Create the system roles, permissions and default grants."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._roles = RoleRepository(session)
        self._permissions = PermissionRepository(session)

    async def seed(self) -> None:
        perms: dict[str, Permission] = {}
        for code, resource, action, name, description in PERMISSIONS:
            existing = await self._permissions.get_by_code(code)
            if existing is None:
                existing = await self._permissions.add(
                    Permission(
                        code=code, resource=resource, action=action,
                        name=name, description=description,
                    )
                )
            perms[code] = existing

        for role_name, display in SYSTEM_ROLES.items():
            granted = [
                perms[code]
                for code in ROLE_PERMISSIONS.get(role_name, set())
                if code in perms
            ]
            role = await self._roles.get_by_name(role_name)
            if role is None:
                role = await self._roles.add(
                    Role(
                        name=role_name, description=f"{display} (system role)",
                        is_system=True, priority=_priority(role_name),
                        permissions=granted,
                    )
                )
            else:
                merged = {p.id: p for p in role.permissions}
                merged.update({p.id: p for p in granted})
                role.permissions = list(merged.values())
        await self._session.commit()


def _priority(role_name: str) -> int:
    from database.policies.catalog import ROLE_PRIORITY

    return ROLE_PRIORITY.get(role_name, 0)
