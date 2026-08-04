"""Role and permission repositories (RBAC)."""

from __future__ import annotations

from sqlalchemy import select

from database.models.access import Permission, Role
from database.repositories.base import DataRepository


class RoleRepository(DataRepository[Role]):
    model = Role

    async def get_by_name(self, name: str) -> Role | None:
        result = await self.session.execute(select(Role).where(Role.name == name))
        return result.scalar_one_or_none()

    async def list_with_permissions(self) -> list[Role]:
        result = await self.session.execute(select(Role).order_by(Role.id))
        return list(result.scalars().all())

    async def grant_permissions(self, role: Role, permissions: list[Permission]) -> Role:
        existing = {p.id for p in role.permissions}
        for permission in permissions:
            if permission.id not in existing:
                role.permissions.append(permission)
        await self.session.flush()
        return role

    async def revoke_permission(self, role: Role, permission: Permission) -> Role:
        role.permissions = [p for p in role.permissions if p.id != permission.id]
        await self.session.flush()
        return role


class PermissionRepository(DataRepository[Permission]):
    model = Permission

    async def get_by_code(self, code: str) -> Permission | None:
        result = await self.session.execute(select(Permission).where(Permission.code == code))
        return result.scalar_one_or_none()

    async def get_by_codes(self, codes: list[str]) -> list[Permission]:
        result = await self.session.execute(select(Permission).where(Permission.code.in_(codes)))
        return list(result.scalars().all())

    async def list_by_resource(self, resource: str | None = None) -> list[Permission]:
        stmt = select(Permission).order_by(Permission.resource, Permission.action)
        if resource:
            stmt = stmt.where(Permission.resource == resource)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
