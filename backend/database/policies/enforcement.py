"""Permission enforcement.

Primary source of truth is the database role→permission grants; when the roles
table has not been seeded the static catalog (with role hierarchy) is used so
the application works before migrations/seeds run.
"""

from __future__ import annotations

from app.core.exceptions import AuthorizationError
from database.policies.catalog import ROLE_PERMISSIONS, ROLE_PRIORITY
from database.repositories import RoleRepository


class PermissionEnforcer:
    """Check a user role against the permission catalog."""

    def __init__(self, role_repository: RoleRepository | None = None) -> None:
        self._roles = role_repository
        self._db_cache: dict[str, set[str]] = {}

    async def permissions_for_role(self, role_name: str) -> set[str]:
        cached = self._db_cache.get(role_name)
        if cached is not None:
            return cached
        granted: set[str] = set()
        if self._roles is not None:
            role = await self._roles.get_by_name(role_name)
            if role is not None and role.permissions:
                granted = {p.code for p in role.permissions}
        if not granted:
            granted = ROLE_PERMISSIONS.get(role_name, set()) or self._inherited(role_name)
        self._db_cache[role_name] = granted
        return granted

    def _inherited(self, role_name: str) -> set[str]:
        """Fallback: inherit permissions from the next-lower role in the hierarchy."""
        granted: set[str] = set()
        order = sorted(ROLE_PRIORITY.items(), key=lambda item: item[1])
        for name, _priority in order:
            if ROLE_PRIORITY.get(name, 0) > ROLE_PRIORITY.get(role_name, 0):
                break
            granted |= ROLE_PERMISSIONS.get(name, set())
        return granted

    async def has_permission(self, role_name: str, code: str) -> bool:
        return code in await self.permissions_for_role(role_name)

    async def require(self, role_name: str, code: str) -> None:
        """Raise :class:`AuthorizationError` unless the role has the permission."""
        if not await self.has_permission(role_name, code):
            raise AuthorizationError(
                f"permission required: {code}", detail={"role": role_name, "permission": code}
            )
