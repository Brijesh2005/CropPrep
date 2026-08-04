"""App configuration service (versioned key/value store)."""

from __future__ import annotations

from typing import Any

from database.repositories import AppConfigurationRepository


class ConfigService:
    """Manage the versioned application configuration key/value store."""

    def __init__(self, repository: AppConfigurationRepository) -> None:
        self._repo = repository

    async def get(self, key: str) -> dict[str, Any] | None:
        config = await self._repo.get_by_key(key)
        if config is None:
            return None
        return {"key": config.key, "value": config.value, "category": config.category}

    async def set(
        self, *, key: str, value: dict | None, category: str | None = None,
        description: str | None = None, is_secret: bool = False,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        config = await self._repo.set_value(
            key=key, value=value, category=category, description=description,
            is_secret=is_secret, updated_by=updated_by,
        )
        await self._repo.commit()
        return {"key": config.key, "value": config.value, "version": config.version}

    async def list(self, *, category: str | None = None, include_secrets: bool = False) -> dict[str, Any]:
        rows = await self._repo.list_by_category(category)
        return {
            "items": [
                {
                    "key": c.key,
                    "value": c.value if include_secrets or not c.is_secret else None,
                    "category": c.category,
                    "version": c.version,
                    "is_secret": c.is_secret,
                    "description": c.description,
                }
                for c in rows
            ]
        }
