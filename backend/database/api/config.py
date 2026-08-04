"""Versioned application configuration key/value store.

Complement the Phase 8 ``/config`` routes (read-only runtime configuration)
with a writable, versioned configuration store.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.core.security import ROLE_ADMIN
from app.dependencies.enterprise import get_config_service
from app.dependencies.security import get_current_user_optional, require_role
from database.api.schemas import ConfigSetRequest
from database.services.config_service import ConfigService

router = APIRouter(prefix="/config-store", tags=["config-store"])


@router.get(
    "/{key}",
    summary="Read a configuration value",
)
async def get_config(
    key: str,
    _: Any = Depends(get_current_user_optional),
    service: ConfigService = Depends(get_config_service),
) -> dict[str, Any] | None:
    return await service.get(key)


@router.put(
    "",
    summary="Set a configuration value (admin)",
    description="Upsert a versioned configuration key/value pair.",
)
async def set_config(
    body: ConfigSetRequest,
    user: Any = Depends(require_role(ROLE_ADMIN)),
    service: ConfigService = Depends(get_config_service),
) -> dict[str, Any]:
    return await service.set(
        key=body.key,
        value=body.value,
        category=body.category,
        description=body.description,
        is_secret=body.is_secret,
        updated_by=user.id,
    )


@router.get(
    "",
    summary="List configuration entries",
)
async def list_config(
    category: str | None = Query(default=None),
    include_secrets: bool = Query(default=False),
    user: Any = Depends(require_role(ROLE_ADMIN)),
    service: ConfigService = Depends(get_config_service),
) -> dict[str, Any]:
    return await service.list(category=category, include_secrets=include_secrets)
