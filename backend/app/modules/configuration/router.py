"""Configuration routes: view the non-sensitive runtime configuration."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.core.security import ROLE_ADMIN
from app.dependencies.container import get_container
from app.dependencies.security import get_current_user_optional, require_role
from app.modules.configuration.service import ConfigurationService

router = APIRouter(prefix="/config", tags=["configuration"])


def _service(request: Request) -> ConfigurationService:
    container = get_container(request)
    return ConfigurationService(container.config.resolve("settings"))


@router.get("", summary="Public configuration")
async def public_config(request: Request, _: Any = Depends(get_current_user_optional)) -> dict:
    return _service(request).public()


@router.get("/full", summary="Full configuration (admin)")
async def full_config(request: Request, _: Any = Depends(require_role(ROLE_ADMIN))) -> dict:
    return _service(request).summary()
