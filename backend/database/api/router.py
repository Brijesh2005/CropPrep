"""Phase 10 enterprise API router — aggregates every enterprise router."""

from __future__ import annotations

from fastapi import APIRouter

from database.api.admin import router as admin_router
from database.api.auth import router as auth_router
from database.api.catalog import router as catalog_router
from database.api.config import router as config_router
from database.api.experiments import router as experiments_router
from database.api.feedback import router as feedback_router
from database.api.notifications import router as notifications_router
from database.api.predictions import router as predictions_router
from database.api.registry import router as registry_router
from database.api.spatial import router as spatial_router
from database.api.users import router as users_router


def build_enterprise_router() -> APIRouter:
    """Assemble the Phase 10 enterprise router."""
    api = APIRouter()
    api.include_router(auth_router)
    api.include_router(users_router)
    api.include_router(predictions_router)
    api.include_router(notifications_router)
    api.include_router(feedback_router)
    api.include_router(admin_router)
    api.include_router(registry_router)
    api.include_router(catalog_router)
    api.include_router(spatial_router)
    api.include_router(experiments_router)
    api.include_router(config_router)
    return api
