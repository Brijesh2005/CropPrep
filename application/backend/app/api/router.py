"""API router — aggregates every module's router under the API prefix.

REPLACES application/backend/app/api/router.py (only change from the R5
version: model_info_router is imported and included, for GET /model).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.modules.admin.router import router as admin_router
from app.modules.auth.router import router as auth_router
from app.modules.configuration.router import router as config_router
from app.modules.dataset.router import router as dataset_router
from app.modules.explainability.router import router as explainability_router
from app.modules.gis.router import router as gis_router
from app.modules.history.router import router as history_router
from app.modules.inference.router import router as inference_router
from app.modules.model_info.router import router as model_info_router
from app.modules.monitoring.router import router as monitoring_router
from app.modules.predictions.router import router as predictions_router
from app.modules.users.router import router as users_router
from database.api import build_enterprise_router


def build_api_router(prefix: str) -> APIRouter:
    """Assemble the root API router with every module attached."""
    api = APIRouter(prefix=prefix)
    api.include_router(auth_router)
    api.include_router(users_router)
    api.include_router(predictions_router)
    api.include_router(dataset_router)
    api.include_router(gis_router)
    api.include_router(explainability_router)
    api.include_router(history_router)
    api.include_router(admin_router)
    api.include_router(inference_router)
    api.include_router(model_info_router)
    api.include_router(config_router)
    api.include_router(monitoring_router)
    api.include_router(build_enterprise_router())
    return api
