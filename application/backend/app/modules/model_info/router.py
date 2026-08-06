"""NEW module: GET /model — model + dataset version, readiness, eval metrics.

Wiring (2-line addition to the existing app/api/router.py, not a full
replacement):

    from app.modules.model_info.router import router as model_info_router
    ...
    api.include_router(model_info_router)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.dependencies.container import get_model_container

router = APIRouter(prefix="/model", tags=["model"])


@router.get("", summary="Model + dataset version, readiness, evaluation metrics")
async def model_info(model_container: Any = Depends(get_model_container)) -> dict[str, Any]:
    registry = model_container.resolve("model_registry")
    info = registry.version_info()
    package = getattr(registry, "package", None)
    return {
        **info,
        "manifest": package.manifest if package else None,
        "model_config": package.model_config if package else None,
    }
