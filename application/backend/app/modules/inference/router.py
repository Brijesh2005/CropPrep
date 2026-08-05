"""Inference routes: engine status / warmup (internal service)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import ROLE_ADMIN
from app.dependencies.security import get_current_user_optional, require_role
from app.modules.inference.dependencies import get_inference_engine, get_model_registry
from app.modules.inference.schemas import InferenceStatus

router = APIRouter(prefix="/inference", tags=["inference"])


@router.get(
    "/status",
    response_model=InferenceStatus,
    summary="Inference engine status",
    description="Model readiness, version, queue and cache state.",
)
async def inference_status(
    _: Any = Depends(get_current_user_optional),
    engine: Any = Depends(get_inference_engine),
) -> InferenceStatus:
    return InferenceStatus(**engine.status())


@router.post(
    "/warmup",
    summary="Warm up the inference engine",
    description="Run a single forward pass to pre-allocate the model (admin).",
)
async def inference_warmup(
    _: Any = Depends(require_role(ROLE_ADMIN)),
    registry: Any = Depends(get_model_registry),
):
    registry.warmup()
    return {"message": "model warmed up", "version": registry.version}
