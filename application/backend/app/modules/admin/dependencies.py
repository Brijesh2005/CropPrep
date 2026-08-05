"""Admin module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies.container import get_model_container
from app.dependencies.database import get_session
from app.modules.admin.service import AdminService


def get_admin_service(
    session: AsyncSession = Depends(get_session),
    model_container: Any = Depends(get_model_container),
) -> AdminService:
    return AdminService(
        session,
        model_container.resolve("inference_engine"),
        model_container.resolve("model_registry"),
        model_container.resolve("dataset_service"),
    )
