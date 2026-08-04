"""GIS module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_model_container
from app.modules.gis.service import GISService


def get_gis_service(model_container: Any = Depends(get_model_container)) -> GISService:
    return model_container.resolve("gis_service")
