"""Inference module dependencies."""

from __future__ import annotations

from typing import Any

from fastapi import Depends

from app.dependencies.container import get_model_container


def get_inference_engine(model_container: Any = Depends(get_model_container)) -> Any:
    return model_container.resolve("inference_engine")


def get_model_registry(model_container: Any = Depends(get_model_container)) -> Any:
    return model_container.resolve("model_registry")
