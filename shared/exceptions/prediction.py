"""Prediction platform specific shared exceptions."""

from __future__ import annotations

from .base import CropFusionError


class PredictionError(CropFusionError):
    """Raised when a prediction request cannot be fulfilled."""

    code = "CF-PRED-001"


class PredictionInputError(PredictionError):
    """Raised when prediction input is missing, malformed or out of range."""

    code = "CF-PRED-002"


class ModelNotLoadedError(PredictionError):
    """Raised when a requested model is not loaded in the serving runtime."""

    code = "CF-PRED-003"
