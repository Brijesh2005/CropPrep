"""Inference module exceptions."""

from __future__ import annotations

from app.core.exceptions import InferenceError, PredictionError

__all__ = ["InferenceError", "PredictionError"]


class ModelNotReadyError(InferenceError):
    code = "B-INFER-100"
