"""Predictions module exceptions."""

from __future__ import annotations

from app.core.exceptions import PredictionError

__all__ = ["PredictionError"]


class LocationNotPredictableError(PredictionError):
    code = "B-PREDICT-100"
