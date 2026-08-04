"""Predictions module schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    year: int | None = Field(default=None, ge=2000, le=2100)
    season: str | None = Field(default=None, max_length=32)
    include_explanation: bool = False


class MapPredictionRequest(BaseModel):
    points: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class PredictionResponse(BaseModel):
    prediction_id: int | None = None
    location: dict = Field(default_factory=dict)
    coordinates: dict = Field(default_factory=dict)
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    crop_probs: dict = Field(default_factory=dict)
    model_version: str = ""
    inference_time_ms: float = 0.0
    explanation_summary: dict | None = None
    fallback: bool = False


class HistoryItem(BaseModel):
    prediction_id: int
    location: dict = Field(default_factory=dict)
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    model_version: str = ""
    created_at: datetime | None = None
