"""Predictions module schemas. REPLACES app/modules/predictions/schemas.py."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    include_explanation: bool = False


class MapPredictionRequest(BaseModel):
    points: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class CropCandidate(BaseModel):
    crop: str
    probability: float


class PredictionResponse(BaseModel):
    prediction_id: int | None = None
    village: str = ""
    district: str = ""
    taluk: str | None = None
    coordinates: dict = Field(default_factory=dict)  # {"lon": ..., "lat": ...}
    season: str = ""
    year: int | None = None
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    crop_probs: dict[str, float] = Field(default_factory=dict)
    top3: list[CropCandidate] = Field(default_factory=list)
    model_version: str = ""
    dataset_version: str = ""
    inference_time_ms: float = 0.0
    feature_importance: dict[str, float] | None = None
    explanation_summary: dict | None = None
    fallback: bool = False


class HistoryItem(BaseModel):
    prediction_id: int
    village: str = ""
    district: str = ""
    coordinates: dict = Field(default_factory=dict)
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    model_version: str = ""
    created_at: datetime | None = None
