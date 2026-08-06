"""Inference module schemas. REPLACES app/modules/inference/schemas.py.

Farmer mode is location-only per the R6 spec: no year/season fields exist on
the request at all (previously they were optional; now they're removed so
the API contract itself enforces "farmer only selects location").
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="Longitude (WGS84)")
    lat: float = Field(..., ge=-90, le=90, description="Latitude (WGS84)")
    include_explanation: bool = Field(default=False)


class MapPredictionRequest(BaseModel):
    points: list[PredictionRequest] = Field(..., min_length=1, max_length=500)


class CropCandidate(BaseModel):
    crop: str
    probability: float


class LocationInfo(BaseModel):
    village: str = ""
    district: str = ""
    taluk: str | None = None
    lon: float
    lat: float
    season: str = ""
    year: int | None = None


class PredictionResponse(BaseModel):
    prediction_id: int | None = None
    location: LocationInfo
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    crop_probs: dict[str, float] = Field(default_factory=dict)
    top3: list[CropCandidate] = Field(default_factory=list)
    model_version: str
    dataset_version: str = ""
    inference_time_ms: float = 0.0
    explanation_summary: dict | None = None
    fallback: bool = False


class InferenceStatus(BaseModel):
    ready: bool = False
    model_version: str = ""
    dataset_version: str = ""
    queue_size: int = 0
    cache_enabled: bool = True
    device: str = "cpu"
