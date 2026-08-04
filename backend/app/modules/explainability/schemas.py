"""Explainability module schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    year: int | None = Field(default=None, ge=2000, le=2100)
    season: str | None = Field(default=None, max_length=32)


class ExplanationResponse(BaseModel):
    observation_id: str = ""
    crop: str = ""
    crop_probs: dict = Field(default_factory=dict)
    yield_prediction: float | None = None
    confidence: dict = Field(default_factory=dict)
    top_features: list[list] = Field(default_factory=list)
    important_dates: list[str] = Field(default_factory=list)
    modality_gates: dict = Field(default_factory=dict)
    reasoning: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)
