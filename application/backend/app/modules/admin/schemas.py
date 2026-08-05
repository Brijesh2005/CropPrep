"""Admin module schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Dashboard(BaseModel):
    model_ready: bool = False
    model_version: str = ""
    device: str = ""
    prediction_count: int = 0
    users_count: int = 0
    dataset_ready: bool = False
    queue_size: int = 0


class Statistics(BaseModel):
    total_predictions: int = 0
    crop_distribution: dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0
    avg_inference_time_ms: float = 0.0
    fallback_count: int = 0


class RetrainResponse(BaseModel):
    message: str
    started: bool = True
