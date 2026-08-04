"""History module schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    prediction_id: int
    location: dict = Field(default_factory=dict)
    recommended_crop: str
    expected_yield: float | None = None
    confidence: float = 0.0
    model_version: str = ""
    created_at: datetime | None = None


class HistoryPage(BaseModel):
    items: list[HistoryItem] = Field(default_factory=list)
    total: int = 0
    limit: int = 0
    offset: int = 0
