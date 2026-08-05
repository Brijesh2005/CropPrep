"""Prediction history DTOs (architecture contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class HistoryFilters:
    """Filter set for searching stored predictions (mirrors the API's search)."""

    user_id: int | None = None
    crop: str | None = None
    season: str | None = None
    year: int | None = None
    district: str | None = None
    taluk: str | None = None
    village: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_confidence: float | None = None
    limit: int = 50
    offset: int = 0


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    """A stored prediction row as exposed to callers."""

    prediction_id: int
    location: dict[str, Any] = field(default_factory=dict)
    recommended_crop: str = ""
    expected_yield: float | None = None
    confidence: float = 0.0
    model_version: str = ""
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """Paged history result."""

    items: list[HistoryRecord] = field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


__all__ = ["HistoryFilters", "HistoryPage", "HistoryRecord"]
