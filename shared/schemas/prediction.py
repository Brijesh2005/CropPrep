"""Prediction request / result schemas (NOT the API request models)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..enums import CropType, Season


@dataclass(slots=True)
class PredictionInputSchema:
    """Canonical input contract for a yield prediction.

    Note: the Application platform's FastAPI request models live in
    ``application/backend/app/schemas`` and are intentionally NOT moved here.
    """

    location: str
    crop: CropType = CropType.UNKNOWN
    season: Season = Season.UNKNOWN
    year: int | None = None
    boundary_geojson: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "crop": self.crop.value,
            "season": self.season.value,
            "year": self.year,
            "boundary_geojson": self.boundary_geojson,
            "extra": self.extra,
        }


@dataclass(slots=True)
class PredictionResultSchema:
    """Canonical output contract for a yield prediction."""

    location: str
    predicted_yield: float
    unit: str = "t/ha"
    confidence: float | None = None
    model_version: str = "0.0.0"
    dataset_version: str = "0.0.0"
    season: Season = Season.UNKNOWN
    crop: CropType = CropType.UNKNOWN
    predicted_at: datetime = field(default_factory=datetime.now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "predicted_yield": self.predicted_yield,
            "unit": self.unit,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "season": self.season.value,
            "crop": self.crop.value,
            "predicted_at": self.predicted_at.isoformat(),
            "extra": self.extra,
        }
