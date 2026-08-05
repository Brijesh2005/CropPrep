"""Inference DTOs (architecture contract, R1.4).

These models describe the *future* inference-only surface of the Prediction
Platform. R1.4 deliberately does NOT implement inference, load models or touch
backend endpoints; it only fixes the contracts that a later phase will bind the
real implementation to.

The key interface decision: a prediction is requested with **latitude and
longitude only**. Date, season and historical context are auto-resolved by the
GIS layer (see ``application/gis``) instead of being supplied by the client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from shared.enums import Season


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    """A location-only inference request.

    No ``year`` / ``season`` fields: the platform resolves them automatically
    from the current date and the location's historical context. Mirrors the
    farmer-mode ``POST /predict`` interface (``application_mode == "farmer"``).
    """

    lon: float
    lat: float
    include_explanation: bool = False

    def __post_init__(self) -> None:
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon out of range [-180, 180]: {self.lon}")
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat out of range [-90, 90]: {self.lat}")


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Everything the inference engine needs beyond lon/lat, auto-resolved."""

    #: Geographic / administrative context (village, taluk, district).
    location: dict[str, Any] = field(default_factory=dict)
    #: Resolved cropping season for the request date.
    season: Season = Season.UNKNOWN
    #: Year the prediction targets (resolved from the request date).
    year: int | None = None
    #: Calendar date used to resolve the season.
    request_date: date = field(default_factory=date.today)
    #: Pointer into the inference package's historical context store.
    historical_context: dict[str, Any] = field(default_factory=dict)
    #: Free-form extras (client id, source, …).
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "season": self.season.value,
            "year": self.year,
            "request_date": self.request_date.isoformat(),
            "historical_context": self.historical_context,
            "extra": self.extra,
        }


@dataclass(slots=True)
class PredictionResult:
    """Canonical inference result produced by a future engine."""

    recommended_crop: str
    crop_probs: dict[str, float] = field(default_factory=dict)
    expected_yield: float | None = None
    confidence: float = 0.0
    model_version: str = ""
    dataset_version: str = ""
    inference_time_ms: float = 0.0
    fallback: bool = False
    explanation_summary: dict[str, Any] | None = None
    context: PredictionContext | None = None
    predicted_at: datetime = field(default_factory=datetime.utcnow)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_crop": self.recommended_crop,
            "crop_probs": self.crop_probs,
            "expected_yield": self.expected_yield,
            "confidence": self.confidence,
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "inference_time_ms": self.inference_time_ms,
            "fallback": self.fallback,
            "explanation_summary": self.explanation_summary,
            "context": self.context.to_dict() if self.context else None,
            "predicted_at": self.predicted_at.isoformat(),
            "extra": self.extra,
        }


__all__ = [
    "PredictionContext",
    "PredictionRequest",
    "PredictionResult",
]
