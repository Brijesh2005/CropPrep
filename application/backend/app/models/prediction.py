"""Prediction + ExplanationRecord ORM models.

NEW FILE — ``app/repositories/prediction.py`` and ``app/core/database.py``
(``create_all``) already import ``app.models.prediction.Prediction`` /
``ExplanationRecord``, but this module did not exist anywhere in the R5
scaffold. This fills that gap so ``/predict`` and ``/history`` can actually
persist and read prediction rows.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Prediction(Base):
    """One stored crop recommendation + yield prediction."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Location (resolved by LocationResolver at prediction time).
    location_lon: Mapped[float] = mapped_column(Float, nullable=False)
    location_lat: Mapped[float] = mapped_column(Float, nullable=False)
    location_name: Mapped[str | None] = mapped_column(String(255), nullable=True)  # village
    district: Mapped[str | None] = mapped_column(String(255), nullable=True)
    taluk: Mapped[str | None] = mapped_column(String(255), nullable=True)
    season: Mapped[str | None] = mapped_column(String(32), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Prediction outcome.
    crop: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    crop_probs: Mapped[dict] = mapped_column(JSON, default=dict)
    top3: Mapped[list] = mapped_column(JSON, default=list)
    yield_prediction: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Provenance.
    model_version: Mapped[str] = mapped_column(String(64), default="")
    dataset_version: Mapped[str] = mapped_column(String(64), default="")
    inference_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32), default="point")  # point | map
    fallback: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    explanation: Mapped["ExplanationRecord | None"] = relationship(
        back_populates="prediction", uselist=False, cascade="all, delete-orphan"
    )


class ExplanationRecord(Base):
    """Explainability summary attached to a prediction (feature importance etc.)."""

    __tablename__ = "explanation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("predictions.id"), nullable=False, unique=True, index=True
    )
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    prediction: Mapped[Prediction] = relationship(back_populates="explanation")


__all__ = ["ExplanationRecord", "Prediction"]
