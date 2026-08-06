"""Prediction metadata: flexible per-prediction context snapshots."""

from __future__ import annotations

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from database.models.mixins import TimestampMixin


class PredictionMetadata(Base, TimestampMixin):
    """1:1 with a prediction; stores the inputs and context snapshot used."""

    __tablename__ = "prediction_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("predictions.id", ondelete="CASCADE"), unique=True, index=True
    )
    #: The exact request inputs used for the prediction.
    inputs: Mapped[dict] = mapped_column(JSON, default=dict)
    #: Snapshot of the observation / sample resolved by STAM.
    feature_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    weather: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    soil: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    client_context: Mapped[dict | None] = mapped_column(JSON, nullable=True)
