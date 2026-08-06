"""User profile models: preferences and saved locations."""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from database.models.mixins import TimestampMixin, geometry_column


class UserPreference(Base, TimestampMixin):
    """Per-user preferences (language, theme, notifications, metadata)."""

    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    preferred_language: Mapped[str] = mapped_column(String(16), default="en")
    theme: Mapped[str] = mapped_column(String(16), default="system")  # light|dark|system
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    notifications: Mapped[dict] = mapped_column(JSON, default=dict)
    #: Flexible preference bag (units, alert thresholds, ...).
    metadata_: Mapped[dict] = mapped_column(JSON, default=dict)


class UserLocation(Base, TimestampMixin):
    """A named, saved geographic location belonging to a user."""

    __tablename__ = "user_locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    point: Mapped[bytes | None] = geometry_column()
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
