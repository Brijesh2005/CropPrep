"""Model and dataset registries."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from database.models.mixins import TimestampMixin


class ModelVersion(Base, TimestampMixin):
    """Registry entry for a trained model checkpoint (Phase 5/6)."""

    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    training_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    checkpoint_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("dataset_versions.id", ondelete="SET NULL"), nullable=True
    )
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), default="draft")  # draft|active|archived
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class DatasetVersion(Base, TimestampMixin):
    """Registry entry for an ingested dataset version (Phase 2 manager)."""

    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_dataset_name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    source: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column(JSON, default=dict)
    #: pending | validating | valid | invalid
    validation_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    checksum_algorithm: Mapped[str | None] = mapped_column(String(16), nullable=True)
    downloaded_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    file_count: Mapped[int | None] = mapped_column(default=None)
    size_bytes: Mapped[int | None] = mapped_column(default=None)
    schema: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
