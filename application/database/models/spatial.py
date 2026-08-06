"""Spatial models: administrative boundaries and point locations.

Geometry columns are portable binary in the ORM; the Alembic migration promotes
them to real PostGIS ``geometry`` columns (with GIST indexes) on PostgreSQL.
"""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from database.models.mixins import TimestampMixin, geometry_column


class AdministrativeBoundary(Base, TimestampMixin):
    """A self-referencing administrative unit (village / taluk / district)."""

    __tablename__ = "administrative_boundaries"

    id: Mapped[int] = mapped_column(primary_key=True)
    level: Mapped[str] = mapped_column(String(16), index=True)  # village|taluk|district
    code: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("administrative_boundaries.id", ondelete="CASCADE"), nullable=True, index=True
    )
    #: Centroid of the unit (geometry(Point, 4326) on PostgreSQL).
    centroid: Mapped[bytes | None] = geometry_column()
    #: Polygon geometry (geometry(MultiPolygon, 4326) on PostgreSQL).
    geometry: Mapped[bytes | None] = geometry_column()
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)


class SpatialLocation(Base, TimestampMixin):
    """A point location (dataset sample, user pin, or resolved coordinate)."""

    __tablename__ = "spatial_locations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    location_type: Mapped[str] = mapped_column(String(32), default="point", index=True)
    lon: Mapped[float] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    #: geometry(Point, 4326) on PostgreSQL.
    point: Mapped[bytes | None] = geometry_column()
    admin_boundary_id: Mapped[int | None] = mapped_column(
        ForeignKey("administrative_boundaries.id", ondelete="SET NULL"), nullable=True
    )
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
