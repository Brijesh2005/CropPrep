"""Shared model mixins and column helpers for the enterprise data layer.

* :class:`TimestampMixin` — ``created_at`` / ``updated_at`` columns.
* :class:`UUIDPKMixin` — UUID primary key (native on PostgreSQL, CHAR(32) elsewhere).
* :func:`geometry_column` — a portable geometry column: a binary column in the ORM
  that is promoted to a real PostGIS ``geometry`` type by the Alembic migration on
  PostgreSQL (GeoAlchemy2's SQLite DDL events require Spatialite, so the ORM keeps
  a portable binary representation and all spatial predicates run through
  dialect-guarded SQL in the spatial repositories).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, LargeBinary
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Uuid


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=True
    )


class UUIDPKMixin:
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


class UUIDRefMixin:
    """A unique UUID business key on an integer-PK table."""

    ref_uuid: Mapped[uuid.UUID] = mapped_column(Uuid, default=uuid.uuid4, unique=True)


def geometry_column() -> Mapped[bytes | None]:
    """Declare a portable geometry column (see module docstring)."""
    return mapped_column(LargeBinary, nullable=True)
