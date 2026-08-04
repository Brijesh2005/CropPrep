"""GIS module exceptions."""

from __future__ import annotations

from app.core.exceptions import GISError, NotFoundError

__all__ = ["GISError", "NotFoundError"]


class LocationNotFoundError(NotFoundError):
    code = "B-GIS-100"
