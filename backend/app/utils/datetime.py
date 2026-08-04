"""Date/time helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """The current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)
