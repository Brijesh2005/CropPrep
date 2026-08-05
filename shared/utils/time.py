"""Time / date helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone


def utc_now() -> datetime:
    """Current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return utc_now().isoformat()


def to_iso(value: datetime | date) -> str:
    """Render a date/datetime as an ISO-8601 string."""
    return value.isoformat()


def parse_iso(value: str) -> datetime | None:
    """Parse an ISO-8601 string into a datetime, or None on failure."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
