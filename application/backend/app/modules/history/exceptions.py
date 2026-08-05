"""History module exceptions."""

from __future__ import annotations

from app.core.exceptions import NotFoundError

__all__ = ["NotFoundError"]


class HistoryNotFoundError(NotFoundError):
    code = "B-HISTORY-100"
