"""Common response schemas."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageResponse(BaseModel):
    """A simple message envelope."""

    message: str


class ErrorResponse(BaseModel):
    """Structured error envelope."""

    error: dict[str, Any] = Field(
        default_factory=lambda: {"code": "B-ERROR", "message": "error", "detail": None}
    )


class Page(BaseModel, Generic[T]):
    """Generic paginated result."""

    items: list[T] = Field(default_factory=list)
    total: int = 0
    limit: int = 0
    offset: int = 0
