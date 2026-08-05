"""Health module schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = ""
    checks: dict[str, Any] = Field(default_factory=dict)
