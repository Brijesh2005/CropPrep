"""Monitoring module schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class MetricsResponse(BaseModel):
    metrics: dict[str, Any]
