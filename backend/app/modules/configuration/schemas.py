"""Configuration module schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ConfigurationResponse(BaseModel):
    data: dict[str, Any]
