"""GIS module schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LocationResponse(BaseModel):
    id: str
    lon: float
    lat: float
    name: str = ""
    admin: dict = Field(default_factory=dict)
    distance_km: float | None = None


class NearestRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-90, le=90)
    k: int = Field(default=1, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=255)
    limit: int = Field(default=10, ge=1, le=100)


class BoundaryResponse(BaseModel):
    name: str
    geometry_type: str
    bbox: list[float] = Field(default_factory=list)
    features: int = 0
