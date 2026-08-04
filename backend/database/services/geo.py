"""Geo helpers: coordinate validation and distance math."""

from __future__ import annotations

import math

from app.core.exceptions import ValidationError


def validate_coordinates(lon: float, lat: float) -> tuple[float, float]:
    """Validate lon/lat ranges and raise a typed error on invalid input."""
    if not (-180.0 <= lon <= 180.0):
        raise ValidationError(f"longitude out of range: {lon}")
    if not (-90.0 <= lat <= 90.0):
        raise ValidationError(f"latitude out of range: {lat}")
    return float(lon), float(lat)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def bbox_from_center(lon: float, lat: float, radius_km: float) -> tuple[float, float, float, float]:
    """Approximate bounding box around a point for a radius in km."""
    lat_deg = radius_km / 111.320
    lon_deg = radius_km / (111.320 * max(math.cos(math.radians(lat)), 0.01))
    return (lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg)
