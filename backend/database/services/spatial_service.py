"""Spatial service: coordinate validation, boundary resolution and caching."""

from __future__ import annotations

from typing import Any

from database.models.spatial import AdministrativeBoundary, SpatialLocation
from database.repositories import AdministrativeBoundaryRepository, SpatialLocationRepository
from database.services.geo import bbox_from_center, validate_coordinates
from database.services.redis_store import RedisStore


class SpatialService:
    """Geo operations backed by the spatial repositories (+ Redis cache)."""

    def __init__(
        self,
        locations: SpatialLocationRepository,
        boundaries: AdministrativeBoundaryRepository,
        store: RedisStore,
    ) -> None:
        self._locations = locations
        self._boundaries = boundaries
        self._store = store

    # ------------------------------------------------------------------ #
    # Boundary resolution
    # ------------------------------------------------------------------ #
    async def resolve_admin_region(
        self, lon: float, lat: float, *, cache: bool = True
    ) -> dict[str, Any] | None:
        lon, lat = validate_coordinates(lon, lat)
        cache_key = f"geo:resolve:{lon:.5f}:{lat:.5f}"
        if cache:
            cached = await self._store.get(cache_key)
            if cached is not None:
                return cached
        boundary = await self._boundaries.resolve(lon, lat)
        if boundary is None:
            return None
        chain = await self._boundary_chain(boundary)
        result = {
            "village": chain.get("village"),
            "taluk": chain.get("taluk"),
            "district": chain.get("district"),
            "boundary_id": boundary.id,
            "boundary_code": boundary.code,
            "boundary_name": boundary.name,
        }
        if cache:
            await self._store.set(cache_key, result, ttl=3600)
        return result

    async def _boundary_chain(self, boundary: AdministrativeBoundary) -> dict[str, str | None]:
        chain: dict[str, str | None] = {"village": None, "taluk": None, "district": None}
        current = boundary
        while current is not None:
            chain[current.level] = current.name
            if current.parent_id is None:
                break
            current = await self._boundaries.get(current.parent_id)
        return chain

    # ------------------------------------------------------------------ #
    # Locations
    # ------------------------------------------------------------------ #
    async def nearest_locations(
        self, lon: float, lat: float, *, limit: int = 10, radius_km: float = 50.0
    ) -> list[dict[str, Any]]:
        lon, lat = validate_coordinates(lon, lat)
        rows = await self._locations.nearest(lon, lat, limit=limit, radius_km=radius_km)
        return [
            {
                "id": r["id"], "name": r["name"], "location_type": r["location_type"],
                "lon": r["lon"], "lat": r["lat"], "distance_km": round(r.get("distance_km", 0), 3),
            }
            for r in rows
        ]

    async def locations_in_radius(self, lon: float, lat: float, radius_km: float) -> list[dict[str, Any]]:
        lon, lat = validate_coordinates(lon, lat)
        min_lon, min_lat, max_lon, max_lat = bbox_from_center(lon, lat, radius_km)
        rows = await self._locations.within_bbox(min_lon, min_lat, max_lon, max_lat)
        return [
            {"id": loc.id, "name": loc.name, "lon": loc.lon, "lat": loc.lat}
            for loc in rows
        ]

    async def create_location(
        self, *, name: str, lon: float, lat: float, location_type: str = "point",
        properties: dict | None = None, source: str | None = None,
    ) -> dict[str, Any]:
        lon, lat = validate_coordinates(lon, lat)
        boundary = await self._boundaries.resolve(lon, lat)
        loc = await self._locations.save(
            SpatialLocation(
                name=name, lon=lon, lat=lat, location_type=location_type,
                properties=properties or {}, source=source,
                admin_boundary_id=boundary.id if boundary else None,
                is_active=True,
            )
        )
        return {"id": loc.id, "name": loc.name, "lon": loc.lon, "lat": loc.lat}

    # ------------------------------------------------------------------ #
    # Boundaries
    # ------------------------------------------------------------------ #
    async def list_boundaries(self, level: str, *, parent_id: int | None = None) -> list[dict[str, Any]]:
        rows = await self._boundaries.list_by_level(level, parent_id=parent_id)
        return [
            {
                "id": b.id, "level": b.level, "code": b.code, "name": b.name,
                "parent_id": b.parent_id,
            }
            for b in rows
        ]

    async def boundary_counts(self) -> dict[str, int]:
        return await self._boundaries.count_by_level()
