"""Spatial repositories: administrative boundaries and point locations.

PostGIS-backed queries run when the dialect is PostgreSQL; otherwise a
deterministic fallback (haversine distance, bounding-box containment) keeps the
test-suite and local development functional on SQLite.
"""

from __future__ import annotations

import math

from sqlalchemy import and_, func, select, text

from database.models.spatial import AdministrativeBoundary, SpatialLocation
from database.repositories.base import DataRepository


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometres."""
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _point_wkt(lon: float, lat: float) -> str:
    return f"POINT({lon} {lat})"


class SpatialLocationRepository(DataRepository[SpatialLocation]):
    """Point locations with nearest-neighbour and bbox queries."""

    model = SpatialLocation

    async def nearest(self, lon: float, lat: float, *, limit: int = 10, radius_km: float | None = None) -> list[dict]:
        if self._is_postgres:
            sql = text(
                """
                SELECT id, name, location_type, lon, lat, properties, source, is_active,
                       ST_Distance(point, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)) / 1000.0 AS distance_km
                FROM spatial_locations
                WHERE is_active = true
                  AND ST_DWithin(point, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :radius_m)
                ORDER BY point <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
                LIMIT :limit
                """
            )
            radius_m = (radius_km or 50.0) * 1000.0
            rows = (await self.session.execute(sql, {"lon": lon, "lat": lat, "radius_m": radius_m, "limit": limit})).mappings().all()
            return [dict(r) for r in rows]

        result = await self.session.execute(
            select(SpatialLocation).where(SpatialLocation.is_active.is_(True))
        )
        items = [
            {"id": loc.id, "name": loc.name, "location_type": loc.location_type,
             "lon": loc.lon, "lat": loc.lat, "properties": loc.properties,
             "source": loc.source, "is_active": loc.is_active}
            for loc in result.scalars().all()
        ]
        for item in items:
            item["distance_km"] = haversine_km(lon, lat, item["lon"], item["lat"])
        items = [item for item in items if radius_km is None or item["distance_km"] <= radius_km]
        items.sort(key=lambda item: item["distance_km"])
        return items[:limit]

    async def within_bbox(
        self, min_lon: float, min_lat: float, max_lon: float, max_lat: float
    ) -> list[SpatialLocation]:
        result = await self.session.execute(
            select(SpatialLocation)
            .where(
                SpatialLocation.is_active.is_(True),
                SpatialLocation.lon >= min_lon,
                SpatialLocation.lon <= max_lon,
                SpatialLocation.lat >= min_lat,
                SpatialLocation.lat <= max_lat,
            )
            .order_by(SpatialLocation.id)
        )
        return list(result.scalars().all())

    async def create_point(self, *, name: str, lon: float, lat: float, location_type: str = "point", properties: dict | None = None, source: str | None = None) -> SpatialLocation:
        point = None
        if self._is_postgres:
            point = await self._st_point(lon, lat)
        return await self.save(
            SpatialLocation(
                name=name, lon=lon, lat=lat, location_type=location_type,
                point=point, properties=properties or {}, source=source,
            )
        )

    async def _st_point(self, lon: float, lat: float) -> bytes | None:
        row = (await self.session.execute(
            text("SELECT ST_AsBinary(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))"),
            {"lon": lon, "lat": lat},
        )).scalar()
        return bytes(row) if row is not None else None

    @property
    def _is_postgres(self) -> bool:
        dialect = self.session.bind.dialect if self.session.bind is not None else None
        return dialect is not None and dialect.name == "postgresql"


class AdministrativeBoundaryRepository(DataRepository[AdministrativeBoundary]):
    """Administrative units with point-in-polygon resolution."""

    model = AdministrativeBoundary

    async def resolve(self, lon: float, lat: float) -> AdministrativeBoundary | None:
        """Return the most specific boundary containing the coordinate."""
        if self._is_postgres:
            sql = text(
                """
                SELECT * FROM administrative_boundaries
                WHERE is_active = true
                  AND ST_Contains(geometry, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))
                ORDER BY CASE level WHEN 'village' THEN 1 WHEN 'taluk' THEN 2 WHEN 'district' THEN 3 ELSE 4 END
                LIMIT 1
                """
            )
            row = (await self.session.execute(sql, {"lon": lon, "lat": lat})).mappings().first()
            if row is None:
                return None
            return await self.get(int(row["id"]))

        districts = await self.list_by_level("district")
        district = self._containing(districts, lon, lat)
        if district is None:
            return None
        taluks = await self.list_by_level("taluk", parent_id=district.id)
        taluk = self._containing(taluks, lon, lat) or district
        villages = await self.list_by_level("village", parent_id=taluk.id)
        village = self._containing(villages, lon, lat) or taluk
        return village

    def _containing(self, boundaries: list[AdministrativeBoundary], lon: float, lat: float) -> AdministrativeBoundary | None:
        """Fallback containment using the centroid bounding box."""
        best = None
        best_area = None
        for boundary in boundaries:
            props = boundary.properties or {}
            min_lon = props.get("bbox_min_lon")
            min_lat = props.get("bbox_min_lat")
            max_lon = props.get("bbox_max_lon")
            max_lat = props.get("bbox_max_lat")
            if None in (min_lon, min_lat, max_lon, max_lat):
                continue
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                area = (max_lon - min_lon) * (max_lat - min_lat)
                if best_area is None or area < best_area:
                    best, best_area = boundary, area
        return best

    async def list_by_level(self, level: str, *, parent_id: int | None = None) -> list[AdministrativeBoundary]:
        stmt = select(AdministrativeBoundary).where(
            AdministrativeBoundary.level == level, AdministrativeBoundary.is_active.is_(True)
        )
        if parent_id is not None:
            stmt = stmt.where(AdministrativeBoundary.parent_id == parent_id)
        result = await self.session.execute(stmt.order_by(AdministrativeBoundary.name))
        return list(result.scalars().all())

    async def get_by_code(self, code: str) -> AdministrativeBoundary | None:
        result = await self.session.execute(
            select(AdministrativeBoundary).where(AdministrativeBoundary.code == code)
        )
        return result.scalar_one_or_none()

    async def count_by_level(self) -> dict[str, int]:
        result = await self.session.execute(
            select(AdministrativeBoundary.level, func.count(AdministrativeBoundary.id))
            .group_by(AdministrativeBoundary.level)
        )
        return {level: int(count) for level, count in result.all()}

    async def create_boundary(
        self,
        *,
        level: str,
        name: str,
        code: str | None = None,
        parent_id: int | None = None,
        properties: dict | None = None,
    ) -> AdministrativeBoundary:
        centroid = geometry_bytes = None
        if self._is_postgres and properties:
            if "centroid_lon" in properties and "centroid_lat" in properties:
                row = (await self.session.execute(
                    text("SELECT ST_AsBinary(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326))"),
                    {"lon": properties["centroid_lon"], "lat": properties["centroid_lat"]},
                )).scalar()
                centroid = bytes(row) if row is not None else None
        return await self.save(
            AdministrativeBoundary(
                level=level, code=code, name=name, parent_id=parent_id,
                centroid=centroid, geometry=geometry_bytes, properties=properties or {},
                source="seed",
            )
        )

    @property
    def _is_postgres(self) -> bool:
        dialect = self.session.bind.dialect if self.session.bind is not None else None
        return dialect is not None and dialect.name == "postgresql"
