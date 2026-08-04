"""Administrative boundary seeding.

Districts come from the real ICRISAT district-level dataset
(``Tabular_Datasets/ICRISAT-District Level Data.csv``). Taluks and villages are
synthetic but deterministic (nested inside each district), mirroring the sample
locations used elsewhere in the project (e.g. Moodabidri / Bantwal / Sullia in
Karnataka). Each unit stores a pseudo bounding box in ``properties`` so the
SQLite fallback point-in-boundary resolution works without PostGIS.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from database.repositories import AdministrativeBoundaryRepository

ICRISAT_PATH = Path(
    r"D:\CropPrep\Tabular_Datasets\ICRISAT-District Level Data.csv"
)

#: (taluk, village...) pairs synthesised under every district.
SYNTHETIC_CHILDREN = [
    ("Taluk A", ["Village A1", "Village A2", "Village A3"]),
    ("Taluk B", ["Village B1", "Village B2"]),
]


class BoundarySeeder:
    """Seed districts (real) + taluks/villages (synthetic)."""

    def __init__(self, session: AsyncSession, *, csv_path: Path | None = None) -> None:
        self._session = session
        self._repo = AdministrativeBoundaryRepository(session)
        self._csv_path = csv_path or ICRISAT_PATH

    async def seed(self) -> None:
        districts = self._load_districts()
        if await self._repo.count_rows(
            self._repo.model.level == "district"
        ) > 0:
            return
        for index, (state, name, code) in enumerate(districts):
            props = self._bbox_for(index, len(districts))
            props["state"] = state
            district = await self._repo.create_boundary(
                level="district", name=name, code=str(code), properties=props
            )
            for taluk_name, villages in SYNTHETIC_CHILDREN:
                taluk = await self._repo.create_boundary(
                    level="taluk", name=f"{name} {taluk_name}", parent_id=district.id,
                    properties=self._child_bbox(props, hashlib.md5(taluk_name.encode(), usedforsecurity=False).hexdigest()),
                )
                for village_name in villages:
                    await self._repo.create_boundary(
                        level="village",
                        name=f"{name} {village_name}",
                        parent_id=taluk.id,
                        properties=self._child_bbox(
                            props, hashlib.md5(village_name.encode(), usedforsecurity=False).hexdigest()
                        ),
                    )
        await self._session.commit()

    async def boundary_counts(self) -> dict[str, int]:
        return await self._repo.count_by_level()

    def _load_districts(self) -> list[tuple[str, str, int]]:
        if not self._csv_path.exists():
            return [("Karnataka", "Mangalore", 1)]
        try:
            import pandas as pd  # type: ignore[import-not-found]

            df = pd.read_csv(
                self._csv_path, usecols=["State Name", "Dist Name", "Dist Code"]
            ).drop_duplicates(subset=["Dist Code"])
            return [
                (str(row["State Name"]), str(row["Dist Name"]), int(row["Dist Code"]))
                for row in df.to_dict("records")
            ]
        except Exception:  # pragma: no cover - fallback for odd environments
            return [("Karnataka", "Mangalore", 1)]

    @staticmethod
    def _bbox_for(index: int, total: int) -> dict:
        """Deterministic pseudo bounding box spread across India's extent."""
        lon = 68.5 + ((index * 37) % 100) / 100.0 * 28.5
        lat = 8.0 + ((index * 53) % 100) / 100.0 * 27.0
        return {
            "bbox_min_lon": round(lon - 0.5, 4),
            "bbox_min_lat": round(lat - 0.4, 4),
            "bbox_max_lon": round(lon + 0.5, 4),
            "bbox_max_lat": round(lat + 0.4, 4),
            "centroid_lon": round(lon, 4),
            "centroid_lat": round(lat, 4),
        }

    @staticmethod
    def _child_bbox(parent: dict, seed: str) -> dict:
        factor = int(seed[:4], 16) / 0xFFFF
        dx = (factor - 0.5) * 0.4
        dy = ((int(seed[4:8], 16) / 0xFFFF) - 0.5) * 0.3
        lon = parent["centroid_lon"] + dx
        lat = parent["centroid_lat"] + dy
        size = 0.15
        return {
            "bbox_min_lon": round(lon - size, 4),
            "bbox_min_lat": round(lat - size, 4),
            "bbox_max_lon": round(lon + size, 4),
            "bbox_max_lat": round(lat + size, 4),
            "centroid_lon": round(lon, 4),
            "centroid_lat": round(lat, 4),
        }
