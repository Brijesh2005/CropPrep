"""Catalog service: crops and seasons."""

from __future__ import annotations

from typing import Any

from database.models.catalog import Crop, Season
from database.repositories import CropRepository, SeasonRepository


class CatalogService:
    """Manage the crop and season catalogs."""

    def __init__(self, crops: CropRepository, seasons: SeasonRepository) -> None:
        self._crops = crops
        self._seasons = seasons

    # ------------------------------------------------------------------ #
    # Crops
    # ------------------------------------------------------------------ #
    async def create_crop(
        self, *, code: str, name: str, scientific_name: str | None = None,
        category: str | None = None, description: str | None = None,
        metadata_: dict | None = None,
    ) -> Crop:
        if await self._crops.get_by_code(code) is not None:
            raise ValueError(f"crop code already exists: {code}")
        return await self._crops.save(
            Crop(
                code=code, name=name, scientific_name=scientific_name,
                category=category, description=description,
                metadata_=metadata_ or {}, is_active=True,
            )
        )

    async def list_crops(self, *, search: str | None = None) -> dict[str, Any]:
        rows = await self._crops.search(search) if search else await self._crops.list_active()
        return {
            "items": [
                {
                    "id": c.id, "code": c.code, "name": c.name,
                    "scientific_name": c.scientific_name, "category": c.category,
                    "description": c.description, "metadata": c.metadata_,
                }
                for c in rows
            ]
        }

    # ------------------------------------------------------------------ #
    # Seasons
    # ------------------------------------------------------------------ #
    async def create_season(
        self, *, code: str, name: str, start_month: int | None = None,
        end_month: int | None = None, region: str | None = None,
        description: str | None = None, metadata_: dict | None = None,
    ) -> Season:
        if await self._seasons.get_by_code(code) is not None:
            raise ValueError(f"season code already exists: {code}")
        return await self._seasons.save(
            Season(
                code=code, name=name, start_month=start_month, end_month=end_month,
                region=region, description=description,
                metadata_=metadata_ or {}, is_active=True,
            )
        )

    async def list_seasons(self) -> dict[str, Any]:
        rows = await self._seasons.list_active()
        return {
            "items": [
                {
                    "id": s.id, "code": s.code, "name": s.name,
                    "start_month": s.start_month, "end_month": s.end_month,
                    "region": s.region, "description": s.description,
                    "metadata": s.metadata_,
                }
                for s in rows
            ]
        }
