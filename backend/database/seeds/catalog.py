"""Crop and season catalog seeding (idempotent)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from database.models.catalog import Crop, Season
from database.repositories import CropRepository, SeasonRepository


CROPS = [
    ("rice", "Rice", "Oryza sativa", "cereal", "Primary cereal crop across South Asia."),
    ("wheat", "Wheat", "Triticum aestivum", "cereal", "Staple rabi cereal."),
    ("maize", "Maize", "Zea mays", "cereal", "Versatile grain and fodder crop."),
    ("sorghum", "Sorghum", "Sorghum bicolor", "cereal", "Drought-tolerant kharif grain."),
    ("pearl_millet", "Pearl Millet", "Cenchrus americanus", "millet", "Hardy kharif millet."),
    ("finger_millet", "Finger Millet", "Eleusine coracana", "millet", "Nutritious small millet."),
    ("chickpea", "Chickpea", "Cicer arietinum", "pulse", "Major rabi pulse."),
    ("pigeonpea", "Pigeonpea", "Cajanus cajan", "pulse", "Kharif pulse intercropped with cereals."),
    ("groundnut", "Groundnut", "Arachis hypogaea", "oilseed", "Kharif oilseed."),
    ("soybean", "Soybean", "Glycine max", "oilseed", "Kharif oilseed and protein crop."),
    ("cotton", "Cotton", "Gossypium hirsutum", "fibre", "Major kharif fibre crop."),
    ("sugarcane", "Sugarcane", "Saccharum officinarum", "cash", "Long-duration cash crop."),
    ("potato", "Potato", "Solanum tuberosum", "tuber", "Cool-season tuber crop."),
    ("sunflower", "Sunflower", "Helianthus annuus", "oilseed", "Rabi/summer oilseed."),
]

SEASONS = [
    ("kharif", "Kharif", "South-West monsoon season (June-Oct).", 6, 10, "South Asia"),
    ("rabi", "Rabi", "North-East monsoon / winter season (Oct-Mar).", 10, 3, "South Asia"),
    ("zaid", "Zaid", "Summer season (Mar-Jun).", 3, 6, "South Asia"),
]


class CatalogSeeder:
    """Seed the crop and season catalogs."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._crops = CropRepository(session)
        self._seasons = SeasonRepository(session)

    async def seed(self) -> None:
        for code, name, scientific, category, description in CROPS:
            if await self._crops.get_by_code(code) is None:
                await self._crops.add(
                    Crop(
                        code=code, name=name, scientific_name=scientific,
                        category=category, description=description, is_active=True,
                    )
                )
        for code, name, description, start, end, region in SEASONS:
            if await self._seasons.get_by_code(code) is None:
                await self._seasons.add(
                    Season(
                        code=code, name=name, start_month=start, end_month=end,
                        region=region, description=description, is_active=True,
                    )
                )
        await self._session.commit()
