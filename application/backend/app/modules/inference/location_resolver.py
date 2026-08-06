"""Location Resolver + Historical Context Lookup (pipeline steps 1-2).

    Location -> Location Resolver -> Historical Context Lookup -> Feature Builder

Reads only ``location_index.parquet``, ``village_metadata.parquet`` and
``historical_context.parquet`` from the loaded release package. Does not
touch the live GIS module, shapefiles, or the Dataset Manager — those are the
GIS layer's concern (map UI, reverse geocoding for display). This resolver is
the inference-time equivalent: fast, nearest-neighbour, in-memory.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from inference_package.release import ReleasePackage


class LocationNotServedError(RuntimeError):
    """Raised when a lon/lat falls outside every known location's service radius."""


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    village: str
    district: str
    taluk: str | None
    lon: float
    lat: float
    distance_km: float
    season: str
    year: int
    historical_context: dict[str, Any] = field(default_factory=dict)
    village_metadata: dict[str, Any] = field(default_factory=dict)


class LocationResolver:
    """Nearest-village lookup + season/year + historical-context resolution."""

    #: Beyond this distance a location is considered "not served" by the
    #: exported package (avoids silently predicting for unrelated regions).
    MAX_SERVICE_RADIUS_KM = 50.0

    def __init__(self, package: ReleasePackage) -> None:
        self._package = package
        self._locations = self._prepare_location_index(package.location_index)
        self._village_meta = package.village_metadata
        self._history = package.historical_context

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def resolve(self, lon: float, lat: float, *, at: _dt.date | None = None) -> ResolvedLocation:
        at = at or _dt.date.today()
        idx, distance_km = self._nearest(lon, lat)
        if distance_km > self.MAX_SERVICE_RADIUS_KM:
            raise LocationNotServedError(
                f"({lon}, {lat}) is {distance_km:.1f} km from the nearest known "
                f"location, outside the {self.MAX_SERVICE_RADIUS_KM} km service radius"
            )
        row = self._locations.iloc[idx]
        village = str(row["village"])
        district = str(row.get("district", ""))
        taluk = row.get("taluk")
        taluk = str(taluk) if pd.notna(taluk) else None

        season = self._resolve_season(at)
        year = at.year

        village_meta = self._lookup_village_metadata(village, district)
        history = self._lookup_historical_context(village, district, season, year)

        return ResolvedLocation(
            village=village,
            district=district,
            taluk=taluk,
            lon=float(row["lon"]),
            lat=float(row["lat"]),
            distance_km=distance_km,
            season=season,
            year=year,
            historical_context=history,
            village_metadata=village_meta,
        )

    # ------------------------------------------------------------------ #
    # Season resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_season(at: _dt.date) -> str:
        """India cropping-season calendar: Kharif (Jun-Oct), Rabi (Nov-Mar), Zaid (Apr-May)."""
        month = at.month
        if month in (6, 7, 8, 9, 10):
            return "Kharif"
        if month in (11, 12, 1, 2, 3):
            return "Rabi"
        return "Zaid"

    # ------------------------------------------------------------------ #
    # Nearest-neighbour lookup
    # ------------------------------------------------------------------ #

    def _prepare_location_index(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"lon", "lat", "village"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"location_index.parquet missing columns: {missing}")
        return df.reset_index(drop=True)

    def _nearest(self, lon: float, lat: float) -> tuple[int, float]:
        lons = self._locations["lon"].to_numpy(dtype="float64")
        lats = self._locations["lat"].to_numpy(dtype="float64")
        distances = self._haversine_km(lon, lat, lons, lats)
        idx = int(np.argmin(distances))
        return idx, float(distances[idx])

    @staticmethod
    def _haversine_km(lon0: float, lat0: float, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        r = 6371.0088
        phi0, phi1 = np.radians(lat0), np.radians(lats)
        d_phi = np.radians(lats - lat0)
        d_lambda = np.radians(lons - lon0)
        a = np.sin(d_phi / 2) ** 2 + np.cos(phi0) * np.cos(phi1) * np.sin(d_lambda / 2) ** 2
        return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    # ------------------------------------------------------------------ #
    # Metadata / historical context lookups
    # ------------------------------------------------------------------ #

    def _lookup_village_metadata(self, village: str, district: str) -> dict[str, Any]:
        df = self._village_meta
        if df.empty:
            return {}
        mask = (df.get("village") == village) & (df.get("district") == district)
        matches = df[mask]
        if matches.empty:
            matches = df[df.get("village") == village]
        if matches.empty:
            return {}
        return _row_to_dict(matches.iloc[0])

    def _lookup_historical_context(
        self, village: str, district: str, season: str, year: int
    ) -> dict[str, Any]:
        df = self._history
        if df.empty:
            return {}
        mask = (df.get("village") == village) & (df.get("district") == district)
        subset = df[mask]
        if subset.empty:
            subset = df[df.get("village") == village]
        if subset.empty:
            return {}
        if "season" in subset.columns:
            season_subset = subset[subset["season"] == season]
            if not season_subset.empty:
                subset = season_subset
        if "year" in subset.columns:
            # Prefer the most recent year at or before the target year (climatology
            # fallback: nearest available year) rather than the exact year, since
            # the farmer never supplies one.
            subset = subset.assign(_year_gap=(subset["year"] - year).abs())
            subset = subset.sort_values("_year_gap")
        return _row_to_dict(subset.iloc[0])


def _row_to_dict(row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        elif pd.isna(value) if not isinstance(value, (list, dict)) else False:
            out[key] = None
        else:
            out[key] = value
    return out


__all__ = ["LocationNotServedError", "LocationResolver", "ResolvedLocation"]
