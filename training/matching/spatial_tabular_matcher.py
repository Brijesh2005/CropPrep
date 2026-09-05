"""R5.2.9 enhanced spatial-tabular matcher.

The R5.2.7 pipeline matched government survey points against a tabular
district-level source (``district_grid``) that carried **no** environmental
feature columns, so every frozen observation exposed only
``lat / lon / spatial_match_distance_km / year / season`` on its tabular
branch.  This module implements the R5.2.9 enhancement: for every observation
it spatially matches the point against the dense DK_Features grid-cell point
cloud (2018-2023, Dakshina Kannada) and interpolates the **real** per-cell
environmental features (rainfall, soil, elevation, humidity, temperature and
annual + seasonal vegetation composites) onto the survey point using K-NN
inverse-distance weighting.

Key properties:

- **Geographic distance**: nearest-neighbour search runs on a 3-D unit-sphere
  KDTree (true chord distance ⇒ correct ordering); the reported distance is the
  haversine great-circle distance in metres.
- **K-NN + IDW**: continuous features are inverse-distance weighted
  (``1 / (d + eps) ** power``) over the K nearest grid cells; NaN neighbours are
  dropped and weights renormalised per feature.
- **Categorical majority**: categorical fields resolve by distance-weighted
  majority vote.
- **Year-aware**: features come from the grid file of the observation's own
  year, falling back to the nearest available year (never mixing unrelated
  years silently — the grid year used is always recorded).
- **Season-aware**: seasonal composite columns (``Kharif_*`` / ``Rabi_*``)
  are selected by the observation season; the other season's composites are
  still real per-cell values and are emitted so the feature schema is dense.
- **Leakage-safe**: ``Yield_Proxy_NPP`` is excluded from the interpolated
  schema so the crop benchmark can never be answered from the tabular branch.
  :func:`validate_no_leakage` guards the emitted schema.
- **Provenance**: every match records the nearest ``system:index``,
  neighbour distances, grid year, method and a confidence tier.

The matching hierarchy (A exact cell -> B grid-cell proximity -> C K-NN IDW)
is pure spatial: no external geocoding API, no LLM name matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from shared.logging import get_logger, log_dict

logger = get_logger("training.matching.spatial_tabular_matcher")

#: Mean Earth radius (IAU 1976), metres — used for great-circle distances.
EARTH_RADIUS_M = 6_371_008.8

#: Dakshina Kannada bounding box the DK grid inhabits (sanity check only).
DK_LAT_BOUNDS: tuple[float, float] = (12.40, 13.40)
DK_LON_BOUNDS: tuple[float, float] = (74.60, 75.90)

#: Distributed columns never interpolated / emitted as features.
_META_COLUMNS = frozenset(
    {
        "system:index",
        ".geo",
        "Country",
        "State",
        "District",
        "Year",
        "Latitude",
        "Longitude",
        "Season",
    }
)

#: Target-leakage columns that must never enter the benchmark feature schema.
DEFAULT_EXCLUDED_COLUMNS = frozenset({"Yield_Proxy_NPP"})

#: Continuous features resolved by inverse-distance weighting.
# NOTE: ``Area_sq_km`` is intentionally absent — per the R5.2.9 feature-quality
# audit it is district-constant (std 0) and would inject a constant into the
# tabular encoder.
DEFAULT_CONTINUOUS_FEATURES = frozenset(
    {
        "Annual_Rainfall_mm",
        "Dewpoint_C",
        "EVI",
        "Elevation",
        "NDRE",
        "NDVI",
        "NDWI",
        "Relative_Humidity_Pct",
        "S2_Obs_Count",
        "SAVI",
        "Slope",
        "Soil_Clay_Pct",
        "Soil_Moisture",
        "Soil_Organic_Carbon",
        "Soil_Sand_Pct",
        "Soil_pH",
        "Temperature_C",
        # Seasonal composites — every DK grid cell carries the full Kharif and
        # Rabi composite set, so all six are interpolated and emitted with a
        # row-constant schema. ``env_season_for_features`` records which set is
        # relevant for the observation season.
        "Kharif_NDVI",
        "Kharif_EVI",
        "Kharif_NDWI",
        "Rabi_NDVI",
        "Rabi_EVI",
        "Rabi_NDWI",
    }
)

#: Categorical features resolved by distance-weighted majority vote.
DEFAULT_CATEGORICAL_FEATURES = frozenset(
    {
        "Is_Cropland",
        "Land_Cover_Class",
        "Soil_Type_Class",
    }
)

#: Season -> seasonal composite columns present on every DK grid cell.
SEASON_COMPOSITES: dict[str, tuple[str, ...]] = {
    "Kharif": ("Kharif_NDVI", "Kharif_EVI", "Kharif_NDWI"),
    "Rabi": ("Rabi_NDVI", "Rabi_EVI", "Rabi_NDWI"),
}

#: Emission names (lowercase snake-case) for continuous DK columns.
_CONTINUOUS_EMIT_NAMES = {
    "Annual_Rainfall_mm": "annual_rainfall_mm",
    "Area_sq_km": "area_sq_km",
    "Dewpoint_C": "dewpoint_c",
    "EVI": "evi",
    "Elevation": "elevation",
    "NDRE": "ndre",
    "NDVI": "ndvi",
    "NDWI": "ndwi",
    "Relative_Humidity_Pct": "relative_humidity_pct",
    "S2_Obs_Count": "s2_obs_count",
    "SAVI": "savi",
    "Slope": "slope",
    "Soil_Clay_Pct": "soil_clay_pct",
    "Soil_Moisture": "soil_moisture",
    "Soil_Organic_Carbon": "soil_organic_carbon",
    "Soil_Sand_Pct": "soil_sand_pct",
    "Soil_pH": "soil_ph",
    "Temperature_C": "temperature_c",
    "Kharif_NDVI": "kharif_ndvi",
    "Kharif_EVI": "kharif_evi",
    "Kharif_NDWI": "kharif_ndwi",
    "Rabi_NDVI": "rabi_ndvi",
    "Rabi_EVI": "rabi_evi",
    "Rabi_NDWI": "rabi_ndwi",
}

_CONFIDENCE_NAMES: tuple[str, ...] = ("HIGH", "MEDIUM", "LOW", "VERY_LOW")


def _emit_name(column: str) -> str:
    return _CONTINUOUS_EMIT_NAMES.get(column, column.lower().replace(" ", "_"))


def haversine_m(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """Great-circle distance in metres (vectorised)."""
    lat1 = np.atleast_1d(np.radians(np.asarray(lat1, dtype=float)))
    lon1 = np.atleast_1d(np.radians(np.asarray(lon1, dtype=float)))
    lat2 = np.atleast_1d(np.radians(np.asarray(lat2, dtype=float)))
    lon2 = np.atleast_1d(np.radians(np.asarray(lon2, dtype=float)))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _to_unit_sphere(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Map (lat, lon) degrees to 3-D unit-sphere points."""
    phi = np.radians(lat)
    lam = np.radians(lon)
    cos_phi = np.cos(phi)
    return np.column_stack([cos_phi * np.cos(lam), cos_phi * np.sin(lam), np.sin(phi)])


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #


@dataclass
class DKGrid:
    """A single year's DK_Features grid cell cloud with a K-D tree index.

    Attributes:
        year: The grid year.
        frame: The full tidy DataFrame for the year.
        continuous_columns / categorical_columns: Subset columns applied.
        valid_continuous: Continuous columns that actually exist in the frame.
        tree: cKDTree over unit-sphere coordinates.
    """

    year: int
    frame: pd.DataFrame
    continuous_columns: frozenset[str]
    categorical_columns: frozenset[str]
    valid_continuous: tuple[str, ...]
    valid_categorical: tuple[str, ...]
    tree: cKDTree

    @classmethod
    def build(
        cls,
        path: Path,
        *,
        continuous_columns: Sequence[str],
        categorical_columns: Sequence[str],
    ) -> "DKGrid":
        if not path.exists():
            raise FileNotFoundError(f"DK grid file not found: {path}")
        frame = pd.read_csv(path)
        missing = {"Latitude", "Longitude"} - set(frame.columns)
        if missing:
            raise ValueError(
                f"{path.name} missing coordinate columns: {sorted(missing)}"
            )

        lat = frame["Latitude"].astype(float).to_numpy()
        lon = frame["Longitude"].astype(float).to_numpy()
        if not (
            DK_LAT_BOUNDS[0] <= lat.min() and lat.max() <= DK_LAT_BOUNDS[1]
        ) or not (
            DK_LON_BOUNDS[0] <= lon.min() and lon.max() <= DK_LON_BOUNDS[1]
        ):
            raise ValueError(f"{path.name} coordinates outside DK bounds")

        year = int(float(frame["Year"].iloc[0])) if "Year" in frame.columns else 0

        valid_cont = tuple(
            c for c in continuous_columns
            if c in frame.columns and c in _CONTINUOUS_EMIT_NAMES
        )
        valid_cat = tuple(c for c in categorical_columns if c in frame.columns)

        return cls(
            year=year,
            frame=frame.reset_index(drop=True),
            continuous_columns=frozenset(continuous_columns),
            categorical_columns=frozenset(categorical_columns),
            valid_continuous=valid_cont,
            valid_categorical=valid_cat,
            tree=cKDTree(_to_unit_sphere(lat, lon)),
        )


# --------------------------------------------------------------------------- #
# Match result
# --------------------------------------------------------------------------- #


@dataclass
class MatchResult:
    """Result of a spatial-tabular match for a single survey point."""

    lat: float
    lon: float
    year: int
    season: str | None
    matched: bool
    distances_m: list[float] = field(default_factory=list)
    dk_indices: list[int] = field(default_factory=list)
    nearest_index: str | None = None
    grid_year: int | None = None
    season_for_features: str | None = None
    method: str = "no_match"
    confidence: str | None = None
    features: dict[str, float | str | None] = field(default_factory=dict)
    support: float | None = None

    @property
    def nearest_distance_m(self) -> float | None:
        return self.distances_m[0] if self.distances_m else None


# --------------------------------------------------------------------------- #
# Matcher
# --------------------------------------------------------------------------- #


class SpatialTabularMatcher:
    """Enhanced spatial-tabular matcher (R5.2.9).

    Loads the DK_Features grid clouds (one per year) and interpolates
    environmental features onto survey points using exact-haversine K-NN
    inverse-distance weighting.

    Args:
        dk_dir: Directory holding ``DK_Features_YYYY.csv`` files.
        max_search_radius_km: Points beyond this radius are not matched
            (distance still recorded; ``matched=False``).
        knn_k: Number of nearest grid cells used for interpolation.
        idw_power: Exponent for inverse-distance weights.
        years: Grid years to load (defaults to all files present).
        continuous_columns: Continuous DK columns to interpolate.
        categorical_columns: Categorical DK columns to resolve by vote.
        excluded_columns: Columns forced out of the emitted schema
            (leakage guard).
    """

    def __init__(
        self,
        dk_dir: str | Path,
        *,
        max_search_radius_km: float = 5.0,
        knn_k: int = 5,
        idw_power: float = 2.0,
        years: Sequence[int] | None = None,
        continuous_columns: Sequence[str] | None = None,
        categorical_columns: Sequence[str] | None = None,
        excluded_columns: Sequence[str] | None = None,
    ) -> None:
        self.dk_dir = Path(dk_dir)
        self.max_search_radius_m = max_search_radius_km * 1000.0
        self.knn_k = int(knn_k)
        self.idw_power = float(idw_power)
        self.continuous_columns = tuple(
            continuous_columns or DEFAULT_CONTINUOUS_FEATURES
        )
        self.categorical_columns = tuple(
            categorical_columns or DEFAULT_CATEGORICAL_FEATURES
        )
        self.excluded_columns = frozenset(
            excluded_columns or DEFAULT_EXCLUDED_COLUMNS
        )

        if (frozenset(self.continuous_columns) | frozenset(self.categorical_columns)) & self.excluded_columns:
            raise ValueError(
                "Excluded (leakage) columns appear in the interpolated schema: "
                f"{sorted((frozenset(self.continuous_columns) | frozenset(self.categorical_columns)) & self.excluded_columns)}"
            )

        self._grids: dict[int, DKGrid] = {}
        available = self._discover_years(years)
        for year in available:
            grid = DKGrid.build(
                self.dk_dir / f"DK_Features_{year}.csv",
                continuous_columns=self.continuous_columns,
                categorical_columns=self.categorical_columns,
            )
            self._grids[grid.year] = grid
            log_dict(
                logger,
                logging.INFO,
                "DK grid loaded",
                year=grid.year,
                cells=len(grid.frame),
                continuous=len(grid.valid_continuous),
                categorical=len(grid.valid_categorical),
            )
        if not self._grids:
            raise ValueError(f"No DK_Features_*.csv files found under {self.dk_dir}")
        self._years = sorted(self._grids)

    def _discover_years(self, years: Sequence[int] | None) -> list[int]:
        if years:
            return [int(y) for y in years]
        found = []
        for path in sorted(self.dk_dir.glob("DK_Features_*.csv")):
            stem = path.stem  # e.g. "DK_Features_2020"
            digits = stem.rsplit("_", 1)[-1]
            if digits.isdigit():
                found.append(int(digits))
        return found

    @property
    def years(self) -> list[int]:
        """Available grid years (sorted ascending)."""
        return list(self._years)

    @property
    def emitted_feature_names(self) -> list[str]:
        """Emitted (lowercase) continuous + categorical feature names."""
        names = [_emit_name(c) for c in self.continuous_columns]
        names += [_emit_name(c) for c in self.categorical_columns]
        return names

    def _nearest_grid_year(self, year: int) -> int:
        if year in self._grids:
            return year
        return min(self._years, key=lambda y: abs(y - year))

    def _choose_season(self, season: str | None) -> str | None:
        if season is None:
            return None
        if season in SEASON_COMPOSITES:
            return season
        for candidate in SEASON_COMPOSITES:
            if candidate.lower() in str(season).lower():
                return candidate
        return None

    def match(
        self,
        lon: float,
        lat: float,
        year: int,
        season: str | None = None,
    ) -> MatchResult:
        """Match a single survey point against the DK grid and interpolate.

        Args:
            lon / lat: WGS84 coordinates of the survey point.
            year: Observation year (drives grid selection).
            season: ``Kharif`` / ``Rabi`` / ``Zaid`` / None — drives seasonal
                composite selection.

        Returns:
            A :class:`MatchResult` with interpolated features and provenance.
        """
        lon_f = float(lon)
        lat_f = float(lat)
        grid_year = self._nearest_grid_year(int(year))
        grid = self._grids[grid_year]

        point = _to_unit_sphere(
            np.array([lat_f]), np.array([lon_f])
        )[0]
        tree_dist, tree_idx = grid.tree.query(
            point, k=min(self.knn_k * 2, len(grid.frame))
        )
        tree_idx = np.atleast_1d(tree_idx).tolist()

        lat_g = grid.frame["Latitude"].to_numpy()[tree_idx]
        lon_g = grid.frame["Longitude"].to_numpy()[tree_idx]
        real_dist = haversine_m(lat_f, lon_f, lat_g, lon_g)
        order = np.argsort(real_dist)
        tree_idx = [tree_idx[i] for i in order]
        real_dist = sorted(real_dist)

        if not real_dist or real_dist[0] > self.max_search_radius_m:
            return MatchResult(
                lat=lat_f,
                lon=lon_f,
                year=int(year),
                season=season,
                matched=False,
                distances_m=list(real_dist),
                dk_indices=tree_idx,
                nearest_index=str(grid.frame["system:index"].iloc[tree_idx[0]])
                if tree_idx
                else None,
                grid_year=grid_year,
                season_for_features=self._choose_season(season),
                method="out_of_range",
                confidence="VERY_LOW",
            )

        k = min(self.knn_k, len(tree_idx))
        idx = tree_idx[:k]
        dist = real_dist[:k]
        nearest = grid.frame.iloc[idx]

        features: dict[str, float | str | None] = {}
        weights = np.array([1.0 / (d + 1e-6) ** self.idw_power for d in dist])

        for column in grid.valid_continuous:
            col = nearest[column].to_numpy(dtype=float)
            finite = ~np.isnan(col)
            if not finite.any():
                features[_emit_name(column)] = None
                continue
            if col[0] == col[0] and abs(dist[0]) < 1.0:
                features[_emit_name(column)] = float(col[0])
                continue
            w = weights[finite]
            values = col[finite]
            features[_emit_name(column)] = float(np.sum(w * values) / max(w.sum(), 1e-12))

        for column in grid.valid_categorical:
            col = nearest[column].astype(str).to_numpy()
            valid = col != "nan"
            if not valid.any():
                features[_emit_name(column)] = None
                continue
            dist_arr = np.asarray(dist)
            votes: dict[str, float] = {}
            nearest_for: dict[str, float] = {}
            for weight, value, d in zip(weights[valid], col[valid], dist_arr[valid]):
                votes[value] = votes.get(value, 0.0) + weight
                nearest_for[value] = min(nearest_for.get(value, np.inf), float(d))
            winner = max(sorted(votes), key=lambda v: (votes[v], -nearest_for[v]))
            features[_emit_name(column)] = winner

        season_for_features = self._choose_season(season)
        if abs(dist[0]) < 1.0:
            method = "exact_cell"
        else:
            method = "knn_idw"

        confidence = self._confidence(dist[0], int(year), grid_year, season_for_features)

        n_feature_cols = max(
            len(grid.valid_continuous) + len(grid.valid_categorical), 1
        )
        support = (
            sum(1 for value in features.values() if value is not None) / n_feature_cols
        )

        return MatchResult(
            lat=lat_f,
            lon=lon_f,
            year=int(year),
            season=season,
            matched=True,
            distances_m=list(dist),
            dk_indices=idx,
            nearest_index=str(grid.frame["system:index"].iloc[idx[0]]),
            grid_year=grid_year,
            season_for_features=season_for_features,
            method=method,
            confidence=confidence,
            features=features,
            support=support,
        )

    def _confidence(
        self,
        nearest_m: float,
        obs_year: int,
        grid_year: int,
        season_for_features: str | None,
    ) -> str:
        year_exact = obs_year == grid_year
        season_ok = season_for_features is None or season_for_features in SEASON_COMPOSITES
        if not year_exact or not season_ok:
            return "LOW"
        if nearest_m <= 250.0:
            return "HIGH"
        if nearest_m <= self.max_search_radius_m:
            return "MEDIUM"
        return "VERY_LOW"

    def match_rows(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        lon_col: str = "lon",
        lat_col: str = "lat",
        year_col: str = "year",
        season_col: str = "season",
    ) -> list[MatchResult]:
        """Batch wrapper around :meth:`match` for record dicts/rows."""
        out = []
        for row in records:
            try:
                out.append(
                    self.match(
                        float(row[lon_col]),
                        float(row[lat_col]),
                        int(float(row[year_col])),
                        str(row[season_col]) if row.get(season_col) else None,
                    )
                )
            except (TypeError, ValueError, KeyError) as exc:
                log_dict(
                    logger,
                    logging.WARNING,
                    "Match skipped for invalid record",
                    record=row,
                    error=str(exc),
                )
                def _p(col: str) -> Any:
                    try:
                        return float(str(row.get(col, "")).strip())
                    except (TypeError, ValueError):
                        return float("nan")

                def _s(col: str) -> str | None:
                    value = row.get(col)
                    return str(value) if value is not None else None

                out.append(
                    MatchResult(
                        lat=_p(lat_col),
                        lon=_p(lon_col),
                        year=int(_p(year_col)) if not np.isnan(_p(year_col)) else 0,
                        season=_s(season_col),
                        matched=False,
                        method="invalid_record",
                        confidence="VERY_LOW",
                    )
                )
        return out

    def validate_no_leakage(self, schema: Sequence[str]) -> None:
        """Assert no excluded (target-leakage) column is present in ``schema``."""
        leaked = [c for c in schema if c.lower().replace("_", "") in {
            c.lower().replace("_", "") for c in self.excluded_columns
        }]
        if leaked:
            raise ValueError(f"Leakage columns present in schema: {leaked}")