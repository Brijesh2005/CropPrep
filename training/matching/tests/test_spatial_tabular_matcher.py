"""Tests for the R5.2.9 spatial-tabular matcher."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.matching.spatial_tabular_matcher import (
    DKGrid,
    SEASON_COMPOSITES,
    SpatialTabularMatcher,
    haversine_m,
)


def _sample_grid(frame: pd.DataFrame, columns_prefix: str = "DK_Features") -> None:
    pass


@pytest.fixture
def dk_dir(tmp_path):
    """A small synthetic DK grid (2x2 cells, two years) on disk."""
    cells = []
    base_lat, base_lon = 12.70, 75.00
    for row in range(2):
        for col in range(2):
            lat = base_lat + row * 0.01
            lon = base_lon + col * 0.01
            for year in (2020, 2022):
                cells.append(
                    {
                        "system:index": f"{row}_{col}",
                        "Latitude": lat,
                        "Longitude": lon,
                        "Year": year,
                        "Season": "Annual + Kharif/Rabi composites",
                        "Annual_Rainfall_mm": 1000.0 + row * 100 + col * 10 + year,
                        "NDVI": 0.1 + row * 0.1 + col * 0.05,
                        "Kharif_NDVI": 0.2 + row * 0.1 + col * 0.05,
                        "Rabi_NDVI": 0.05 + row * 0.05 + col * 0.02,
                        "EVI": 1.0 + row + col,
                        "Soil_pH": 50.0 + row + col,
                        "Land_Cover_Class": f"c{row + col}",
                        "Is_Cropland": str((row + col) % 2),
                        "Soil_Type_Class": "soil_" + str(row),
                    }
                )
    frame = pd.DataFrame(cells)
    for year in (2020, 2022):
        path = tmp_path / f"DK_Features_{year}.csv"
        frame[frame["Year"] == year].to_csv(path, index=False)
    return tmp_path


def test_haversine_known_distance():
    # ~1 degree of latitude ~ 111.2 km.
    d = float(haversine_m(0.0, 0.0, 1.0, 0.0)[0])
    assert 111_000 < d < 112_000
    # Zero distance.
    assert float(haversine_m(12.5, 75.0, 12.5, 75.0)[0]) < 1e-6


def test_grid_build_bounds(dk_dir):
    bad = dk_dir / "DK_Features_2020.csv"
    frame = pd.read_csv(bad)
    frame.loc[0, "Latitude"] = 15.0  # outside DK bounds
    path = dk_dir / "broken.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        DKGrid.build(path, continuous_columns=["NDVI"], categorical_columns=[])


def test_leakage_guard(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    for name in m.emitted_feature_names:
        assert "yield_proxy" not in name.lower()
    m.validate_no_leakage(m.emitted_feature_names)
    with pytest.raises(ValueError):
        m.validate_no_leakage([*m.emitted_feature_names, "yield_proxy_npp"])
    # Excluded columns cannot be pulled into the schema at init.
    with pytest.raises(ValueError):
        SpatialTabularMatcher(
            dk_dir,
            years=[2020],
            continuous_columns=["NDVI", "Yield_Proxy_NPP"],
        )


def test_season_composite_schema(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    emitted = set(m.emitted_feature_names)
    for composite in SEASON_COMPOSITES["Kharif"]:
        assert composite.lower() in emitted


def test_exact_cell_value(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    # Land exactly on the (0,0) cell of 2020.
    r = m.match(75.00, 12.70, 2020, "Kharif")
    assert r.matched
    assert r.grid_year == 2020
    assert r.method in ("exact_cell", "knn_idw")
    assert abs(r.nearest_distance_m) < 10.0
    # Nearest-cell constant for the corner cell (lat=12.70, lon=75.00):
    assert abs(r.features["annual_rainfall_mm"] - (1000.0 + 2020)) < 1.0
    assert abs(r.features["ndvi"] - 0.1) < 1e-6
    assert r.season_for_features == "Kharif"
    assert "kharif_ndvi" in r.features
    assert abs(r.features["kharif_ndvi"] - 0.2) < 1e-6


def test_year_exact_and_fallback(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020, 2022], knn_k=2)
    # Exact year.
    r = m.match(75.00, 12.70, 2020, "Kharif")
    assert r.grid_year == 2020
    # Nearest-available-year fallback (2021 -> 2020, 2023 -> 2022).
    r21 = m.match(75.00, 12.70, 2021, "Kharif")
    assert r21.grid_year == 2020
    assert r21.confidence == "LOW"  # year fallback downgrades confidence
    r21 = m.match(75.00, 12.70, 2023, "Kharif")
    assert r21.grid_year == 2022


def test_confidence_tiers(dk_dir):
    m = SpatialTabularMatcher(
        dk_dir, years=[2020, 2022], knn_k=2, max_search_radius_km=5.0
    )
    # On-cell, exact year -> HIGH.
    assert m.match(75.00, 12.70, 2020, "Kharif").confidence == "HIGH"
    # Far but within radius, exact year -> MEDIUM.
    mid = m.match(75.005, 12.705, 2020, "Kharif")  # ~700 m off-cell
    assert mid.matched and mid.confidence == "MEDIUM"
    # Outside radius -> not matched.
    far = m.match(76.5, 12.70, 2020, "Kharif")
    assert not far.matched
    assert far.confidence == "VERY_LOW"


def test_idw_matches_cell_and_interpolates(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=4)
    # Interior point should produce an interpolated NDVI between neighbours.
    r = m.match(75.005, 12.705, 2020, "Kharif")
    assert r.matched
    assert 0.1 <= r.features["ndvi"] <= 0.3
    assert r.support > 0.0


def test_categorical_majority(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    r = m.match(75.00, 12.70, 2020, "Kharif")
    assert r.features["land_cover_class"] in {"c0", "c1", "c3"}
    assert r.features["soil_type_class"] in {"soil_0", "soil_1"}


def test_match_rows_batch_and_bad_record(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    records = [
        {"lon": 75.00, "lat": 12.70, "year": 2020, "season": "Kharif"},
        {"lon": 75.005, "lat": 12.705, "year": 2020, "season": "Rabi"},
        {"lon": "garbage", "lat": 12.70, "year": 2020, "season": "Kharif"},
    ]
    results = m.match_rows(records)
    assert results[0].matched
    assert results[1].matched
    assert results[1].season_for_features == "Rabi"
    assert not results[2].matched
    assert results[2].method == "invalid_record"


def test_invalid_season_falls_back(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    r = m.match(75.00, 12.70, 2020, "Zaid")
    assert r.matched
    # Zaid is not in the composite map: composites still emitted, features
    # resolved from the grid, but the observation has a season_for_features.
    assert "kharif_ndvi" in r.features


def test_dk_dir_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SpatialTabularMatcher(tmp_path, years=[2020])


def test_null_ndarray_handling(dk_dir):
    m = SpatialTabularMatcher(dk_dir, years=[2020], knn_k=2)
    # No NaNs in the fixture; confirm emission still full-coverage.
    r = m.match(75.00, 12.70, 2020, "Kharif")
    assert r.support == pytest.approx(1.0)


def test_nan_neighbour_renormalisation(tmp_path):
    cells = [
        {"system:index": "0_0", "Latitude": 12.70, "Longitude": 75.00,
         "Year": 2020, "Season": "x", "NDVI": 0.5, "EVI": np.nan},
        {"system:index": "0_1", "Latitude": 12.70, "Longitude": 75.01,
         "Year": 2020, "Season": "x", "NDVI": 0.9, "EVI": 1.1},
        {"system:index": "1_0", "Latitude": 12.71, "Longitude": 75.00,
         "Year": 2020, "Season": "x", "NDVI": 0.7, "EVI": 1.3},
    ]
    frame = pd.DataFrame(cells)
    path = tmp_path / "DK_Features_2020.csv"
    frame.to_csv(path, index=False)
    m = SpatialTabularMatcher(tmp_path, years=[2020], knn_k=3,
                              continuous_columns=["NDVI", "EVI"],
                              categorical_columns=[])
    # Point near the NaN cell but closer to valid neighbours.
    r = m.match(75.001, 12.7001, 2020, None)
    assert r.matched
    assert r.features["ndvi"] is not None
    assert r.features["evi"] is not None  # renormalised over non-NaN neighbours
    # EVI is renormalised over the non-NaN neighbours (0.9 / 1.3 weighted).
    assert 0.9 < r.features["evi"] < 1.3