"""Tests for :meth:`DatasetManager.get_historical_context`.

The historical context aggregates satellite records by year for a recurring
season window — the "same location + same season across all years" evidence
used before model inference.
"""

from __future__ import annotations

import pandas as pd
import pytest

from services.dataset_manager.models import HistoricalContext
from services.dataset_manager.tests.helpers import make_tiff


@pytest.fixture
def historical_catalog(tmp_path):
    root = tmp_path / "datasets"
    catalog = root / "raw" / "kaggle-crop-yield"
    (catalog / "2019_images" / "R10m").mkdir(parents=True)
    (catalog / "2020_images" / "R10m").mkdir(parents=True)
    (catalog / "2021_images" / "R10m").mkdir(parents=True)

    pd.DataFrame({"village": ["A"], "crop": ["Rice"], "yield_kg": [5200]}).to_csv(
        catalog / "crop_yield.csv", index=False
    )

    make_tiff(catalog / "2019_images" / "R10m" / "S2_NDVI_20190701.tif", seed=1)
    make_tiff(catalog / "2019_images" / "R10m" / "S2_EVI_20190701.tif", seed=2)
    make_tiff(catalog / "2020_images" / "R10m" / "S2_NDVI_20200601.tif", seed=3)
    make_tiff(catalog / "2020_images" / "R10m" / "S2_NDVI_20200901.tif", seed=4)
    # Rabi window month (March) in a different year.
    make_tiff(catalog / "2021_images" / "R10m" / "S2_NDVI_20210301.tif", seed=5)
    return root


def test_historical_context_by_season_months(manager_factory, historical_catalog):
    manager = manager_factory(historical_catalog)
    manager.generate_metadata(force=True)

    ctx = manager.get_historical_context(
        window_months=[6, 7, 8, 9, 10], resolution="R10m"
    )
    assert isinstance(ctx, HistoricalContext)
    assert ctx.window_months == [6, 7, 8, 9, 10]
    assert ctx.years == [2019, 2020]
    assert ctx.total_records == 4
    assert ctx.per_year[2019] == {"records": 2, "ndvi": 1, "evi": 1}
    assert ctx.per_year[2020] == {"records": 2, "ndvi": 2}


def test_historical_context_crossing_season(manager_factory, historical_catalog):
    manager = manager_factory(historical_catalog)
    manager.generate_metadata(force=True)

    ctx = manager.get_historical_context(
        window_months=[11, 12, 1, 2, 3], resolution="R10m"
    )
    assert ctx.years == [2021]
    assert ctx.total_records == 1
    assert ctx.per_year[2021] == {"records": 1, "ndvi": 1}


def test_historical_context_no_filter_counts_everything(
    manager_factory, historical_catalog
):
    manager = manager_factory(historical_catalog)
    manager.generate_metadata(force=True)

    ctx = manager.get_historical_context()
    assert ctx.years == [2019, 2020, 2021]
    assert ctx.total_records == 5
    assert ctx.window_months is None


def test_historical_context_year_restriction(manager_factory, historical_catalog):
    manager = manager_factory(historical_catalog)
    manager.generate_metadata(force=True)

    ctx = manager.get_historical_context(window_months=[6, 7, 8, 9, 10], years=[2020])
    assert ctx.years == [2020]
    assert ctx.total_records == 2


def test_historical_context_serializable(manager_factory, historical_catalog):
    manager = manager_factory(historical_catalog)
    manager.generate_metadata(force=True)

    payload = manager.get_historical_context(
        window_months=[6, 7, 8, 9, 10]
    ).to_dict()
    assert payload["years"] == [2019, 2020]
    assert payload["per_year"]["2019"] == {"records": 2, "ndvi": 1, "evi": 1}
    assert payload["window_months"] == [6, 7, 8, 9, 10]
    assert "generated_at" in payload
