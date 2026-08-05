"""Tests for shared constants."""

from __future__ import annotations

from shared.constants import (
    CRS_UTM_43N,
    DEFAULT_KAGGLE_HANDLE,
    DIR_RAW,
    ENV_PREFIX_BACKEND,
    ENV_PREFIX_DATASET,
    EXCLUDE_DIRS,
    PROVIDER_KAGGLE_IMAGE,
    RASTER_SUFFIXES,
)


def test_kaggle_handle_default() -> None:
    assert "crop-yield-forecasting" in DEFAULT_KAGGLE_HANDLE


def test_raster_suffixes() -> None:
    assert ".tif" in RASTER_SUFFIXES
    assert ".tiff" in RASTER_SUFFIXES


def test_crs_utm() -> None:
    assert CRS_UTM_43N == "EPSG:32643"


def test_exclude_dirs() -> None:
    assert ".cropfusion" in EXCLUDE_DIRS
    assert ".git" in EXCLUDE_DIRS


def test_env_prefixes_distinct() -> None:
    prefixes = {ENV_PREFIX_DATASET, ENV_PREFIX_BACKEND}
    assert "DM_" in prefixes
    assert "BACKEND_" in prefixes


def test_provider_names() -> None:
    assert PROVIDER_KAGGLE_IMAGE == "kaggle-image"
    assert DIR_RAW == "raw"
