"""Tests for :class:`PatchExtractorImpl` — geographic patch extraction.

Uses the fully-wired R2.2 manager fixture (real image provider + metadata
repository) so extraction, CRS conversion, edge padding and persistence are all
exercised against synthetic GeoTIFFs — no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pytest

from training.dataset_manager.exceptions import DatasetNotFoundError


def test_extract_ndvi_2019(r22_manager_factory):
    manager = r22_manager_factory()
    patch = manager.get_patch(13.08, 74.89, 8, index_type="NDVI", year=2019)
    assert isinstance(patch, np.ndarray)
    assert patch.shape == (8, 8)
    assert np.isfinite(patch).all()


def test_extract_returns_raw_float32(r22_manager_factory):
    manager = r22_manager_factory()
    patch = manager.get_patch(13.08, 74.89, 16, index_type="NDVI", year=2019)
    assert patch.dtype == np.float32
    assert patch.min() >= 0.0 and patch.max() <= 1.0


def test_extract_ndvi_2020_picks_different_raster(r22_manager_factory):
    manager = r22_manager_factory()
    patch = manager.get_patch(12.90, 75.00, 8, index_type="NDVI", year=2020)
    assert patch.shape == (8, 8)


def test_extract_with_metadata_persists(r22_manager_factory):
    manager = r22_manager_factory()
    array, metadata = manager.patch_extractor.extract_with_metadata(
        13.08, 74.89, 8, index_type="NDVI", year=2019
    )
    assert array.shape == (8, 8)
    assert metadata.size == 8
    assert metadata.crs is not None
    assert "NDVI" in str(metadata.path)
    assert metadata.padded is False
    patches = manager.list_patches()
    assert len(patches) >= 1
    assert patches[-1]["size"] == 8


def test_extract_edge_padding(r22_manager_factory):
    manager = r22_manager_factory()
    # Far north-west of every raster -> window is clamped, then edge-padded.
    patch, metadata = manager.patch_extractor.extract_with_metadata(
        30.0, 90.0, 8, index_type="NDVI", year=2019
    )
    assert patch.shape == (8, 8)
    assert metadata.padded is True


def test_extract_no_matching_index_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(DatasetNotFoundError):
        manager.get_patch(13.08, 74.89, 8, index_type="EVI", year=2020)


def test_extract_no_matching_resolution_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(DatasetNotFoundError):
        manager.get_patch(13.08, 74.89, 8, index_type="NDVI", resolution="R20m")


def test_extract_invalid_size_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(ValueError):
        manager.get_patch(13.08, 74.89, 0, index_type="NDVI")


def test_extract_invalid_coordinates_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(ValueError):
        manager.get_patch(float("nan"), 74.89, 8, index_type="NDVI")


def test_extract_infinity_coordinates_raises(r22_manager_factory):
    manager = r22_manager_factory()
    with pytest.raises(ValueError):
        manager.get_patch(float("inf"), 74.89, 8, index_type="NDVI")
