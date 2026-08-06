"""End-to-end tests for the R2.2 Dataset Manager extension.

Verifies registry wiring, the multi-source API (tabular / image / patch /
location / years / indices / statistics / search / health), config-driven
provider overrides, reports, and the extended metadata repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd
import pytest

from training.dataset_manager.exceptions import DatasetNotFoundError


def test_registry_wiring(r22_manager_factory):
    manager = r22_manager_factory()
    assert manager.provider_registry.names() == [
        "git_repository_tabular",
        "kaggle_hub_image",
    ]
    # Legacy attributes are resolved through the registry.
    assert manager.tabular_provider is manager.provider_registry.resolve("git_repository_tabular")
    assert manager.image_provider is manager.provider_registry.resolve("kaggle_hub_image")


def test_provider_manifests(r22_manager_factory):
    manager = r22_manager_factory()
    manifests = manager.provider_manifests()
    assert set(manifests) == {"git_repository_tabular", "kaggle_hub_image"}
    assert manifests["git_repository_tabular"]["available"] is True
    assert manifests["kaggle_hub_image"]["available"] is True


def test_load_tabular_and_get_csv(r22_manager_factory):
    manager = r22_manager_factory()
    frame = manager.load_tabular("crop_yield")
    assert list(frame.columns) == [
        "village", "district", "latitude", "longitude", "year", "yield_kg"
    ]
    assert len(frame) == 3
    assert manager.get_csv("crop_yield").equals(frame)


def test_get_image(r22_manager_factory):
    manager = r22_manager_factory()
    array = manager.get_image("NDVI", year=2019)
    assert array.shape == (20, 20)
    with pytest.raises(DatasetNotFoundError):
        manager.get_image("EVI", year=2020)


def test_get_patch(r22_manager_factory):
    manager = r22_manager_factory()
    patch = manager.get_patch(13.08, 74.89, 8, index_type="NDVI", year=2019)
    assert patch.shape == (8, 8)


def test_get_location_by_name(r22_manager_factory):
    manager = r22_manager_factory()
    result = manager.get_location(name="Moodabidri")
    assert result.found is True
    assert result.records[0].latitude == 13.08
    assert result.records[0].longitude == 74.89


def test_get_location_by_coordinates(r22_manager_factory):
    manager = r22_manager_factory()
    result = manager.get_location(latitude=13.08, longitude=74.89)
    assert result.found is True
    assert result.records[0].name == "Moodabidri"


def test_get_location_by_radius(r22_manager_factory):
    manager = r22_manager_factory()
    result = manager.get_location(latitude=13.08, longitude=74.89, radius_km=30)
    assert result.found is True


def test_get_location_unknown(r22_manager_factory):
    manager = r22_manager_factory()
    result = manager.get_location(name="Nowhere")
    assert result.found is False
    assert result.records == []


def test_available_years_and_indices(r22_manager_factory):
    manager = r22_manager_factory()
    assert manager.get_available_years() == [2019, 2020]
    assert manager.get_available_years(index_type="EVI") == [2019]
    assert manager.get_available_indices() == ["EVI", "NDVI"]
    assert manager.get_available_indices(resolution="R10m") == ["EVI", "NDVI"]
    assert manager.get_resolutions() == ["R10m"]


def test_statistics_and_search(r22_manager_factory):
    manager = r22_manager_factory()
    stats = manager.statistics()
    assert stats.total_images == 3
    assert stats.total_tabular_rows == 3
    hits = manager.search("crop_yield")
    assert any(h["type"] == "tabular" for h in hits)
    assert manager.search("Moodabidri") == []


def test_health_availability_discovery(r22_manager_factory):
    manager = r22_manager_factory()
    health = manager.health()
    assert "git_repository_tabular" in health
    assert health["kaggle_hub_image"]["available"] is True
    assert manager.availability() == {
        "git_repository_tabular": True,
        "kaggle_hub_image": True,
    }
    registrations = manager.discovery()
    assert {r["name"] for r in registrations} == {
        "git_repository_tabular",
        "kaggle_hub_image",
    }


def test_spatial_metadata(r22_manager_factory):
    manager = r22_manager_factory()
    metadata = manager.spatial_metadata()
    assert metadata.count == 3
    assert metadata.villages == 3
    assert metadata.bounds is not None


def test_metadata_repository_accessors(r22_manager_factory):
    manager = r22_manager_factory()
    assert len(manager.provider_metadata()) == 2
    assert manager.list_patches() == []
    manager.get_patch(13.08, 74.89, 8, index_type="NDVI", year=2019)
    assert len(manager.list_patches()) == 1


def test_reports(r22_manager_factory):
    manager = r22_manager_factory()
    paths = manager.reports()
    assert len(paths) == 7


def test_config_disables_provider(r22_manager_factory):
    manager = r22_manager_factory(
        settings_overrides={
            "providers": {
                "registry": {
                    "providers": [
                        {
                            "name": "kaggle_hub_image",
                            "kind": "image",
                            "enabled": False,
                            "priority": 100,
                        }
                    ]
                }
            }
        }
    )
    assert manager.provider_registry.has("kaggle_hub_image")
    assert manager.availability()["kaggle_hub_image"] is False
    with pytest.raises(DatasetNotFoundError):
        manager.provider_registry.resolve("kaggle_hub_image")


def test_config_registers_additional_provider(r22_manager_factory, tmp_path: Path):
    extra = tmp_path / "extra_tabular"
    extra.mkdir()
    pd.DataFrame({"village": ["X"], "yield_kg": [9]}).to_csv(
        extra / "extra.csv", index=False
    )
    manager = r22_manager_factory(
        settings_overrides={
            "providers": {
                "registry": {
                    "providers": [
                        {
                            "name": "aux_tabular",
                            "kind": "tabular",
                            "enabled": True,
                            "priority": 50,
                            "config": {"root": str(extra)},
                        }
                    ]
                }
            }
        }
    )
    assert manager.provider_registry.has("aux_tabular")
    tabulars = manager.provider_registry.resolve_by_kind("tabular")
    assert len(tabulars) == 2
    # Default provider outranks the auxiliary one.
    assert tabulars[0].name == "git_repository_tabular"
    assert tabulars[1].name == "aux_tabular"


def test_historical_context_via_manager(r22_manager_factory):
    manager = r22_manager_factory()
    context = manager.build_historical_context(
        village="Moodabidri", index_type="NDVI"
    )
    assert context.location == "Moodabidri"
    assert context.years == [2019, 2020]
    records = manager.temporal_metadata()
    assert len(records) >= 1
