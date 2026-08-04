"""Tests for the dataset registry."""

from __future__ import annotations

from pathlib import Path

from services.dataset_manager.dataset_registry import SQLiteRegistry
from services.dataset_manager.models import DatasetStatus


def _registry(tmp_path: Path) -> SQLiteRegistry:
    return SQLiteRegistry(tmp_path / "registry.db")


def test_register_and_get(tmp_path: Path):
    registry = _registry(tmp_path)
    dataset_id = registry.register(
        name="kaggle-crop-yield", source="kaggle", root_path=tmp_path / "root"
    )
    entry = registry.get(dataset_id)
    assert entry is not None
    assert entry["name"] == "kaggle-crop-yield"
    assert entry["status"] == DatasetStatus.PENDING.value
    assert entry["version"] == "0.0.0"


def test_get_by_name_returns_latest(tmp_path: Path):
    registry = _registry(tmp_path)
    registry.register(name="ds", source="x", root_path=tmp_path / "a")
    registry.register(name="ds", source="x", root_path=tmp_path / "b")
    latest = registry.get_by_name("ds")
    assert latest["dataset_id"] > 0
    assert latest["root_path"].endswith("b")


def test_update_status_and_checksum(tmp_path: Path):
    registry = _registry(tmp_path)
    dataset_id = registry.register(name="ds", source="x", root_path=tmp_path / "a")
    registry.update_status(dataset_id, DatasetStatus.READY)
    registry.update_checksum(dataset_id, "abc123", 42)
    entry = registry.get(dataset_id)
    assert entry["status"] == "ready"
    assert entry["checksum"] == "abc123"
    assert entry["file_count"] == 42


def test_list_and_status_filter(tmp_path: Path):
    registry = _registry(tmp_path)
    a = registry.register(name="a", source="x", root_path=tmp_path / "a")
    b = registry.register(name="b", source="x", root_path=tmp_path / "b")
    registry.update_status(a, DatasetStatus.READY)
    registry.update_status(b, DatasetStatus.FAILED)
    assert len(registry.list()) == 2
    ready = registry.list_by_status(DatasetStatus.READY)
    assert len(ready) == 1 and ready[0]["name"] == "a"


def test_remove(tmp_path: Path):
    registry = _registry(tmp_path)
    dataset_id = registry.register(name="ds", source="x", root_path=tmp_path / "a")
    assert registry.remove(dataset_id) is True
    assert registry.get(dataset_id) is None
    assert registry.remove(dataset_id) is False


def test_unknown_update_field_rejected(tmp_path: Path):
    registry = _registry(tmp_path)
    dataset_id = registry.register(name="ds", source="x", root_path=tmp_path / "a")
    import pytest

    from services.dataset_manager.exceptions import RegistryError

    with pytest.raises(RegistryError):
        registry._update(dataset_id, not_a_field=1)
