"""Tests for the semantic version manager."""

from __future__ import annotations

from pathlib import Path

import pytest

from services.dataset_manager.dataset_registry import SQLiteRegistry
from services.dataset_manager.exceptions import RegistryError
from services.dataset_manager.version_manager import (
    SQLiteVersionManager,
    bump_version,
)


@pytest.fixture
def version_manager(tmp_path: Path) -> tuple[SQLiteVersionManager, int]:
    registry = SQLiteRegistry(tmp_path / "registry.db")
    dataset_id = registry.register(name="ds", source="kaggle", root_path=tmp_path / "root")
    manager = SQLiteVersionManager(tmp_path / "registry.db", registry)
    return manager, dataset_id


def test_bump_patch(version_manager):
    manager, dataset_id = version_manager
    entry = manager.bump(dataset_id, "patch", message="fix")
    assert entry.version == "0.0.1"
    assert manager.current(dataset_id) == "0.0.1"


def test_bump_minor_resets_patch(version_manager):
    manager, dataset_id = version_manager
    manager.bump(dataset_id, "patch")
    manager.bump(dataset_id, "patch")
    entry = manager.bump(dataset_id, "minor")
    assert entry.version == "0.1.0"


def test_bump_major_resets_all(version_manager):
    manager, dataset_id = version_manager
    manager.bump(dataset_id, "patch")
    manager.bump(dataset_id, "minor")
    entry = manager.bump(dataset_id, "major")
    assert entry.version == "1.0.0"


def test_history_ordered(version_manager):
    manager, dataset_id = version_manager
    manager.bump(dataset_id, "patch", message="one")
    manager.bump(dataset_id, "patch", message="two")
    manager.bump(dataset_id, "patch", message="three")
    versions = manager.list(dataset_id)
    assert [v.version for v in versions] == ["0.0.3", "0.0.2", "0.0.1"]


def test_rollback(version_manager):
    manager, dataset_id = version_manager
    manager.bump(dataset_id, "patch", message="v1")
    manager.bump(dataset_id, "minor", message="v2")
    assert manager.current(dataset_id) == "0.1.0"
    entry = manager.rollback(dataset_id, "0.0.1")
    assert entry.version == "0.0.1"
    assert manager.current(dataset_id) == "0.0.1"
    # The rolled-back row becomes current again.
    versions = manager.list(dataset_id)
    current = [v for v in versions if v.is_current]
    assert len(current) == 1 and current[0].version == "0.0.1"


def test_rollback_missing_version_raises(version_manager):
    manager, dataset_id = version_manager
    with pytest.raises(RegistryError):
        manager.rollback(dataset_id, "9.9.9")


def test_invalid_version_rejected(version_manager):
    manager, dataset_id = version_manager
    with pytest.raises(RegistryError):
        manager.snapshot(dataset_id, "not-a-version", message="x", checksum=None, file_count=0)


def test_snapshot_records_checksum_and_registry(version_manager):
    manager, dataset_id = version_manager
    manager.snapshot(
        dataset_id, "2.0.0", message="rebuild", checksum="cafe", file_count=99
    )
    assert manager.current(dataset_id) == "2.0.0"
    entry = manager.registry.get(dataset_id)
    assert entry["checksum"] == "cafe"
    assert entry["file_count"] == 99


def test_bump_version_helper():
    assert bump_version("0.0.0", "patch") == "0.0.1"
    assert bump_version("1.2.3", "minor") == "1.3.0"
    assert bump_version("1.2.3", "major") == "2.0.0"
    with pytest.raises(RegistryError):
        bump_version("1.2.3", "banana")
    with pytest.raises(RegistryError):
        bump_version("abc", "patch")
