"""Dataset-source snapshot tests (R5.2 Task 9).

Regression: ``DatasetManager`` never exposed ``metadata_db_path``, so
``persist_dataset_sources`` always raised ``INF-DS-001`` even when the database
existed, which cascaded into a missing ``location_index.parquet`` and a skipped
``village_metadata.parquet`` in the release package.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.inference.dataset_sources import (
    DatasetSourceError,
    _resolve_db_path,
    persist_dataset_sources,
)


def test_resolve_db_path_via_public_manager_api(tmp_path):
    db = tmp_path / "metadata.db"
    db.write_bytes(b"SQLITE")
    manager = SimpleNamespace(metadata_db_path=lambda: db)
    assert _resolve_db_path(manager) == str(db)


def test_resolve_db_path_falls_back_to_settings(tmp_path):
    db = tmp_path / "metadata.db"
    db.write_bytes(b"SQLITE")
    settings = SimpleNamespace(metadata_db_path=lambda: db)
    manager = SimpleNamespace(settings=settings)
    assert _resolve_db_path(manager) == str(db)


def test_resolve_db_path_falls_back_to_metadata_repository(tmp_path):
    db = tmp_path / "metadata.db"
    db.write_bytes(b"SQLITE")
    repo = SimpleNamespace(db_path=db)
    manager = SimpleNamespace(metadata_repository=repo)
    assert _resolve_db_path(manager) == str(db)


def test_resolve_db_path_returns_none_when_absent():
    assert _resolve_db_path(SimpleNamespace()) is None


def test_persist_dataset_sources_raises_without_db(tmp_path):
    """INF-DS-001 must surface (not silently pass) when no DB can be resolved."""
    manager = SimpleNamespace()
    with pytest.raises(DatasetSourceError):
        persist_dataset_sources(manager, tmp_path / "out")


def test_persist_dataset_sources_writes_all_sources(tmp_path):
    db = tmp_path / "state" / "metadata.db"
    db.parent.mkdir()
    db.write_bytes(b"SQLITE")

    record = SimpleNamespace(
        to_dict=lambda: {"lon": 74.8, "lat": 13.0, "village": "Udupi",
                         "district": "Udupi", "taluk": "Udupi"}
    )
    spatial_index = SimpleNamespace(records=lambda: [record])
    availability = SimpleNamespace(
        years=[2018, 2019],
        per_year={"2018": 12, "2019": 10},
        dataset_version="2.0.0",
    )
    manager = SimpleNamespace(
        metadata_repository=SimpleNamespace(db_path=db),
        spatial_index=spatial_index,
        get_historical_context=lambda **_: availability,
    )

    out = tmp_path / "out"
    sources = persist_dataset_sources(manager, out)
    assert sources.metadata_db.exists()
    assert sources.location_index.exists()
    assert sources.historical_context.exists()
    assert "village" in str(
        _read_parquet(sources.location_index).columns
    )


def _read_parquet(path):
    import pandas as pd

    return pd.read_parquet(path)
