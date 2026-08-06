"""Preprocess + metadata loader tests (Phase R6)."""

from __future__ import annotations

import shutil
import sqlite3

import pandas as pd
import pytest

from training.runtime import ReleaseLayout, RuntimeConfig
from training.runtime.exceptions import MetadataLoadError, PreprocessLoadError
from training.runtime.metadata_loader import MetadataLoader
from training.runtime.preprocess_loader import PreprocessLoader
from training.runtime.tests.conftest import clone_release

FEATURE_NAMES = [
    "rainfall_mm",
    "temperature",
    "soil_moisture",
    "soil_type",
]


def _cloned(release_env, tmp_path, name="clone"):
    target = tmp_path / name
    clone_release(release_env.release_path, target)
    return target


# --------------------------------------------------------------------- #
# Preprocess loader
# --------------------------------------------------------------------- #


def test_preprocess_load(release_env):
    loader = PreprocessLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    assert loader.feature_names == FEATURE_NAMES
    assert loader.num_classes == 3
    health = loader.health()
    assert health.loaded is True
    assert health.num_features == 4
    assert health.fitted is True
    assert health.config_loaded is True
    assert health.metadata_loaded is True


def test_preprocess_metadata(release_env):
    loader = PreprocessLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    meta = loader.load_metadata()
    assert meta["feature_names"] == FEATURE_NAMES
    assert meta["num_features"] == 4
    assert meta["num_classes"] == 3
    assert meta["fitted"] is True


def test_preprocess_accessors(release_env):
    loader = PreprocessLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    assert loader.feature_scalers is not None
    assert loader.label_encoder is not None
    assert loader.feature_scalers.fitted is True


def test_preprocess_missing_scalers(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "preprocess" / "feature_scalers.pkl").unlink()
    loader = PreprocessLoader(ReleaseLayout(target))
    with pytest.raises(PreprocessLoadError):
        loader.load()


def test_preprocess_not_required(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "preprocess" / "feature_scalers.pkl").unlink()
    config = RuntimeConfig(preprocess={"required": False})
    loader = PreprocessLoader(ReleaseLayout(target), config)
    loader.load()
    assert loader.feature_scalers is None
    assert loader.health().loaded is False


def test_preprocess_corrupt_pickle(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "preprocess" / "label_encoder.pkl").write_bytes(b"not a pickle")
    loader = PreprocessLoader(ReleaseLayout(target))
    with pytest.raises(PreprocessLoadError):
        loader.load()


def test_preprocess_unload(release_env):
    loader = PreprocessLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    loader.unload()
    assert loader.health().loaded is False


# --------------------------------------------------------------------- #
# Metadata loader
# --------------------------------------------------------------------- #


def test_metadata_load(release_env):
    loader = MetadataLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    health = loader.health()
    assert health.loaded is True
    assert health.db_loaded is True
    assert health.historical_loaded is True
    assert health.location_loaded is True
    assert health.feature_lookup_loaded is True
    assert health.row_counts["historical_context"] == 1
    assert health.row_counts["location_index"] == 1


def _with_real_db(release_env, tmp_path, name="clone"):
    target = _cloned(release_env, tmp_path, name)
    db_path = target / "metadata" / "metadata.db"
    db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE fields (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE zones (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO zones VALUES (1, 'kharif')")
    conn.commit()
    conn.close()
    return target


def test_metadata_with_real_db(release_env, tmp_path):
    target = _with_real_db(release_env, tmp_path)
    loader = MetadataLoader(ReleaseLayout(target))
    loader.load()
    assert loader.tables() == ["fields", "zones"]
    rows = loader.query("SELECT name FROM zones ORDER BY id")
    assert rows == [("kharif",)]


def test_metadata_readonly_query(release_env, tmp_path):
    target = _with_real_db(release_env, tmp_path)
    loader = MetadataLoader(ReleaseLayout(target))
    loader.load()
    with pytest.raises(MetadataLoadError):
        loader.query("INSERT INTO zones VALUES (2)")


def test_metadata_dataframes(release_env):
    loader = MetadataLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    historical = loader.historical()
    assert isinstance(historical, pd.DataFrame)
    assert list(historical.columns) == ["season"]
    assert historical.iloc[0]["season"] == "Kharif"

    locations = loader.locations()
    assert list(locations.columns) == ["lon", "lat"]

    lookup = loader.feature_lookup()
    assert list(lookup.columns) == [
        "feature_index",
        "feature_name",
        "feature_type",
        "feature_group",
    ]
    soil = lookup[lookup["feature_name"] == "soil_type"]
    assert len(soil) == 1
    assert soil.iloc[0]["feature_type"] == "categorical"
    assert soil.iloc[0]["feature_index"] == 3


def test_metadata_cache_reuse(release_env):
    loader = MetadataLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    first = loader.historical()
    second = loader.historical()
    assert first is second
    assert loader.cache.contains("historical_context")
    assert loader.cache.get("historical_context") is first


def test_metadata_missing_db(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "metadata" / "metadata.db").unlink()
    config = RuntimeConfig(metadata={"required": True})
    loader = MetadataLoader(ReleaseLayout(target), config)
    with pytest.raises(MetadataLoadError):
        loader.load()


def test_metadata_not_required(release_env, tmp_path):
    target = _cloned(release_env, tmp_path)
    (target / "metadata" / "metadata.db").unlink()
    config = RuntimeConfig(metadata={"required": False})
    loader = MetadataLoader(ReleaseLayout(target), config)
    loader.load()
    assert loader.health().db_loaded is False
    assert loader.health().loaded is True


def test_metadata_unload(release_env):
    loader = MetadataLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    loader.unload()
    assert loader.connection is None
    assert loader.health().loaded is False


def test_metadata_load_config(release_env):
    loader = MetadataLoader(ReleaseLayout(release_env.release_path))
    loader.load()
    cfg = loader.load_config()
    assert cfg["required"] is True
    assert cfg["feature_lookup_required"] is True
    assert "metadata/metadata.db" in cfg["artifacts"]
