"""R2.1 Training Cache tests: buckets, TTL, eviction and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.kaggle.cache import SECTIONS, TrainingCache


@pytest.fixture()
def cache(tmp_path: Path) -> TrainingCache:
    return TrainingCache(tmp_path / "cache")


def test_sections_valid(cache: TrainingCache) -> None:
    assert set(SECTIONS) == {
        "metadata",
        "preprocessing",
        "image_metadata",
        "statistics",
        "validation",
    }


def test_set_get(cache: TrainingCache) -> None:
    cache.set("metadata", "k1", {"a": 1})
    assert cache.get("metadata", "k1") == {"a": 1}
    assert cache.has("metadata", "k1") is True


def test_get_missing(cache: TrainingCache) -> None:
    assert cache.get("metadata", "nope") is None
    assert cache.has("metadata", "nope") is False


def test_ttl_expiry(cache: TrainingCache) -> None:
    cache.set("statistics", "s", 42, ttl_seconds=-10)
    assert cache.get("statistics", "s") is None


def test_unknown_section_rejected(cache: TrainingCache) -> None:
    with pytest.raises(ValueError):
        cache.set("bogus", "k", 1)


def test_delete(cache: TrainingCache) -> None:
    cache.set("validation", "k", 1)
    assert cache.delete("validation", "k") is True
    assert cache.delete("validation", "k") is False
    assert cache.get("validation", "k") is None


def test_clear_section(cache: TrainingCache) -> None:
    cache.set("metadata", "a", 1)
    cache.set("metadata", "b", 2)
    cache.set("validation", "c", 3)
    assert cache.clear("metadata") == 2
    assert cache.stats()["sections"]["metadata"] == 0
    assert cache.stats()["sections"]["validation"] == 1


def test_clear_all(cache: TrainingCache) -> None:
    for section in SECTIONS:
        cache.set(section, "k", 1)
    assert cache.clear() == len(SECTIONS)
    assert cache.stats()["total"] == 0


def test_lru_eviction(tmp_path: Path) -> None:
    cache = TrainingCache(tmp_path / "c", max_entries=2)
    cache.set("metadata", "a", 1)
    cache.set("metadata", "b", 2)
    cache.set("metadata", "c", 3)  # evicts "a"
    assert cache.get("metadata", "a") is None
    assert cache.get("metadata", "b") == 2
    assert cache.get("metadata", "c") == 3


def test_persistence(tmp_path: Path) -> None:
    path = tmp_path / "cache"
    cache = TrainingCache(path)
    cache.set("image_metadata", "img", {"size": 10})
    assert (path / "image_metadata.json").exists()

    reloaded = TrainingCache(path)
    assert reloaded.get("image_metadata", "img") == {"size": 10}


def test_stats(cache: TrainingCache) -> None:
    cache.set("preprocessing", "k", 1)
    stats = cache.stats()
    assert stats["total"] == 1
    assert stats["sections"]["preprocessing"] == 1
    assert stats["cache_dir"] == str(cache.cache_dir)


def test_json_round_trip_types(cache: TrainingCache) -> None:
    payload = {"list": [1, 2], "nested": {"x": True}, "n": 1.5}
    cache.set("statistics", "stats", payload)
    assert cache.get("statistics", "stats") == payload
    raw = json.loads((cache.cache_dir / "statistics.json").read_text(encoding="utf-8"))
    assert raw["entries"]["stats"]["value"] == payload
