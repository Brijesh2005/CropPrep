"""Tests for the SQLite-backed cache manager."""

from __future__ import annotations

import time
from pathlib import Path

from training.dataset_manager.cache_manager import CacheManager
from training.dataset_manager.config import CacheConfig


def test_set_get_roundtrip(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("key:1", {"a": [1, 2, 3], "b": "x"})
    assert cache.get("key:1") == {"a": [1, 2, 3], "b": "x"}


def test_miss_returns_none(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    assert cache.get("nope") is None


def test_ttl_expiry(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("k", "v", ttl_seconds=0)  # already expired
    assert cache.get("k") is None


def test_ttl_negative_disables_expiry(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("k", "v", ttl_seconds=-1)
    assert cache.get("k") == "v"


def test_delete(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("k", "v")
    assert cache.delete("k") is True
    assert cache.delete("k") is False
    assert cache.get("k") is None


def test_delete_prefix(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("scan:/a", 1)
    cache.set("scan:/b", 2)
    cache.set("other:1", 3)
    removed = cache.delete_prefix("scan:")
    assert removed == 2
    assert cache.get("other:1") == 3


def test_clear_and_size(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    for i in range(5):
        cache.set(f"k{i}", i)
    assert cache.size() == 5
    assert cache.clear() == 5
    assert cache.size() == 0


def test_prune_removes_expired(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db")
    cache.set("expired", "v", ttl_seconds=0)
    cache.set("live", "v", ttl_seconds=60)
    pruned = cache.prune()
    assert pruned >= 1
    assert cache.get("live") == "v"


def test_disabled_cache_is_in_memory():
    cache = CacheManager(enabled=False)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert cache.size() == 1


def test_max_entries_eviction(tmp_path: Path):
    cache = CacheManager(tmp_path / "cache.db", config=CacheConfig(max_entries=3))
    for i in range(10):
        cache.set(f"k{i}", i)
    assert cache.size() <= 3
