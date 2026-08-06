"""Runtime cache tests (Phase R6)."""

from __future__ import annotations

import pytest

from training.runtime.cache import RuntimeCache, _estimate_size


def test_get_miss_counts():
    cache = RuntimeCache()
    assert cache.get("missing") is None
    assert cache.info()["misses"] == 1


def test_set_get():
    cache = RuntimeCache()
    cache.set("k", {"a": 1})
    assert cache.get("k") == {"a": 1}
    info = cache.info()
    assert info["entries"] == 1
    assert info["hits"] == 1
    assert info["bytes"] > 0


def test_overwrite_replaces_entry_and_bytes():
    cache = RuntimeCache()
    cache.set("k", "a" * 500)
    first = cache.info()["bytes"]
    cache.set("k", "b")
    assert cache.info()["entries"] == 1
    assert cache.info()["bytes"] < first


def test_byte_eviction():
    cache = RuntimeCache(max_bytes=400)
    cache.set("a", "x" * 300)
    cache.set("b", "y" * 300)
    assert cache.get("a") is None
    assert cache.get("b") == "y" * 300


def test_entry_count_eviction():
    cache = RuntimeCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") is None
    assert cache.contains("b")
    assert cache.contains("c")


def test_ttl_expiry():
    import time

    cache = RuntimeCache(ttl_seconds=0.05)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.1)
    assert cache.get("k") is None


def test_zero_budget_retains_nothing():
    cache = RuntimeCache(max_bytes=0, max_entries=0)
    cache.set("a", "x" * 100)
    assert cache.info()["entries"] == 0
    assert cache.contains("a") is False


def test_clear():
    cache = RuntimeCache()
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.info()["entries"] == 0
    assert cache.info()["bytes"] == 0


def test_evict_all_returns_freed_bytes():
    cache = RuntimeCache()
    cache.set("a", 1)
    cache.set("b", 2)
    freed = cache.evict()
    assert freed > 0
    assert cache.info()["entries"] == 0


def test_evict_empty_returns_zero():
    cache = RuntimeCache()
    assert cache.evict() == 0


def test_estimate_size_dataframe():
    import pandas as pd

    df = pd.DataFrame({"x": [1, 2, 3]})
    assert _estimate_size(df) > 0


def test_estimate_size_fallback_never_negative():
    assert _estimate_size(object()) >= 0


def test_info_reporting():
    cache = RuntimeCache(max_bytes=1000, max_entries=4)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.get("missing")
    info = cache.info()
    assert info["max_bytes"] == 1000
    assert info["max_entries"] == 4
    assert info["hits"] == 1
    assert info["misses"] == 1
    assert info["entries"] == 2
