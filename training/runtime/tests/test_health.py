"""Health reporting + memory monitor tests (Phase R6)."""

from __future__ import annotations

import pytest

from training.runtime import RuntimeConfig
from training.runtime.cache import RuntimeCache
from training.runtime.exceptions import MemoryLimitError
from training.runtime.health import (
    STATUS_LOADING,
    STATUS_NOT_READY,
    STATUS_READY,
    HealthReport,
    MemoryInfo,
    MemoryMonitor,
)


def test_memory_monitor_snapshot():
    monitor = MemoryMonitor()
    info = monitor.snapshot()
    assert info.rss_mb > 0
    assert info.used_mb >= 0
    assert info.limit_mb is None


def test_memory_monitor_respects_interval():
    monitor = MemoryMonitor(
        RuntimeConfig(memory={"check_interval_seconds": 3600}).memory
    )
    first = monitor.snapshot()
    second = monitor.snapshot()
    assert first is second


def test_memory_monitor_hard_limit():
    monitor = MemoryMonitor(RuntimeConfig(memory={"limit_mb": 0}).memory)
    with pytest.raises(MemoryLimitError):
        monitor.check()


def test_memory_monitor_soft_limit_evicts_cache():
    cache = RuntimeCache(max_bytes=0)
    monitor = MemoryMonitor(
        RuntimeConfig(memory={"soft_limit_mb": 0}).memory, cache=cache
    )
    monitor.check()


def test_memory_monitor_no_limit_passes():
    monitor = MemoryMonitor()
    info = monitor.check()
    assert info.rss_mb > 0


def test_memory_info_to_dict():
    info = MemoryInfo(
        rss_mb=100.0,
        used_mb=80.0,
        available_mb=1000.0,
        percent=10.0,
        cache_bytes=5,
        limit_mb=None,
    )
    d = info.to_dict()
    assert d["rss_mb"] == 100.0
    assert d["cache_bytes"] == 5
    assert d["limit_mb"] is None


def test_health_report_ready_to_dict():
    report = HealthReport(status=STATUS_READY, ready=True, version="1.0.0")
    d = report.to_dict()
    assert d["status"] == "ready"
    assert d["ready"] is True
    assert d["version"] == "1.0.0"
    assert "memory" in d
    assert "checks" in d


def test_health_report_default_state():
    report = HealthReport()
    assert report.status == STATUS_NOT_READY
    assert report.ready is False
    assert report.uptime_seconds == 0.0
    assert report.startup_time_ms is None


def test_health_report_loading_state():
    report = HealthReport(status=STATUS_LOADING)
    assert report.status == "loading"
