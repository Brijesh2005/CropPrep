"""Health reporting (Phase R6).

:class:`MemoryMonitor` reads the runtime's process RSS via ``psutil`` (with a
Windows ``ctypes`` fallback) so memory limits can be enforced and reported.
:class:`HealthReport` aggregates the runtime readiness signals into one
snapshot: model / preprocess / metadata readiness, release version, memory,
startup time and uptime.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import RuntimeCache
from .config import MemoryConfig
from .exceptions import HealthError, MemoryLimitError

#: Aggregate readiness states reported by the runtime.
STATUS_READY = "ready"
STATUS_DEGRADED = "degraded"
STATUS_LOADING = "loading"
STATUS_NOT_READY = "not_ready"


@dataclass
class MemoryInfo:
    """Memory snapshot (process RSS + cache usage)."""

    rss_mb: float = 0.0
    used_mb: float = 0.0
    available_mb: float | None = None
    percent: float | None = None
    cache_bytes: int = 0
    limit_mb: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_mb": self.rss_mb,
            "used_mb": self.used_mb,
            "available_mb": self.available_mb,
            "percent": self.percent,
            "cache_bytes": self.cache_bytes,
            "limit_mb": self.limit_mb,
        }


class MemoryMonitor:
    """Monitor process RSS and enforce the configured memory limits.

    Args:
        config: The runtime memory configuration.
        cache: Optional shared :class:`RuntimeCache` whose entries are evicted
            when the soft limit is exceeded.
    """

    def __init__(
        self, config: MemoryConfig | None = None, cache: RuntimeCache | None = None
    ) -> None:
        self.config = config or MemoryConfig()
        self.cache = cache
        self._lock = threading.RLock()
        self._last_snapshot: MemoryInfo | None = None
        self._last_checked = 0.0

    def snapshot(self) -> MemoryInfo:
        """A fresh (or cached-interval) memory snapshot."""
        with self._lock:
            if (
                self._last_snapshot is not None
                and time.monotonic() - self._last_checked
                < self.config.check_interval_seconds
            ):
                return self._last_snapshot
            rss = _process_rss_mb()
            cache_bytes = self.cache.info()["bytes"] if self.cache else 0
            info = MemoryInfo(
                rss_mb=rss,
                used_mb=_process_used_mb(),
                available_mb=_system_available_mb(),
                percent=_system_percent(),
                cache_bytes=cache_bytes,
                limit_mb=self.config.limit_mb,
            )
            self._last_snapshot = info
            self._last_checked = time.monotonic()
            return info

    def check(self) -> MemoryInfo:
        """Enforce limits: evict cache above soft, raise above hard.

        Raises:
            MemoryLimitError: When the hard ``limit_mb`` is exceeded.
        """
        info = self.snapshot()
        soft = self.config.soft_limit_mb
        hard = self.config.limit_mb
        if soft is not None and info.rss_mb > soft and self.cache is not None:
            self.cache.evict()
            info = self.snapshot()
        if hard is not None and info.rss_mb > hard:
            raise MemoryLimitError(
                f"process RSS {info.rss_mb:.1f} MB exceeds the hard limit "
                f"{hard} MB",
                detail=info.to_dict(),
            )
        return info


@dataclass
class HealthReport:
    """Full readiness snapshot for the runtime.

    Attributes:
        status: Aggregate state (``ready`` / ``degraded`` / ``loading`` /
            ``not_ready``).
        release_ready: Whether a release is active and valid.
        model_ready: Whether the model is loaded and warm.
        preprocess_ready: Whether the pipelines are loaded.
        metadata_ready: Whether the metadata artefacts are loaded.
        version: Active release version.
        model_version: Version of the loaded model.
        backend: Model backend in use.
        memory: :class:`MemoryInfo` snapshot.
        startup_time_ms: Milliseconds taken to reach readiness.
        uptime_seconds: Seconds since the runtime started.
        warmup_ok: Whether warm-up completed.
        checks: Per-component readiness booleans.
    """

    status: str = STATUS_NOT_READY
    ready: bool = False
    release_ready: bool = False
    model_ready: bool = False
    preprocess_ready: bool = False
    metadata_ready: bool = False
    version: str | None = None
    model_version: str | None = None
    backend: str | None = None
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    startup_time_ms: float | None = None
    uptime_seconds: float = 0.0
    warmup_ok: bool = False
    checks: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "release_ready": self.release_ready,
            "model_ready": self.model_ready,
            "preprocess_ready": self.preprocess_ready,
            "metadata_ready": self.metadata_ready,
            "version": self.version,
            "model_version": self.model_version,
            "backend": self.backend,
            "memory": self.memory.to_dict(),
            "startup_time_ms": self.startup_time_ms,
            "uptime_seconds": self.uptime_seconds,
            "warmup_ok": self.warmup_ok,
            "checks": dict(self.checks),
        }


def _process_rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except ImportError:  # pragma: no cover - psutil absent
        return _ctypes_rss_mb()


def _ctypes_rss_mb() -> float:
    try:
        import ctypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        psapi = ctypes.windll.psapi  # type: ignore[attr-defined]
        psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
            ctypes.byref(counters),
            counters.cb,
        )
        return float(counters.WorkingSetSize) / (1024 * 1024)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _process_used_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().uss) / (1024 * 1024)
    except Exception:  # pragma: no cover - defensive
        return 0.0


def _system_available_mb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024 * 1024)
    except Exception:  # pragma: no cover - defensive
        return None


def _system_percent() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().percent)
    except Exception:  # pragma: no cover - defensive
        return None
