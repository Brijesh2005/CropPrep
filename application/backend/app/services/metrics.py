"""In-memory metrics registry (request counts, latency, by-path)."""

from __future__ import annotations

import time
from typing import Any


class MetricsRegistry:
    """Thread-safe aggregate of request metrics for the monitoring module."""

    def __init__(self) -> None:
        self._requests = 0
        self._total_ms = 0.0
        self._errors = 0
        self._by_path: dict[str, dict[str, Any]] = {}
        self._started_at = time.time()

    def record(self, path: str, status: int, duration_ms: float) -> None:
        self._requests += 1
        self._total_ms += duration_ms
        if status >= 500:
            self._errors += 1
        entry = self._by_path.setdefault(
            path, {"requests": 0, "total_ms": 0.0, "errors": 0}
        )
        entry["requests"] += 1
        entry["total_ms"] += duration_ms
        if status >= 500:
            entry["errors"] += 1

    def snapshot(self) -> dict[str, Any]:
        uptime_s = max(0.0, time.time() - self._started_at)
        return {
            "requests": self._requests,
            "errors": self._errors,
            "avg_latency_ms": round(self._total_ms / self._requests, 3) if self._requests else 0.0,
            "requests_per_second": round(self._requests / uptime_s, 3) if uptime_s else 0.0,
            "uptime_seconds": round(uptime_s, 1),
            "by_path": {
                path: {
                    "requests": e["requests"],
                    "avg_ms": round(e["total_ms"] / e["requests"], 3) if e["requests"] else 0.0,
                    "errors": e["errors"],
                }
                for path, e in self._by_path.items()
            },
        }
