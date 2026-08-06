"""TrainingProfiler — lightweight, dependency-optional epoch profiler.

Additive, standalone module: import and wrap your existing training loop's
steps with the context managers below. Does not alter trainer.py, losses,
metrics, or any model code — it only measures.

Usage::

    profiler = TrainingProfiler(feature_store=store)
    for batch in loader:
        with profiler.time("data_loading"):
            batch = next(iterator)
        with profiler.time("forward"):
            out = model(batch)
        with profiler.time("backward"):
            loss.backward()
    report = profiler.report()
    profiler.write_report("training/artifacts/training/profiling_report.json")
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class TrainingProfiler:
    """Accumulates wall-clock timings + optional GPU/CPU/memory samples."""

    def __init__(self, feature_store: Any | None = None) -> None:
        self.feature_store = feature_store
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._samples: list[dict[str, Any]] = []
        self._start = time.time()

    @contextmanager
    def time(self, label: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._durations[label].append(time.perf_counter() - t0)

    def sample_system(self) -> None:
        """Snapshot CPU/GPU utilization + memory. Silently skips metrics
        whose optional dependency (psutil / torch+CUDA) isn't installed.
        """
        snapshot: dict[str, Any] = {"t": time.time() - self._start}
        try:
            import psutil
            snapshot["cpu_percent"] = psutil.cpu_percent(interval=None)
            snapshot["ram_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                snapshot["gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1e6
                snapshot["gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1e6
                if hasattr(torch.cuda, "utilization"):
                    snapshot["gpu_utilization_percent"] = torch.cuda.utilization()
        except ImportError:
            pass
        self._samples.append(snapshot)

    def _stats(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {"count": 0, "total_s": 0.0, "mean_s": 0.0}
        return {
            "count": len(values),
            "total_s": round(sum(values), 4),
            "mean_s": round(sum(values) / len(values), 6),
            "max_s": round(max(values), 6),
            "min_s": round(min(values), 6),
        }

    def report(self) -> dict[str, Any]:
        timings = {label: self._stats(v) for label, v in self._durations.items()}
        cache_stats = self.feature_store.stats() if self.feature_store is not None else None

        total_data = sum(self._durations.get("data_loading", []))
        total_fwd = sum(self._durations.get("forward", []))
        total_bwd = sum(self._durations.get("backward", []))
        compute = total_fwd + total_bwd
        io_bottleneck_ratio = round(total_data / compute, 3) if compute else None

        return {
            "elapsed_s": round(time.time() - self._start, 3),
            "timings": timings,
            "io_bottleneck_ratio": io_bottleneck_ratio,  # >1 means data loading dominates compute
            "cache": cache_stats,
            "system_samples": self._samples[-50:],  # keep report bounded
        }

    def write_report(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.report(), indent=2))
