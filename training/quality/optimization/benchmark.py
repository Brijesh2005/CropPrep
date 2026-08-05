"""Optimization benchmark — compare runtimes on a real model.

Measures latency percentiles, throughput and memory for each enabled
``OptimizedRuntime`` mode and writes a machine-readable JSON + a self-contained
HTML report consumable by :mod:`training.quality.monitoring.dashboard`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .runtime import MODE_NAMES, OptimizedRuntime


@dataclass
class VariantResult:
    """Benchmark output for a single runtime mode."""

    name: str
    latency: dict[str, float] = field(default_factory=dict)
    throughput: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    params: int = 0
    device: str = ""


class OptimizationBenchmark:
    """Benchmark eager vs optimised inference runtimes."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        batch_size: int = 1,
        iterations: int = 30,
        warmup: int = 5,
        modes: tuple[str, ...] = ("eager", "autocast", "compiled", "onnx"),
    ) -> None:
        self.model = model.eval()
        self.batch = model.sample_batch(batch_size=batch_size)
        self.batch_size = batch_size
        self.iterations = iterations
        self.warmup = warmup
        self.modes = tuple(m for m in modes if m in MODE_NAMES)

    # ------------------------------------------------------------------ #

    def run(self) -> dict[str, Any]:
        """Run every mode and return a serialisable report."""
        variants: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        eager: dict[str, Any] | None = None

        for mode in self.modes:
            try:
                runtime = OptimizedRuntime(
                    self.model,
                    mode=mode,
                    sample_batch=self.batch if mode == "onnx" else None,
                )
                result = self._benchmark_mode(runtime, mode)
                variants.append(asdict(result))
                if mode == "eager":
                    eager = result.latency
            except Exception as exc:  # pragma: no cover - env-dependent
                errors[mode] = str(exc)

        if eager is None:
            raise RuntimeError("eager baseline failed; cannot benchmark")

        device = str(next(self.model.parameters()).device)
        return {
            "model": getattr(self.model.config, "name", "CropFusionModel"),
            "params": sum(p.numel() for p in self.model.parameters()),
            "device": device,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "variants": variants,
            "errors": errors,
            "speedup": _speedups(variants, eager),
        }

    # ------------------------------------------------------------------ #

    def _benchmark_mode(self, runtime: OptimizedRuntime, mode: str) -> VariantResult:
        for _ in range(self.warmup):
            runtime.predict(self.batch)

        latencies: list[float] = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            runtime.predict(self.batch)
            latencies.append((time.perf_counter() - start) * 1000.0)

        latencies = np.asarray(latencies)
        total_s = latencies.sum() / 1000.0
        throughput = {
            "samples_per_second": (
                self.iterations * self.batch_size / total_s if total_s > 0 else 0.0
            ),
            "batches_per_second": self.iterations / total_s if total_s > 0 else 0.0,
        }
        return VariantResult(
            name=mode,
            latency={
                "mean_ms": float(latencies.mean()),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "min_ms": float(latencies.min()),
            },
            throughput=throughput,
            memory=_memory_mb(),
            params=runtime.num_parameters(),
            device=runtime.device_name(),
        )

    def write_report(self, out_dir: str | Path) -> dict[str, Path]:
        """Run the benchmark and persist JSON + HTML reports."""
        report = self.run()
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "benchmark_report.json"
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        html_path = out / "benchmark_report.html"
        html_path.write_text(_render_html(report), encoding="utf-8")
        return {"json": json_path, "html": html_path}


def _speedups(
    variants: list[dict[str, Any]], eager: Mapping[str, float]
) -> dict[str, float]:
    eager_mean = eager.get("mean_ms", 0.0)
    return {
        v["name"]: round(eager_mean / max(v["latency"].get("mean_ms", 0.0), 1e-9), 3)
        for v in variants
    }


def _memory_mb() -> dict[str, float]:
    try:
        import psutil

        return {"cpu_rss_mb": float(psutil.Process().memory_info().rss / (1024**2))}
    except Exception:
        return {"cpu_rss_mb": 0.0}


def _render_html(report: dict[str, Any]) -> str:
    rows = []
    for variant in report.get("variants", []):
        rows.append(
            f"<tr><td>{variant['name']}</td>"
            f"<td>{variant['latency'].get('mean_ms', 0):.2f}</td>"
            f"<td>{variant['latency'].get('p50_ms', 0):.2f}</td>"
            f"<td>{variant['latency'].get('p95_ms', 0):.2f}</td>"
            f"<td>{variant['latency'].get('p99_ms', 0):.2f}</td>"
            f"<td>{variant['throughput'].get('samples_per_second', 0):.1f}</td></tr>"
        )
    speedup_cells = "".join(
        f"<li>{name}: {value}x</li>" for name, value in report.get("speedup", {}).items()
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>CropFusion Optimization Benchmark</title>
<style>body{{font-family:-apple-system,'Segoe UI',sans-serif;margin:2rem auto;max-width:900px;}}
table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #e2e8f0;padding:.45rem .6rem;text-align:left;font-size:.85rem;}}
th{{background:#f7fafc;}}</style></head><body>
<h1>CropFusion Optimization Benchmark</h1>
<p>model={report['model']} &middot; params={report['params']:,} &middot; device={report['device']} &middot; batch={report['batch_size']} &middot; iters={report['iterations']}</p>
<h2>Speedup vs eager</h2><ul>{speedup_cells or '<li>none</li>'}</ul>
<h2>Latency (ms) and throughput</h2>
<table><tr><th>Mode</th><th>Mean</th><th>p50</th><th>p95</th><th>p99</th><th>samples/s</th></tr>
{''.join(rows)}</table>
</body></html>"""
