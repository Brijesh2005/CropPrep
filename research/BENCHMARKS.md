# CropFusion Benchmarks

Benchmark methodology, tooling and results for inference optimisation.

## Tooling

- `ai/training` ships `Benchmark` / `BenchmarkReport` covering training /
  validation / inference speed, GPU/CPU memory and model size.
- `quality/optimization` ships `OptimizationBenchmark` comparing execution
  modes: **eager**, **torchscript (compiled)**, **onnx fp32** and optional
  **onnx int8** quantization.
- `pytest-benchmark` fixtures are used for regression-guarding hot paths
  (`tests` marker `performance`).
- The MLOps promotion gates require a latency **regression gate** before a
  candidate model can replace the incumbent
  (`MLOPS_MAX_LATENCY_REGRESSION_PCT`, default 10%).

## Benchmark workflow

1. Build the optimized runtimes: `python -m quality.optimization.benchmark ...`
2. Persist results with `cropfusion-mlops benchmark` -> `reports/benchmarks/`.
3. Compare against the incumbent; the promotion gate blocks candidates that
   regress latency beyond the configured threshold.
4. Dashboards: `quality/monitoring/grafana/cropfusion-performance.json`.

## Reported metrics

| Metric | Meaning |
|---|---|
| `mean_latency_ms` | Mean per-request inference latency |
| `p95_latency_ms` | 95th percentile latency |
| `speedup` | Latency ratio vs the eager baseline (eager = 1.0x) |
| `throughput_qps` | Requests per second |
| `model_size_mb` | Serialised model size per mode |

## Expectations

- ONNX Runtime typically yields multi-x speedups over eager torch on CPU.
- Compiled (torchscript) mode is used when a compiler backend is available and
  falls back to eager automatically otherwise.
- INT8 quantization is optional and reserved for latency-critical deployments
  where the accuracy budget permits.

## Related

- [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)
- `quality/optimization/tests/test_optimization.py`
- `docs/PHASE6_COMPLETION_REPORT.md` (evaluation + benchmark definition)
