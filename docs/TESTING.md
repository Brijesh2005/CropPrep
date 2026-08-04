# Testing

CropFusion uses pytest (Python) and Vitest + Testing Library (frontend).

## Running the suites

```bash
# Everything (Python) from the repository root
pytest

# Per-package (recommended in CI - each package configures its own path)
cd services/dataset_manager && pytest -q
cd services/spatial_alignment && pytest -q
cd ai/preprocessing && pytest -q
cd ai/models && pytest -q
cd ai/training && pytest -q
cd ai/explainability && pytest -q
cd quality && pytest -q
cd backend && pytest app/tests -q
cd mlops && pytest -q
```

Frontend:

```bash
cd frontend
npm test -- --run      # one-shot
npm run test:coverage
```

## Coverage

```bash
pytest --cov=ai --cov=services --cov=quality --cov=backend/app \
       --cov-report=term-missing
```

The CI gate enforces a minimum of **80%** overall coverage.

## Markers

Defined in `pytest.ini`. Highlights:

| Marker | Meaning |
|---|---|
| `unit` | fast, isolated, no I/O |
| `integration` | multi-component (DB, inference pipeline) |
| `backend`, `ai`, `gis`, `api` | component scoping |
| `drift`, `fairness`, `monitoring`, `optimization` | quality package |
| `security` | OWASP-focused tests |
| `performance` | pytest-benchmark suites |
| `slow` | long-running; excluded from default runs |

Use `pytest -m "not slow"` to skip long tests.

## Writing tests

- **Backend:** extend `backend/app/tests/`; reuse the shared fixtures in
  `conftest.py` (`app`, `client`, `auth_headers`, `client_with_fake_engine`).
  Tests use in-memory SQLite and stubbed inference by default.
- **Quality:** extend `quality/<area>/tests/`; synthetic data is preferred.
- **MLOps:** see `mlops/tests/test_mlops.py` (registry lifecycle, gates,
  experiments, reports).

## Performance & benchmarks

- pytest-benchmark fixtures guard hot paths.
- Inference optimisation is benchmarked by `quality/optimization`
  (`OptimizationBenchmark`) - see `research/BENCHMARKS.md`.
- Promoted models must pass the latency regression gate
  (`MLOPS_MAX_LATENCY_REGRESSION_PCT`).

## CI

`.github/workflows/ci.yml` runs the matrix (per-package suites) plus lint,
coverage, frontend tests and compose validation on every push/PR. Security
scans run in `.github/workflows/security.yml`.
