# Observation Resolver — R2.3 Sample Generation

The **Observation Resolver** turns the raw Dataset Manager catalogue into a
training-sample corpus: a grid over *locations × years × seasons* where every
cell is resolved through STAM into an
[`AgriculturalObservation`](../../../training/stam/observation.py) with a
quality verdict.

Module: `training/stam/observation_resolver.py`.

## Why it exists

STAM resolves *one* (lat, lon) query at a time. Training needs thousands of
samples with consistent provenance. The resolver:

1. **Plans** the full grid up-front (`ObservationPlan`) — no surprises mid-run.
2. **Resolves** each cell independently, so a failure in one location never
   aborts the batch.
3. **Classifies** every outcome as `accepted`, `rejected` or `error` so QC and
   balancing can reason about *why* cells failed.
4. **Caches** to JSON so a large generation run can resume without re-running
   STAM.

## Core types

| Type | Purpose |
| --- | --- |
| `ObservationResolverConfig` | bbox / max_locations / seasons / min_quality_score / cache knobs |
| `SamplingCell` | one grid cell: location + year + season |
| `ObservationPlan` | ordered cell list + expansion metadata (`total`, `by_year`, …) |
| `ResolvedSample` | per-cell outcome: status + quality score + optional `error` dict |
| `ObservationCorpus` | the full sample set with `accepted()`, `rejected()`, `errors()`, `status_counts()` |

## Flow

```
manager.catalog  ──►  plan(years, seasons, max_locations, bbox)
                          │  (tabular years + image catalog years)
                          ▼
                    ObservationPlan (N cells)
                          │  resolve() — one STAM resolve per cell
                          ▼
                    ObservationCorpus (accepted | rejected | error)
```

## Key behaviour

- **Year inference**: when `years=[]` and `config.infer_years=True`, the plan
  intersects tabular years (from the configured table) with the image catalog
  years so the grid only contains datable cells.
- **Error isolation**: a cell that raises is captured as
  `status="error"` with `error={"code", "message"}` (ST-RESOLVE-* /
  ST-RESOLVE-999 for unexpected exceptions); the rest of the corpus proceeds.
- **Filtering on output**: `include_rejected` / `include_errors` control which
  statuses survive into the final corpus (default keeps all three for QC).
- **Caching**: `use_cache=True` persists the corpus as JSON; `load` restores it.

## Usage

```python
from training.stam import ObservationResolver, ObservationResolverConfig

resolver = ObservationResolver(
    stam,
    ObservationResolverConfig(min_quality_score=0.0, use_cache=False),
)
plan = resolver.plan(years=[2020, 2021], seasons=["Kharif"], max_locations=50)
corpus = resolver.resolve(plan)

print(corpus.status_counts())
# {'accepted': 96, 'rejected': 4, 'error': 0}
```

## Error codes

| Code | Meaning |
| --- | --- |
| `ST-RESOLVE-001` (`SampleResolutionError`) | A cell could not be resolved to an observation |
| `ST-RESOLVE-002` (`SampleCellError`) | A sampling cell is malformed (no location / no year) |
| `ST-RESOLVE-999` | Unexpected exception during cell resolution |

See `training/stam/tests/test_observation_resolver.py` for the contract (15 tests).
