# Sample Quality Reports — R2.3 QC

`training/quality/samples/` is a small QC layer that answers **"how healthy is
the generated corpus?"** with one JSON document, without re-running STAM.

## What it produces

`SampleQualityReport.from_corpus(corpus)` aggregates the resolved corpus into:

| Key | Contents |
| --- | --- |
| `total` / `status` | accepted / rejected / error counts |
| `acceptance_rate` | `accepted / total` |
| `quality_score` | count / min / max / mean / median of accepted quality scores |
| `issue_codes` | histogram of STAM `QualityIssue.code` values across accepted observations (e.g. `ST-Q-*`, `ST-IMAGE-*`, `ST-PAIR-*`, `ST-TEMP-*`) |
| `severity_counts` | info / warning / error / critical histogram |
| `by_crop` / `by_year` / `by_season` | per-key `{total, accepted, rate}` |
| `top_error_codes` | the most common `error.code` among failed cells |

Because the corpus keeps rejected **and** error cells (default), the report can
separate "data simply not available" (rejected) from "generation crashed here"
(error) — the two failures need different fixes.

## Usage

```python
from training.quality.samples import build_report

report = build_report(corpus, output_dir="data/out/qc")
# writes data/out/qc/sample_quality_report.json
```

`build_report` also accepts a plain list of `ResolvedSample` objects.

## Reading the report

- Low `acceptance_rate` + high rejected → missing image pairs / seasons for
  those cells (check `by_year`, `by_season`).
- Non-empty `top_error_codes` → a real defect (e.g. `ST-RESOLVE-999`); fix the
  resolver or the catalog before scaling up the grid.
- `issue_codes` dominated by `critical` severities → accepted samples that may
  still be unusable; raise `min_quality_score` in the resolver and regenerate.

## Error codes

| Code | Class | Meaning |
| --- | --- | --- |
| `SQ-ERR-001` | `SampleQualityError` | Reporting failed (e.g. empty corpus) |
| `SQ-CONFIG-001` | `SampleQualityConfigError` | Invalid QC configuration |

Loggers live under `cropfusion.quality.samples.*`.

See `training/quality/samples/tests/test_report.py` (9 tests) for the contract.
