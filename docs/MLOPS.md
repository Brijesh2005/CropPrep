# MLOps

CropFusion ships a self-contained MLOps toolkit (`mlops/`) for model
governance: registry, promotion gates, experiment tracking, scheduled quality
monitoring and release reporting. Everything is file-based by default (no
external platform required) and runs either from the CLI or the `admin`
container.

## Installation

The `mlops` package is part of the repository (importable from the repo root)
and provides the `cropfusion-mlops` console script when the umbrella package is
installed. It can also run as `python -m mlops`.

## Registry lifecycle

```
draft --(gates pass)--> staging --(approval)--> production --(issues)--> rollback
                          |                           |
                          +----(superseded)----> archived
```

```bash
# Register a trained checkpoint as a draft
cropfusion-mlops register yieldnet 1.2.0 model.pt --accuracy 0.87 --loss 0.31 \
    --git-commit abc1234 --notes "seasonal retrain"

# Inspect
cropfusion-mlops list --status production
cropfusion-mlops info yieldnet 1.2.0

# Promote (runs gates; fails closed)
cropfusion-mlops promote yieldnet 1.2.0 --target production --accuracy 0.87

# Roll back on production issues
cropfusion-mlops rollback yieldnet 1.0.0
```

Promotion to `production` demotes the incumbent to `staging` and archives
superseded versions (keep `MLOPS_KEEP_PRODUCTION_VERSIONS`, default 3).

## Promotion gates

Gates are defined in `mlops/gates.py` and stored on the model manifest:

1. **Metrics** — accuracy >= `MLOPS_MIN_ACCURACY` (0.80) and no regression vs
   the incumbent.
2. **Regression** — latency regression <= `MLOPS_MAX_LATENCY_REGRESSION_PCT`
   (10%).
3. **Drift** — reference vs current comparison is not high-severity.
4. **Fairness** — no protected group "at risk".

```bash
# Run gates without promoting
cropfusion-mlops gate yieldnet 1.2.0 --accuracy 0.87 --incumbent 1.1.0
```

## Experiment tracking

```bash
cropfusion-mlops experiment yieldnet --accuracy 0.87 --loss 0.31 \
    --dataset-version 2024.1 --notes "lr 1e-3"
cropfusion-mlops experiments-list --model yieldnet
```

Runs are stored append-only in `experiments/runs.jsonl` (JSONL).

## Scheduler (admin container)

```bash
docker compose --profile mlops up -d
# or standalone:
python -m mlops.scheduler --once
python -m mlops.scheduler            # loops every MLOPS_INTERVAL_SECONDS
```

Each cycle:

1. Runs the drift battery over `MLOPS_DRIFT_REFERENCE_DATA` and writes reports.
2. Runs the fairness evaluator if `reports/fairness_inputs.json` is present.
3. Verifies registry invariants.

Reports land under `reports/` (`drift/`, `fairness/`, `benchmarks/`,
`releases/`).

## Release reporting

```bash
cropfusion-mlops benchmark ...                 # (via quality tooling)
python -m mlops.reports                        # programmatic API
```

Promotion writes a release report (`reports/releases/<model>-<version>-*.md`)
capturing metrics, hyperparameters, provenance and gate results - suitable for
inclusion in release packages and audits.

## Configuration

All settings are environment-driven (`MLOPS_` prefix):

| Variable | Default | Purpose |
|---|---|---|
| `MLOPS_REGISTRY_DIR` | `models/registry` | model store root |
| `MLOPS_REPORTS_DIR` | `reports` | report output |
| `MLOPS_EXPERIMENTS_DIR` | `experiments` | experiment log |
| `MLOPS_INTERVAL_SECONDS` | `3600` | scheduler interval |
| `MLOPS_MIN_ACCURACY` | `0.80` | metrics gate threshold |
| `MLOPS_MAX_LATENCY_REGRESSION_PCT` | `10.0` | regression gate threshold |
| `MLOPS_KEEP_PRODUCTION_VERSIONS` | `3` | production retention |
| `MLOPS_DRIFT_REFERENCE_DATA` | — | reference dataset for drift |
