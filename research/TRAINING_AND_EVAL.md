# CropFusion Training & Evaluation

How models are trained, evaluated, versioned and promoted into production.

## Training protocol (ai/training)

1. **Preprocess** via `ai/preprocessing` into multimodal batches.
2. **Train** the Phase 5 architecture with the Phase 6 loop: optimisers,
   LR schedulers, early stopping, checkpointing.
3. **Evaluate** on the held-out validation set with the Phase 6 `Evaluator`
   (metrics, latency, memory, parameters, multi-task score).
4. **Record** the run with `cropfusion-mlops experiment` (append-only JSONL in
   `experiments/runs.jsonl`).
5. **Register** the checkpoint with `cropfusion-mlops register` (draft).

## Evaluation metrics

- **Classification:** accuracy, precision, recall, F1 (macro/micro/weighted),
  ROC-AUC, top-K, confusion matrix.
- **Regression:** MSE, RMSE, MAE, R², MAPE.
- **Combined:** `multi_task = 0.5*crop_acc + 0.5*max(0, 1 - nRMSE)`.
- **System:** latency (mean/p50/p95/p99), throughput, params, size, memory.

## Promotion gates (mlops/gates.py)

Before a candidate reaches production it must pass:

1. **Metrics gate** - accuracy >= `MLOPS_MIN_ACCURACY` (default 0.80) and no
   accuracy regression vs the incumbent.
2. **Regression gate** - latency regression <=
   `MLOPS_MAX_LATENCY_REGRESSION_PCT` (default 10%).
3. **Drift gate** - reference vs current comparison has no high-severity drift.
4. **Fairness gate** - no protected group is "at risk".

Gate results are stored on the model manifest and summarised in release
reports under `reports/releases/`.

## Model lifecycle

```
draft --(gates pass)--> staging --(approval)--> production --(issues)--> rollback
                                                       |
                                                       +--(newer promoted)--> staged --> archived
```

- Promotion is automated via CLI but remains human-in-the-loop: `promote`
  fails closed when gates fail.
- `rollback` restores the previous production version immediately.
- Old production versions are archived automatically (keep
  `MLOPS_KEEP_PRODUCTION_VERSIONS`, default 3).

## Reproducibility

- Pinned environment: `requirements.txt` + `environment.yml`.
- Git commit and dataset version are captured on every registration.
- Experiment runs record the full training config.
