# Evaluation Guide

## Metrics

**Classification** (`crop` task):

| Metric | Notes |
|--------|-------|
| Accuracy | top-1 |
| Top-K accuracy | `metrics.top_k` (default 5) |
| Precision / Recall / F1 | `metrics.average` (macro / micro / weighted) |
| ROC-AUC | one-vs-rest macro, only when `metrics.roc_auc: true` |
| Confusion matrix | per-run, used by the visualizer |

**Regression** (`yield` task):

| Metric | Notes |
|--------|-------|
| MSE / RMSE / MAE | direct |
| R² | coefficient of determination |
| MAPE | % mean absolute percentage error (guards zero targets) |

**Combined**:

`multi_task_score = 0.5 * crop_accuracy + 0.5 * max(0, 1 - nRMSE)`
where `nRMSE = RMSE / std(targets)`. This is the default ablation ranking
metric.

## Running evaluation

```python
from ai.training import Evaluator, MultiTaskLoss

evaluator = Evaluator(model, metrics_config=training_config.metrics)
result = evaluator.evaluate(test_loader, MultiTaskLoss(training_config.loss))

result.metrics            # per-task metrics
result.confusion_matrix   # {'crop': [[...]]}
result.per_task_loss      # {'crop': ..., 'yield': ...}
result.predictions        # {'crop_pred', 'crop_target', 'yield_pred', 'yield_target'}
result.feature_embeddings # shared representation (PCA-able)
result.feature_labels     # crop labels for coloring
```

## Inference benchmark

```python
from ai.training import Benchmark

bench = Benchmark(model, batch_size=32, iterations=100, warmup_iterations=10)
report = bench.run(sample_batch=model.sample_batch(batch_size=32))
```

Measures:

* **Inference speed** — mean / p50 / p95 / p99 latency (ms) + samples/second
  and batches/second.
* **Training speed** — samples/second through forward + backward + optimizer
  step (over a loader).
* **Validation speed** — samples/second in eval mode.
* **GPU memory** — peak CUDA allocation (MB) during the timed passes.
* **CPU memory** — process RSS (MB, via `psutil`).
* **Model size** — parameter count, trainable parameters, and float32 size (MB).

Enable it inside an experiment with:

```yaml
benchmark:
  enabled: true
  iterations: 100
  warmup_iterations: 10
  batch_size: 32
```

## Visualizations

`visualization.enabled: true` (default) renders:

* `loss_curves.png` — train / val loss over epochs,
* `accuracy_curves.png` — crop accuracy / F1 over epochs,
* `lr_curves.png` — learning-rate schedule,
* `regression_scatter.png` — predicted vs actual yield,
* `confusion_matrix.png` — crop confusion heatmap,
* `precision_recall.png` — precision-recall by crop class,
* `feature_distribution.png` — PCA of the shared representation by class,
* `dashboard.html` — self-contained dashboard embedding all charts + metrics.

## Resource accounting

`EvaluationResult` and `BenchmarkReport` both report:

* `parameters` — total parameter count,
* `trainable_parameters` — parameters with `requires_grad=True`,
* `model_size_mb` — float32 parameter footprint,
* `gpu_memory_mb` — peak CUDA allocation (0 on CPU),
* `cpu_rss_mb` — process resident-set size.
