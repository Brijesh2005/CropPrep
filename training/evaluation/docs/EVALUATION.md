# Evaluation (Phase R5)

`training.evaluation` runs a trained `CropFusionModel` over a Phase-4 loader in
evaluation mode and reduces per-task extended metrics, PR curves, confusion
matrices, raw predictions, shared embeddings, forward-pass latency and the
per-sample fusion gates into one `EvaluationOutcome`.

## One pass, one contract

```python
from training.evaluation import EvaluationConfig, MultimodalEvaluator

config = EvaluationConfig(general={"device": "cpu", "collect_embeddings": True})
outcome = MultimodalEvaluator(model, config).evaluate(loader)

outcome.num_samples        # 16
outcome.metrics["crop"]    # accuracy, balanced_accuracy, precision, recall,
                           # f1, roc_auc, auprc, confusion_matrix, per_class
outcome.metrics["yield"]   # mse, rmse, mae, median_absolute_error, bias,
                           # r2, mape, within_tolerance, error_histogram
outcome.pr_curves["crop"]  # one-vs-rest precision-recall per class
outcome.predictions        # per-task targets / preds / probs arrays
outcome.embeddings         # [N, D] shared representations
outcome.latency_ms         # mean / p50 / p95 forward latency
outcome.gates              # per-sample image_gate / tabular_gate / fusion_gate
```

The loader yields Phase-4 batch dicts (`tabular`, `ndvi`, `evi`,
`temporal_mask`) plus `crop_label` / `yield_label` targets. The model stays in
`eval()` with gradients disabled — the same contract used at inference time.

## Report artefacts

`generate_evaluation_reports` writes markdown + JSON plus figures:

- `evaluation_report.md` / `.json` — per-task metric + per-class tables,
- `confusion_matrix.png`, `pr_curves.png`, `error_histogram.png`,
- `per_class_comparison.csv`.

```python
from training.evaluation.reports import generate_evaluation_reports

paths = generate_evaluation_reports(outcome, config, directory="artifacts/evaluation")
```

## Configuration

`EvaluationConfig` sections: `general` (device / seed / batch_size /
collect_embeddings / output_dir), `metrics` (top_k, average, roc_auc,
pr_curves, histogram_bins, error_percentiles, tolerance_fraction),
`comparison`, `ablation`, `error_analysis`. Resolution:
`EVAL_<SECTION>__<KEY>` env > YAML (`EVAL_CONFIG_FILE`) > defaults. Lists
passed through env use JSON, e.g. `EVAL_METRICS__ERROR_PERCENTILES='[0.5]'`.
