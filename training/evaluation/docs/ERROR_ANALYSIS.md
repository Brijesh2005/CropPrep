# Error Analysis (Phase R5)

`ErrorAnalysis` turns an `EvaluationOutcome` into failure diagnostics, and the
fusion-gate tie-in answers the explainability question *"does the model lean
on the wrong modality exactly when it fails?"*.

## What is analysed

- **Classification** — per-class error rates and false-positive counts, the
  top true→pred confusion pairs and the misclassified samples (with true /
  predicted probabilities when available).
- **Regression** — signed bias, absolute-error statistics, outlier samples
  (residuals above a percentile) and failure cases (relative error above a
  threshold), plus the worst predictions.
- **Groups** — error rates grouped by optional per-sample metadata (village /
  district / season / year) so systematic weaknesses become visible.
- **Fusion gates** — mean `image_gate` / `tabular_gate` / `fusion_gate` over
  the correct vs. the failing samples, highlighting modality over-reliance.

```python
from training.evaluation import ErrorAnalysis, EvaluationConfig

report = ErrorAnalysis(config).analyze(outcome, sample_metadata=metadata)
report.task_reports["crop"]        # per_class, top_confusions, misclassified
report.task_reports["yield"]       # outliers, failures, worst_predictions
report.task_reports["crop"]["group_breakdown"]
report.fusion_analysis             # gate means for overall / correct / error
```

`sample_metadata` must have exactly one entry per evaluated sample; a length
mismatch raises `ErrorAnalysisError`.

## Report

`generate_error_analysis_reports` writes `error_analysis.md` / `.json` plus
`fusion_gates.png` — a grouped bar chart of each gate's mean over
correct/error buckets (emitted whenever gates were collected).

## Explaining the gates figure

- `image_gate` high on errors ⇒ the model leans on imagery where it fails;
- `tabular_gate` high on errors ⇒ tabular features are the weak leg;
- `fusion_gate` low on errors ⇒ the model is *under-using* the fused signal
  exactly when it gets it wrong.
