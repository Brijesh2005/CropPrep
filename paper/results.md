# Results

## Frozen test (single evaluation)

| model | accuracy | balanced acc | macro-F1 | weighted-F1 | ROC-AUC |
|---|---|---|---|---|---|
| tabular | 0.7571 | 0.4962 | 0.4757 | 0.7062 | 0.4224 |
| imagery | 0.252 | 0.4847 | 0.2449 | 0.201 | 0.4566 |
| cropfusion | 0.2612 | 0.4885 | 0.2558 | 0.2175 | 0.5337 |

## Fusion delta
- vs tabular: Δ balanced acc -0.0077, Δ macro-F1
  -0.2199, Δ AUC 0.1113.
- vs imagery: Δ balanced acc 0.0038, Δ macro-F1
  0.0109, Δ AUC 0.0771.
