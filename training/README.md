# Training Platform (`training/`)

The **Training Platform** owns everything that happens before a model is
deployed: dataset preparation, model development, evaluation, quality gates,
experiment tracking and the MLOps loop. It depends only on `shared/` (never on
`application/`).

## Layout

| Directory | Responsibility |
| --------- | -------------- |
| [`models/`](models/README.md) | TabTransformer, Dual-CNN (NDVI/EVI), temporal Transformer, cross-modal fusion, exporters |
| [`preprocessing/`](preprocessing/README.md) | Tabular + satellite preprocessing, feature engineering |
| [`training/`](training/README.md) | Training engine, loss/metrics, checkpoints |
| [`explainability/`](explainability/README.md) | SHAP / saliency explanations |
| [`dataset_manager/`](dataset_manager/README.md) | Dataset profiling, validation, versioning |
| [`stam/`](stam/README.md) | Spatio-temporal alignment (formerly `services/spatial_alignment`) |
| [`quality/`](quality/README.md) | Drift, fairness, monitoring, optimization gates |
| [`mlops/`](mlops/README.md) | Model registry, promotion gates, scheduler, experiments, reports |
| [`feature_engineering/`](feature_engineering/) | Feature engineering pipeline (placeholder) |
| [`evaluation/`](evaluation/) | Offline evaluation harness (placeholder) |
| [`experiments/`](experiments/) | Experiment tracking workspace (placeholder) |
| [`export/`](export/) | Model export artifacts (placeholder) |
| [`hyperparameter_search/`](hyperparameter_search/) | HPO workspace (placeholder) |
| [`kaggle/`](kaggle/README.md) | Kaggle notebooks and runtime |
| [`config/`](config/) | Platform-level configuration (placeholders) |
| [`tests/`](tests/) | Platform-wide test utilities |

## Local install

```bash
pip install -e ./training/models ./training/preprocessing ./training/training ./training/explainability
pip install -e ./training/dataset_manager ./training/stam
```

Run the platform tests with `pytest training`.
