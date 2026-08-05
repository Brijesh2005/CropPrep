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
| [`dataset_manager/`](dataset_manager/README.md) | Dataset profiling, validation, versioning; **provider pattern** (`GitRepositoryTabularProvider` + `KaggleHubImageProvider`) |
| [`stam/`](stam/README.md) | Spatio-temporal alignment (formerly `services/spatial_alignment`) |
| [`quality/`](quality/README.md) | Drift, fairness, monitoring, optimization gates |
| [`mlops/`](mlops/README.md) | Model registry, promotion gates, scheduler, experiments, reports |
| [`feature_engineering/`](feature_engineering/) | Feature engineering pipeline (placeholder) |
| [`evaluation/`](evaluation/) | Offline evaluation harness (placeholder) |
| [`experiments/`](experiments/) | Experiment tracking workspace (placeholder) |
| [`export/`](export/) | Model export artifacts (placeholder) |
| [`hyperparameter_search/`](hyperparameter_search/) | HPO workspace (placeholder) |
| [`kaggle/`](kaggle/README.md) | Kaggle notebooks (`train` / `evaluate` / `export`) and orchestration scripts |
| [`config/`](config/) | Platform config: `dataset` · `training` · `model` · `logging` · `validation` · `kaggle` |
| [`datasets/`](datasets/) | Git-versioned tabular datasets (`tabular/`) + git-ignored Kaggle imagery (`raw/`) |
| [`tests/`](tests/) | Platform-wide test utilities |

## Local install

```bash
pip install -e ./training/models ./training/preprocessing ./training/training ./training/explainability
pip install -e ./training/dataset_manager ./training/stam
```

Run the platform tests with `pytest training`.

## Data sources

The Dataset Manager reads through two independent providers (see
[`docs/diagrams/r1-2-provider-architecture.md`](../docs/diagrams/r1-2-provider-architecture.md)):

- **Git tabular** — `training/datasets/tabular/*.csv` via
  `GitRepositoryTabularProvider` (version controlled, must stay in the repo).
- **Kaggle imagery** — `shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada`
  (Sentinel NDVI/EVI GeoTIFFs) via `KaggleHubImageProvider`, materialised under
  `training/datasets/raw/` (git-ignored; never committed).

Example:

```bash
cropfusion-dataset-manager providers
cropfusion-dataset-manager tabulars
cropfusion-dataset-manager image-catalog
```

Kaggle runs are orchestrated by `training/kaggle/scripts/` and
`training/kaggle/notebooks/` — see the [kaggle README](kaggle/README.md).
