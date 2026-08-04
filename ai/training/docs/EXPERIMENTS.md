# Experiment Guide

`ai.training.experiment.Experiment` runs a complete train → validate → evaluate
→ benchmark → visualize pipeline over a set of STAM observations, honouring
the configured **model-validation strategy**.

## Hold-out experiment

```python
from ai.training import Experiment
from ai.training.config import TrainingConfig

config = TrainingConfig(
    name="baseline",
    validation={"strategy": "holdout"},
    train={"epochs": 50, "early_stopping_patience": 8},
)
report = Experiment(
    config,
    observations,                    # accepted STAM observations
    preprocessor=preprocessor,       # fitted Phase 4 Preprocessor
    extractor=stam.get_patch,
    model_config=model_config,       # or None to derive from the preprocessor
).run()

report.evaluation.metrics          # per-task metric dict
report.evaluation.multi_task_score # combined score
report.benchmark                   # throughput / resources
report.artifacts                   # chart + dashboard paths
report.run_dir                     # metrics.csv / metrics.json / config.yaml ...
```

The hold-out split uses the preprocessor's Phase 4 `split` config (random /
stratified / temporal / spatial). The preprocessor is fitted on the training
split only — no leakage.

## Cross-validation

Set `validation.strategy` to one of `kfold | stratified_kfold | spatial |
temporal`. Each fold refits a fresh preprocessor on the fold's training subset
(so the tabular schema, class count and image size are re-derived per fold),
trains a model, and evaluates on the fold's validation subset. Results are
aggregated into `cross_validation.json`:

```yaml
validation:
  strategy: kfold
  k_folds: 5
  shuffle: true
  seed: 42
```

* `stratified_kfold` — stratifies by the crop attribute.
* `spatial` — whole groups (villages) go to one fold (no spatial leakage).
* `temporal` — whole years go to one fold (no temporal leakage).

## Ablation sweeps

```python
from ai.training import AblationRunner

runner = AblationRunner(
    training_config,
    model_config,
    preprocessor=preprocessor,
    observations=observations,
    extractor=stam.get_patch,
)
report = runner.run()   # runs every variant + compares
print(report.best_variant)
```

Available variants (all expressed as configuration of the Phase 5 model):

| Variant | Effect |
|---------|--------|
| `full` | complete architecture |
| `only_tabular` | `image_encoder.backbone: null` |
| `only_ndvi` | `image_encoder.enable_evi: false` |
| `only_evi` | `image_encoder.enable_ndvi: false` |
| `only_image` | `tabular.numeric_dim: 0` |
| `no_cross_attention` | `cross_attention.enabled: false` |
| `no_adaptive_gate` | `gated_fusion.enabled: false` |

The comparison is ranked by `ablation.compare_metric`
(default `multi_task_score`, highest wins) and written to
`ablation_report.json` + `ablation_comparison.csv`.

## Experiment tracking

`Experiment` writes to `<output_dir>/<run_name>/`:

* `metrics.csv` — one row per epoch,
* `metrics.json` — per-epoch records + final summary,
* `config.yaml` — resolved training + model + preprocessing snapshots,
* `environment.json` — python / torch / hardware,
* `git.json` — commit hash + branch (best-effort),
* TensorBoard / W&B when `logging.tensorboard` / `logging.wandb` are enabled
  (both degrade gracefully if the package is absent).

## Reproducibility

Set `general.seed` and `general.deterministic`. Every random source (torch /
numpy / python) is seeded; determinism is best-effort on CUDA and dependent on
the backend implementation.
