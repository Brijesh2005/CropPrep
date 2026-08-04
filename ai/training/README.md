# CropFusion — Training & Evaluation Framework (Phase 6)

Enterprise-grade training and evaluation framework for the Phase 5 multimodal
CropFusion architecture (crop recommendation + yield prediction). It consumes
**only** the Phase 4 AI-ready batches (Dataset Manager → STAM → preprocessing →
PyTorch DataLoader → model) — no direct file loading.

```
ai/training/
├── __init__.py        # public API
├── config.py          # TrainingConfig (pydantic) + YAML/env loader + template
├── interfaces.py      # ports: Callback / SchedulerHandle / FoldGenerator
├── utils.py           # seeds, determinism, device, DDP, git hash, timing
├── losses.py          # MAE, MultiTaskLoss (fixed/uncertainty/GradNorm)
├── optimizers.py      # AdamW / SGD / RAdam / Lion (self-contained)
├── schedulers.py      # cosine / OneCycle / ReduceLROnPlateau / polynomial / warmup
├── metrics.py         # classification + regression metrics, per-task tracker
├── validator.py       # validation loop + hold-out / K-fold / stratified / spatial / temporal
├── checkpoint.py      # best / latest / periodic + optimizer, scheduler, scaler, RNG
├── callbacks.py       # early stopping, model checkpoint, LR logger, console, TB/W&B
├── logger.py          # CSV / JSON metrics + config snapshot + git hash
├── trainer.py         # the training engine (AMP, clipping, resume, NaN, DDP)
├── evaluator.py       # final metrics, latency, memory, params, multi-task score
├── experiment.py      # Experiment orchestrator (hold-out + cross-validation)
├── ablation.py        # automatic ablation sweep + comparison report
├── benchmark.py       # training / validation / inference throughput + resources
├── visualizer.py      # loss/acc/LR curves, scatter, confusion, PR, PCA, dashboard
├── pyproject.toml
├── docs/              # TRAINING, EXPERIMENTS, HYPERPARAMETERS, EVALUATION, DEVELOPER
└── tests/             # 67 tests
```

## Quick start

```python
from ai.training import Experiment, TrainingConfig
from ai.training.config import load_training_config

training_config = load_training_config("training.yaml")
report = Experiment(
    training_config,
    observations,                 # accepted STAM observations
    preprocessor=preprocessor,    # fitted Phase 4 Preprocessor
    extractor=stam.get_patch,     # patch extractor
    model_config=model_config,    # ModelConfig (or derive from preprocessor)
).run()

print(report.evaluation.metrics)          # per-task metrics
print(report.evaluation.multi_task_score) # combined score
print(report.artifacts)                   # charts + dashboard paths
```

## Features

| Area | Support |
|------|---------|
| Losses | CrossEntropy, label smoothing, focal, MSE, Huber, MAE + dynamic multi-task weighting (fixed / uncertainty / GradNorm) |
| Optimizers | AdamW, SGD, RAdam, Lion |
| Schedulers | Cosine, OneCycle, ReduceLROnPlateau, polynomial, warmup (linear + cosine/poly) |
| Training | AMP (fp16/bf16), gradient clipping / accumulation / checkpointing, early stopping, automatic resume, NaN detection, seed + determinism |
| Distributed | single GPU, multi-GPU DDP, graceful CPU fallback |
| Checkpoints | best / latest / periodic, resume with optimizer + scheduler + scaler + RNG |
| Tracking | TensorBoard (optional), W&B (optional), CSV + JSON logs, config snapshot, git hash |
| Evaluation | accuracy / precision / recall / F1 / ROC-AUC / top-K / confusion matrix; RMSE / MAE / MSE / R² / MAPE; multi-task score; latency; memory; params |
| Validation | hold-out, K-fold, stratified K-fold, spatial, temporal |
| Ablations | full, only-tabular, only-NDVI, only-EVI, only-image, no-cross-attention, no-adaptive-gate + automatic comparison |
| Benchmark | training / validation / inference speed, GPU / CPU memory, model size |
| Visualization | loss / accuracy / LR curves, regression scatter, confusion matrix, precision-recall, feature distribution, HTML dashboard |

## Configuration

Everything is configurable via YAML (or `TRN_*` env vars), validated by
pydantic:

```bash
TRN_CONFIG_FILE=training.yaml python -c "from ai.training.config import load_training_config; c = load_training_config(); print(c.train.epochs)"
```

Generate a template:

```python
from ai.training.config import save_training_template
save_training_template("training.yaml")
```

See [docs/TRAINING.md](docs/TRAINING.md), [docs/HYPERPARAMETERS.md](docs/HYPERPARAMETERS.md)
and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for details.

## Testing

```bash
python -m pytest ai/training -q
```

67 tests covering the training loop, checkpoints, resume, metrics, schedulers,
losses, benchmarks, cross-validation, ablations and visualization.
