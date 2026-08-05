# Training Guide

This guide explains how to run training with the CropFusion training engine.

## Pipeline

```
Dataset Manager ──► STAM ──► Preprocessing (Preprocessor) ──► PyTorch Dataset/DataLoader
                                                                        │
                                                                CropFusionModel
                                                                        │
                    Config ─► Model ─► Loss ─► Optimizer ─► Scheduler ─► Mixed Precision
                                                                        │
                                                              Training Loop ─► Validation Loop
                                                                        │
                                                       Checkpointing ─► Evaluation ─► Reports
```

## Running a training run

```python
from ai.preprocessing import Preprocessor
from ai.models import ModelFactory
from ai.training import Experiment, TrainingConfig

# 1. Observations come from STAM (accepted, quality-filtered).
train, val, test = split_observations(accepted, preprocessor.config.split)

# 2. Fit the preprocessor on the training split only (no leakage).
preprocessor = Preprocessor(config).fit(train, extractor=stam.get_patch)

# 3. Derive the model from the fitted preprocessor.
model_config = ModelFactory.build_config(preprocessor)

# 4. Configure + run the experiment (hold-out by default).
training_config = TrainingConfig(
    train={"epochs": 100, "early_stopping_patience": 10},
    optimizer={"name": "adamw", "lr": 1e-4, "weight_decay": 1e-4},
    scheduler={"name": "cosine", "warmup_epochs": 5},
    loss={"crop_loss": "label_smoothing", "yield_loss": "huber",
          "crop_weight": 0.7, "yield_weight": 0.3},
    general={"seed": 42, "amp": True, "device": "auto"},
)
report = Experiment(
    training_config, accepted,
    preprocessor=preprocessor, extractor=stam.get_patch,
    model_config=model_config,
).run()
```

## The `Trainer` in isolation

For full control, drive the `Trainer` directly:

```python
from ai.training import Trainer, MultiTaskLoss, build_optimizer, build_scheduler

trainer = Trainer(
    model, train_loader, training_config,
    val_loader=val_loader,
    loss_module=MultiTaskLoss(training_config.loss),
    optimizer=build_optimizer(model, training_config.optimizer),
    scheduler_handle=build_scheduler(
        optimizer, training_config.scheduler,
        steps_per_epoch=len(train_loader), total_epochs=training_config.train.epochs,
    ),
)
result = trainer.train()
```

## Training features

| Feature | Config key | Notes |
|---------|-----------|-------|
| AMP (fp16/bf16) | `general.amp`, `general.amp_dtype` | CUDA only; CPU falls back automatically |
| Gradient clipping | `general.gradient_clip`, `general.gradient_clip_type` | `norm` or `value` |
| Gradient accumulation | `general.gradient_accumulation_steps` | Loss auto-scaled by the accumulator |
| Gradient checkpointing | `general.gradient_checkpointing` | Wraps the image encoders |
| Early stopping | `train.early_stopping_*` | monitor metric + patience + min_delta |
| NaN detection | `general.nan_detection`, `general.nan_policy` | `warn` / `skip` / `stop` |
| Seed / determinism | `general.seed`, `general.deterministic` | torch/numpy/python + cudnn |
| Automatic resume | `checkpoint.resume` (+ `resume_path`) | restores model/optimizer/scheduler/scaler/RNG |
| DDP | auto via `torchrun` env (`RANK`/`WORLD_SIZE`) | graceful CPU fallback |

## Resume

```yaml
checkpoint:
  directory: artifacts/training/checkpoints
  resume: true          # resume from the latest checkpoint
  resume_path: null     # or point at a specific checkpoint
```

Resume restores the model weights, optimizer, scheduler, AMP scaler, GradNorm
controller state and the torch/numpy/python random states. The next epoch
continues from the last **completed** epoch.

## Dynamic multi-task weighting

```yaml
loss:
  weighting_mode: fixed        # fixed | uncertainty | gradnorm
  crop_weight: 0.7
  yield_weight: 0.3
  gradnorm_alpha: 1.5          # GradNorm asymmetry (Chen et al., 2018)
```

* `fixed` — constant task weights.
* `uncertainty` — Kendall et al. (2018) learned log-variances.
* `gradnorm` — GradNorm: per-task weights are adapted each step from the shared
  encoder gradient norms.
