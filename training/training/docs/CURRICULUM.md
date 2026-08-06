# Curriculum Training

Five-stage curriculum training warms the CropFusion model up progressively
instead of training every parameter from epoch one. Each stage unfreezes one
component while the rest stay frozen, so gradients first shape a single encoder
and only later the whole network.

## Stages

| # | name | Unfrozen components (top-level `CropFusionModel` attrs) |
| --- | --- | --- |
| 1 | `tabular` | `tab_encoder` |
| 2 | `image` | `ndvi_encoder`, `evi_encoder`, `image_fusion` |
| 3 | `temporal` | `temporal_transformer`, `temporal_proj` |
| 4 | `fusion` | `fusion_engine` (cross attention + gated fusion + shared encoder) |
| 5 | `finetune` | everything (`__all__`) |

Stage scheduling is automatic and data-driven:

* `start_stage` skips earlier stages — `start_stage: 3` begins at `temporal`.
  This doubles as resume-from-any-stage semantics.
* `epochs_per_stage` pins explicit per-stage epoch counts; unspecified stages
  split the remaining budget evenly.
* Without overrides the total epoch budget is split evenly across the *active*
  stages.
* Stages whose components do not exist on the model are dropped (a
  tabular-only model runs `tabular → finetune`), and their budget merges into
  the remaining stages.

## Configuration

```yaml
curriculum:
  enabled: true
  start_stage: 1        # 1..5
  epochs_per_stage: {}  # optional {"tabular": 3, "finetune": 4, ...}
  log_transitions: true # record the active stage name into epoch history
```

Env override: `TRN_CURRICULUM__ENABLED`, `TRN_CURRICULUM__START_STAGE`, etc.

## Freezing semantics

A frozen component gets `requires_grad=False` **and** runs in `eval()` mode, so
BatchNorm statistics and Dropout stay inert while it is frozen; the trainable
scope stays in `train()` mode. Because the trainer calls `model.train()` once
per epoch, `CurriculumCallback` re-applies the per-module mode split through
the `on_model_train_mode` hook after every `model.train()` call.

## Usage

```python
from training.training import (
    CropFusionTrainer, TrainingConfig, build_curriculum, CurriculumCallback,
)

config = TrainingConfig(
    train={"epochs": 25, "early_stopping_patience": 10},
    curriculum={"enabled": True, "epochs_per_stage": {"tabular": 5, "finetune": 5}},
)

trainer = CropFusionTrainer(model, train_loader, config, val_loader=val_loader)
result = trainer.train()
```

`result.stages` holds one entry per epoch transition: `{"stage", "frozen",
"trainable", "epoch"}`. When `log_transitions` is on, each epoch's history log
also carries the active `stage` name.

## Standalone use with the base Trainer

```python
from training.training import build_curriculum, CurriculumCallback
from training.training.trainer import Trainer

curriculum = build_curriculum(model, config.curriculum, num_epochs=config.train.epochs)
callback = CurriculumCallback(curriculum)   # all_ranks=True, DDP-consistent

trainer = Trainer(model, train_loader, config, callbacks=[callback])
```

The callback fires on **every** rank so the parameter graph stays identical
across processes in distributed training.

## Freeze order and optimizer

The optimizer is built by the trainer **after** the curriculum's first stage is
applied, so only the trainable (unfrozen) parameters receive optimizer slots.
Re-running with a frozen-then-unfrozen schedule is safe: `apply_stage` updates
`requires_grad` directly.
