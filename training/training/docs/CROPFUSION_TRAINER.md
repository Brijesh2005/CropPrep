# CropFusionTrainer

`CropFusionTrainer` is the specialised entry point for training a Phase 5
`CropFusionModel`. It extends the base `Trainer` with the Phase 7 training
strategies while inheriting everything else (AMP, gradient handling,
schedulers, checkpoints, callbacks, resume).

## Responsibilities

* **Class-imbalance handling** — one pass over the training loader derives
  per-class weights from crop-label frequencies (`balanced` / `sqrt_inv` /
  `effective_num`) and threads them into the crop loss. See
  [CLASS_WEIGHTS](CLASS_WEIGHTS.md).
* **Curriculum training** — freezes / unfreezes model components across the
  five stages (tabular → image → temporal → fusion → finetune) with automatic
  transitions. See [CURRICULUM](CURRICULUM.md).
* **End-of-run reports** — writes the five report artefacts from the finished
  training history. See [REPORTS](REPORTS.md).
* **torch.compile** — optional model compilation before the DDP wrapper. See
  [COMPILE](COMPILE.md).

## Result

`train()` returns a `CropFusionTrainingResult` (a `TrainingResult` plus):

* `stages` — per-epoch-transition freeze reports
  (`{"stage", "frozen", "trainable", "epoch"}`).
* `reports` — `{report_type: path}` for the written artefacts.
* `summary()` adds `stages` (count) and `reports`.

## Usage

```python
from training.training import CropFusionTrainer, TrainingConfig

config = TrainingConfig(
    name="run_2026_01",
    general={"device": "cpu", "seed": 42},
    train={"epochs": 50, "early_stopping_patience": 10},
    optimizer={"name": "adamw", "lr": 1e-4},
    scheduler={"name": "cosine", "warmup_epochs": 3},
    loss={"crop_loss": "label_smoothing", "class_weight_mode": "balanced"},
    curriculum={"enabled": True},
    checkpoint={"directory": "artifacts/run_2026_01/ckpt"},
    logging={"console": False},
)

trainer = CropFusionTrainer(
    model, train_loader, config,
    val_loader=val_loader,
    callbacks=[my_callback],
)
result = trainer.train()

print(result.summary())
print(result.stages)
print(result.reports)
```

The `curriculum` callback is inserted **before** any user callbacks, so user
`on_epoch_begin` handlers observe the already-applied freeze state. All
constructor keyword arguments mirror `Trainer` (`val_loader`, `loss_module`,
`optimizer`, `scheduler_handle`, `callbacks`, `checkpoint_manager`, `logger`,
`validator`, `device`, `input_map`), so an existing run can switch by changing
the class name.

## Behaviour switches

| Config | Off switch |
| --- | --- |
| Class weights | `loss.class_weight_mode: none` → no stats pass, `class_weights is None` |
| Curriculum | `curriculum.enabled: false` → no callback, `result.stages == []`, no `stage` in history |
| Reports | `general.reports: false` → `result.reports == {}`, nothing written |
| Compile | `general.compile: false` → raw model used as-is |
