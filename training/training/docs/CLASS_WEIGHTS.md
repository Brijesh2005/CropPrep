# Class-Frequency Weighted Losses

Imbalanced crop classes are handled by weighting the crop loss with per-class
weights derived from the training set's label frequencies. There is **no
oversampling**; focal loss remains future work.

## Recipes

| mode | formula | use case |
| --- | --- | --- |
| `balanced` | `N / (C * n_c)` | classic inverse-frequency weighting |
| `sqrt_inv` | `1 / sqrt(n_c)` | softer weighting, dampens extreme imbalance |
| `effective_num` | `(1 - beta) / (1 - beta^n_c)` | Cui et al., 2019; density-aware |

`N` = total samples, `C` = number of classes, `n_c` = count of class `c`. All
weights are floored at `class_weight_eps` and **normalised so the mean weight
is 1** (keeps the loss scale comparable to the unweighted run). Missing
classes (count 0) receive the highest weight and stay finite.

## Configuration

```yaml
loss:
  class_weight_mode: balanced   # none | balanced | sqrt_inv | effective_num
  class_weight_eps: 1e-6
  class_weight_beta: 0.999      # effective_num only
```

Env override: `TRN_LOSS__CLASS_WEIGHT_MODE`, `TRN_LOSS__CLASS_WEIGHT_EPS`,
`TRN_LOSS__CLASS_WEIGHT_BETA`.

## How it works

`CropFusionTrainer.__init__` does a single pass over the training loader to
count `crop_label` occurrences (`_collect_class_counts`), then
`build_class_weights` turns the counts into a `[C]` tensor. The weights are
threaded into the crop loss:

* `cross_entropy` → `CrossEntropyLoss(weight=...)`
* `label_smoothing` → `WeightedLabelSmoothingLoss(weight=...)`
* `focal` → `FocalLoss(alpha=...)`

The weight tensor is a registered buffer, so it moves with the loss module to
the compute device and is device-safe when gathered per sample. `MultiTaskLoss`
accepts `class_weights={"crop": tensor, ...}` — the yield regression task is
never weighted.

## Usage

```python
from training.training import CropFusionTrainer, TrainingConfig
from training.training.losses import (
    build_class_weights, build_multi_task_loss, compute_class_counts,
)

config = TrainingConfig(loss={"class_weight_mode": "balanced"})

trainer = CropFusionTrainer(model, train_loader, config)
trainer.class_frequency   # [C] counts from one pass over train_loader
trainer.class_weights     # [C] normalised weights (mean 1), None when disabled

# Standalone helpers
counts = compute_class_counts(labels)          # [C] float counts
weights = build_class_weights(config.loss, num_classes=3, counts=counts)
loss = build_multi_task_loss(config.loss, class_weights={"crop": weights})
```

When `class_weight_mode: none`, no statistics pass runs, no weights are built,
and `trainer.class_weights` stays `None`. A user-supplied `loss_module` is
respected unchanged — weights are never injected into it.
