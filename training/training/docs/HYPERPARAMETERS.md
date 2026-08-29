# Hyperparameter Guide

Every hyperparameter is configurable via YAML (or `TRN_*` env vars). The
template below is produced by `save_training_template("training.yaml")`.

## Full template

```yaml
name: cropfusion_training

general:
  device: auto            # auto | cpu | cuda (falls back to CPU gracefully)
  output_dir: artifacts/training
  seed: 42
  deterministic: false
  amp: false              # mixed precision (fp16 on CUDA)
  amp_dtype: float16      # float16 | bfloat16
  gradient_clip: null     # max grad norm (null = off)
  gradient_clip_type: norm
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  nan_detection: true
  nan_policy: skip        # warn | skip | stop
  log_every: 1
  validation_frequency: 1

data:
  batch_size: 32
  workers: 0
  pin_memory: false
  prefetch_factor: null
  persistent_workers: false
  drop_last: false
  train_shuffle: true

optimizer:
  name: adamw             # adamw | sgd | radam | lion
  lr: 0.0001
  weight_decay: 0.0001
  betas: [0.9, 0.999]
  eps: 1.0e-8
  momentum: 0.0           # sgd
  nesterov: false
  lion_beta1: 0.9
  lion_beta2: 0.99
  backbone_lr_multiplier: null   # discriminative LR: backbone = lr * mult

scheduler:
  name: cosine            # none | cosine | onecycle | reduce_on_plateau | polynomial | warmup_cosine | warmup_polynomial
  step: epoch             # epoch | step
  warmup_steps: 0
  warmup_epochs: 0
  warmup_ratio: 0.0       # fraction of the schedule, overrides the above
  t_max: null             # cosine cycle length (null = epochs)
  eta_min: 0.0
  pct_start: 0.3          # onecycle
  div_factor: 25.0
  final_div_factor: 10000.0
  factor: 0.1             # reduce_on_plateau
  patience: 10
  threshold: 0.0001
  cooldown: 0
  min_lr: 0.0
  mode: min
  power: 1.0              # polynomial exponent
  end_lr: 0.0             # polynomial final LR

loss:
  crop_loss: label_smoothing   # cross_entropy | label_smoothing | focal
  yield_loss: huber            # mse | huber | mae
  crop_weight: 0.7
  yield_weight: 0.3
  weighting_mode: fixed        # fixed | uncertainty | gradnorm
  label_smoothing: 0.1
  focal_gamma: 2.0
  reduction: mean
  gradnorm_alpha: 1.5
  log_variance_eps: 0.01

train:
  epochs: 100
  early_stopping_metric: crop/macro_f1   # macro-F1 over the full class set
  early_stopping_mode: max
  early_stopping_patience: 10
  early_stopping_min_delta: 0.0
  restore_best_on_stop: true             # load best.pt weights at the last epoch

# R5.4 staged backbone fine-tuning. The image backbones start fully frozen;
# at each schedule epoch the listed block prefixes are unfrozen so the
# pretrained trunk is adapted from the prediction head down. Early blocks
# (stem / blocks.0-2) are typically never unfrozen. Prefixes are matched
# against backbone module names ("blocks.6" -> both backbone.blocks.6).
fine_tuning:
  enabled: false
  schedule:
  - epoch: 0
    prefixes: []
  - epoch: 10
    prefixes: [blocks.6, blocks.5]
  - epoch: 20
    prefixes: [blocks.4, blocks.3]

checkpoint:
  directory: artifacts/training/checkpoints
  keep_last: 3
  save_best: true
  save_latest: true
  save_periodic: null
  resume: false
  resume_path: null

metrics:
  top_k: 5
  average: macro          # macro | micro | weighted
  roc_auc: false
  per_class: false

logging:
  level: INFO
  console: true
  csv: true
  json_logs: true
  tensorboard: false
  tensorboard_dir: artifacts/training/tensorboard
  wandb: false
  wandb_project: cropfusion
  wandb_entity: null
  config_snapshot: true
  git_hash: true

validation:
  strategy: holdout       # holdout | kfold | stratified_kfold | spatial | temporal
  k_folds: 5
  shuffle: true
  seed: 42
  group_column: village
  temporal_column: year

ablation:
  enabled: false
  variants: [full, only_tabular, only_ndvi, only_evi, only_image,
             no_cross_attention, no_adaptive_gate]
  compare_metric: multi_task_score
  compare_mode: max

benchmark:
  enabled: false
  iterations: 100
  warmup_iterations: 10
  batch_size: 32
  measure_training_speed: true
  measure_inference_speed: true
  inference_only: false

visualization:
  enabled: true
  directory: artifacts/training/visualizations
  dashboard: true
  loss_curves: true
  accuracy_curves: true
  lr_curves: true
  regression_scatter: true
  confusion_matrix: true
  precision_recall: true
  feature_distribution: true
```

## Environment variables

`TRN_<SECTION>__<KEY>` overrides YAML and defaults:

```bash
TRN_TRAIN__EPOCHS=50
TRN_OPTIMIZER__LR=0.0005
TRN_OPTIMIZER__NAME=radam
TRN_LOSS__WEIGHTING_MODE=uncertainty
TRN_GENERAL__AMP=true
TRN_VALIDATION__STRATEGY=kfold
TRN_VALIDATION__K_FOLDS=10
```

## Suggested defaults

| Setting | Value |
|---------|-------|
| Optimizer | AdamW, lr 1e-4, weight_decay 1e-4 |
| Discriminative LR | backbone_lr_multiplier 0.3 (backbone ≈3e-5, heads 1e-4) |
| Scheduler | warmup_cosine, 2 warmup epochs + cosine decay to 0 |
| Batch size | 32 (scale with GPUs / gradient accumulation) |
| Loss weights | crop 0.7 / yield 0.3 (yield inert without a yield scaler) |
| Precision | fp16 mixed precision on CUDA |
| Early stopping | monitor crop/macro_f1 (max), patience 5-10 |
| Best-weights restore | restore_best_on_stop: true (best.pt, not last epoch) |
| Image backbone | efficientnetv2_s (default), input_size 128+ |
| Staged fine-tuning | freeze stem, unfreeze blocks.6→3 from epoch 10/20 |
| Transformer depth | temporal 2, shared 2 (raise with data size) |
| Dropout | 0.1 (raise on overfit) |
| Seed | 42 (fix for reproducibility) |
