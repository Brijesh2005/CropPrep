# R4 Optimizer Flow

```mermaid
flowchart TD
    CFG["OptimizerConfig<br/>name · lr · weight_decay · betas · eps"]
    M["CropFusionModel<br/>(raw_model, pre-compile)"]
    CUR["Curriculum first stage applied<br/>(unfrozen params only)"]

    CFG --> BO["build_optimizer(model, config, params=_trainable_params)"]
    M --> BO
    CUR -. "requires_grad flags" .-> BO

    BO --> OPT["AdamW · SGD · RAdam · Lion"]
    OPT --> STEP["optimizer.step() per optimizer step<br/>gradient clipping · AMP scaler"]
    STEP --> LOOP["training loop"]

    SCFG["SchedulerConfig<br/>name · warmup · mode · ..."] --> BS["build_scheduler(config)"]
    BS --> SH["SchedulerHandle<br/>step_period: epoch | step · requires_metric"]
    SH --> STEP

    G["gradient_accumulation_steps<br/>N micro-batches → 1 step"] --> STEP
    NAN["nan_detection<br/>warn | skip | stop"] --> STEP

    LOOP --> ES["Early stopping<br/>val_loss · patience · min_delta · restore best"]
    LOOP --> LR["logs['lr'] = scheduler.get_last_lr()[0]"]

    style OPT fill:#e8f5e9
    style ES fill:#ffebee
    style SH fill:#e3f2fd
```

Notes:

* The optimizer is built **after** the curriculum applies its first stage, so
  only trainable parameters get optimizer slots.
* `lr` is logged per step / epoch from the `SchedulerHandle` for the reports.
* Schedulers that need a metric (`ReduceLROnPlateau`) are stepped with the
  monitored validation value after each epoch.
