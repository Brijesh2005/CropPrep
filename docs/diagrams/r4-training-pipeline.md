# R4 Training Pipeline

```mermaid
flowchart LR
    CFG["TrainingConfig<br/>TRN_* env > YAML > defaults"]
    DATA["Training / validation loaders<br/>tabular · ndvi · evi · temporal_mask ·<br/>crop_label · yield_label"]
    M["CropFusionModel"]

    CFG --> W["class_frequency_weights<br/>one pass over train loader"]
    CFG --> C["Curriculum<br/>5 stages · auto transitions"]
    CFG --> CM["compile?<br/>torch.compile(raw_model)"]

    M --> CM
    DATA --> CFT["CropFusionTrainer<br/>(extends Trainer)"]

    CFT --> L["MultiTaskLoss<br/>crop weighted · yield huber/mse/mae"]
    CFT --> O["Optimizer · AdamW/SGD/RAdam/Lion"]
    CFT --> S["Scheduler · cosine/onecycle/plateau/poly"]
    CFT --> V["Validator · holdout/K-fold/spatial/temporal"]
    CFT --> CK["Checkpoint manager<br/>best / latest / periodic · resume"]

    L --> LOOP["Training loop<br/>per epoch"]
    O --> LOOP
    S --> LOOP
    C --> LOOP
    V --> LOOP
    CK --> LOOP

    LOOP --> R["5 end-of-run reports<br/>training · validation · metrics ·<br/>checkpoint · learning_curve"]

    style CFT fill:#e8f5e9
    style R fill:#e3f2fd
    style CFG fill:#fff3e0
```

Entry points:

```python
trainer = CropFusionTrainer(model, train_loader, config, val_loader=val_loader)
result = trainer.train()   # -> CropFusionTrainingResult (stages, reports)
```
