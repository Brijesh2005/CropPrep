# R4 Validation Flow

```mermaid
flowchart TD
    CFG["ValidationConfig<br/>strategy: holdout | kfold | stratified_kfold |<br/>spatial | temporal · k_folds · group_column"]
    DATA["Preprocessed observations"]

    CFG --> SPLIT["cross_validation_splits(...)"]
    DATA --> SPLIT
    SPLIT --> HOLD["holdout<br/>train / val / test"]
    SPLIT --> KF["kfold / stratified / spatial / temporal<br/>k_folds folds"]

    V["Validator.validate(val_loader, epoch)"]
    HOLD --> V
    KF --> V

    V --> MET["crop metrics<br/>accuracy · top_k · precision · recall · f1 · roc_auc"]
    V --> REG["yield metrics<br/>rmse · mae · r2"]
    MET --> LOGS["epoch logs<br/>val_loss · crop/* · yield/*"]
    REG --> LOGS

    LOGS --> ES["Early stopping<br/>on val_loss (min)"]
    LOGS --> SCH["Scheduler step<br/>(requires_metric → val value)"]
    LOGS --> MC["ModelCheckpoint best<br/>monitor val_loss"]
    LOGS --> REP["validation_report.md<br/>per-epoch val table"]

    style V fill:#e8f5e9
    style REP fill:#e3f2fd
    style ES fill:#ffebee
```

Validation runs every `general.validation_frequency` epochs; without a
`val_loader` the trainer skips validation, early stopping and best-checkpoint
tracking gracefully (the validation report then notes that no validation
metrics were recorded).
