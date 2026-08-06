# R4 Checkpoint Flow

```mermaid
flowchart TD
    CFG["CheckpointConfig<br/>directory · keep_last · save_best ·<br/>save_latest · save_periodic · resume"]
    R["TrainingResult<br/>best_metrics · best_epoch · best_path"]

    CFG --> CM["TrainingCheckpointManager"]
    R --> MC["ModelCheckpoint callback<br/>(on_epoch_end)"]

    MC --> SB{"save_best?<br/>monitor early_stopping_metric"}
    SB -->|improved| B["save_best → best.pt"]
    SB -->|no| SKIP["skip"]

    MC --> SL{"save_latest?"}
    SL -->|yes| LAT["save_latest → latest.pt"]
    MC --> SP{"save_periodic?"}
    SP -->|every N epochs| PER["save_periodic → periodic-N.pt"]

    RESUME{"resume / resume_path?"}
    RESUME -->|resume: true| DISC["resume_latest → discovery in directory"]
    RESUME -->|resume_path| EXPL["restore(resume_path)"]
    DISC --> ST["model · optimizer · scheduler · scaler · RNG restored"]

    B --> P["weights_only-loadable state_dict"]
    LAT --> P
    PER --> P

    CK["Checkpoint report<br/>policy + *.pt artifact list"] --> P
    P --> R

    style CM fill:#e8f5e9
    style B fill:#ffebee
    style ST fill:#e3f2fd
```

The `best_path` reported in `TrainingResult` / `training_report.md` comes from
the checkpoint manager. The checkpoint report lists the actual `*.pt` files in
the checkpoint directory.
