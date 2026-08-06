# R4 Curriculum Flow

```mermaid
flowchart TD
    CFG["CurriculumConfig<br/>enabled · start_stage · epochs_per_stage · log_transitions"]

    CFG --> BUILD["build_curriculum(model, cfg, num_epochs)"]
    BUILD --> ACTIVE["active_stages()<br/>drop stages with no components<br/>start_stage onward"]
    ACTIVE --> SCHED["stage_epochs()<br/>explicit overrides first,<br/>then even split of the budget"]

    SCHED --> CB["CurriculumCallback<br/>on_epoch_begin(epoch)"]
    CB --> S["stage_at(epoch) → stage"]
    S --> FREEZE["apply_stage(stage)<br/>requires_grad=False + eval()<br/>on frozen components"]
    FREEZE --> LOG["stages_log.append({stage, frozen, trainable, epoch})"]

    TRAIN["model.train() per epoch"] --> HOOK["_on_model_train_mode()<br/>→ on_model_train_mode hook"]
    HOOK --> EVAL["apply_eval_mode()<br/>frozen → eval · trainable → train"]
    EVAL --> LOOP["epoch"]

    subgraph schedule["5 stages"]
        direction LR
        T1["1 tabular<br/>tab_encoder"]
        T2["2 image<br/>ndvi/evi/image_fusion"]
        T3["3 temporal<br/>temporal_transformer"]
        T4["4 fusion<br/>fusion_engine"]
        T5["5 finetune<br/>__all__"]
    end
    SCHED -. "epoch budget" .-> schedule

    style CB fill:#e8f5e9
    style FREEZE fill:#ffebee
    style schedule fill:#fff8e1
```

Semantics:

```python
config = TrainingConfig(curriculum={"enabled": True, "start_stage": 3})
# -> active stages: temporal → fusion → finetune (resume-from-any-stage)
```

`all_ranks=True` on the callback keeps the parameter graph identical across
DDP processes.
