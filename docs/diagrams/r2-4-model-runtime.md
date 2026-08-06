# R2.4 Model Runtime

```mermaid
flowchart LR
    subgraph Runtime["runtime.py (applied by apply_runtime)"]
        direction TB
        P["apply_precision<br/>float16 / bfloat16 (norms stay float32)"]
        D["move_to_device<br/>cpu / cuda:N / mps"]
        GC["enable_gradient_checkpointing<br/>TabTransformer · Temporal · Shared encoder"]
        C["compile_model<br/>torch.compile(mode)"]
        DP["wrap_data_parallel<br/>nn.DataParallel"]
        DDP["wrap_distributed<br/>DistributedDataParallel"]
    end

    CFG["RuntimeConfig<br/>precision · device · compile ·<br/>gradient_checkpointing ·<br/>data_parallel · distributed · local_rank"]
    M["CropFusionModel"]

    CFG --> P --> D --> GC --> C --> DP --> DDP --> OUT["Deployed model"]
    M --> P

    style Runtime fill:#e8f5e9
    style CFG fill:#fff3e0
    style OUT fill:#e3f2fd
```

Entry points:

```python
model.to_precision("bfloat16")            # model method
amp_context("bfloat16", "cpu")            # autocast block
model.enable_gradient_checkpointing(True)
model.compile(mode="default", backend="eager")
ModelFactory.apply_runtime(model)         # everything in fixed order
ModelFactory.create_with_runtime(cfg)     # build + configure
```
