# R2.4 Architecture Registry & Checkpoint Rebuild

```mermaid
flowchart LR
    subgraph Registry["ModelFactory._ARCHITECTURES"]
        A1["cropfusion_v1 → CropFusionModel"]
        A2["cropfusion_v2 → (registered, future)"]
    end

    subgraph Save["CheckpointManager.save"]
        W["model_state_dict"]
        AC["architecture + architecture_version"]
        MD["metadata (JSON-safe)"]
        CFG["model_config"]
    end

    subgraph Rebuild["ModelFactory.from_checkpoint"]
        RES["resolve architecture by name"]
        C["create(config, architecture=…)"]
        LOAD["load_state_into (strict)"]
    end

    M["CropFusionModel"] --> Save
    Save --> DISK[".pt checkpoint"]
    DISK --> Rebuild
    Registry --> RES
    RES --> C --> LOAD --> REST["restored model (same class + weights)"]

    style Registry fill:#fff3e0
    style Save fill:#e8f5e9
    style Rebuild fill:#e3f2fd
```

Guarantees:

- checkpoints record the architecture name + schema version, so a registered
  future architecture is rebuilt with its own class, not the built-in one;
- `metadata` is JSON-serialised on save so `torch.load(weights_only=True)`
  accepts it (`torch.__version__` is otherwise rejected by the safe loader);
- unknown explicit architectures raise `MDL-CONFIG-001`; unregistered config
  display names fall back to `CropFusionModel`.
