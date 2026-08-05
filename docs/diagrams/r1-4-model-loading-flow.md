# R1.4 Model Loading Flow

```mermaid
flowchart LR
    subgraph PKG["inference_package/"]
        META["metadata.db"]
        HC["historical_context.parquet"]
        LOC["location_index.parquet"]
        SC["feature_scalers.pkl"]
        LE["label_encoder.pkl"]
        MCFG["model_config.yaml"]
        MVER["model_version.json"]
    end

    subgraph MOD["models/"]
        W1["cropfusion.pt"]
        W2["cropfusion_v1.pt · v2 · latest"]
    end

    VAL["InferencePackageValidator"]
    VRES["ModelVersionResolver"]
    LDR["ModelLoader"]
    PKG2["ModelPackage"]

    PKG --> VAL
    MOD --> VRES
    VAL --> LDR
    VRES --> LDR
    LDR --> PKG2
    PKG2 --> ENG["InferenceEngine (future)"]

    style VAL fill:#e8f5e9
    style VRES fill:#e8f5e9
    style LDR fill:#e8f5e9
    style PKG2 fill:#fff3e0
```

1. Validate the package manifest (required artifacts + kinds).
2. Resolve the served version (`latest` default, or pinned `cropfusion_v*.pt`).
3. Load weights + scalers + encoder + config into a `ModelPackage`.
4. The future engine runs the package — no training code involved.

R1.4 ships the ports (`InferencePackageValidator`, `ModelVersionResolver`,
`ModelLoader`) — no loading is performed yet.
