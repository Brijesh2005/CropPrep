# R1.4 Inference Package Lifecycle

```mermaid
flowchart LR
    TR["Training Platform<br/>export pipeline"] -->|"model + context"| STG["staging dir"]
    STG --> PKG["package assembler"]
    PKG -->|"versioned artifact set"| DEP["deployment"]
    DEP -->|"copy"| IFP["application/inference_package/"]
    DEP -->|"copy"| MOD["application/models/"]
    IFP --> VAL["InferencePackageValidator"]
    MOD --> LDR["ModelLoader"]
    VAL --> LDR
    LDR --> ENG["InferenceEngine"]
    ENG --> RES["PredictionResult"]

    style TR fill:#e8f5e9
    style VAL fill:#fff3e0
    style LDR fill:#fff3e0
    style IFP fill:#fce4ec
    style MOD fill:#fce4ec
```

- `application/inference_package` is **consumed only** — it never generates
  artifacts (the `.gitignore` keeps shipped files out of git).
- The manifest (`manifest.py`) is the single source of truth for the file list.
