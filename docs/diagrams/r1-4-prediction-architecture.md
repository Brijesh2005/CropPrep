# R1.4 Prediction-Only Architecture

```mermaid
flowchart TB
    subgraph CFG["application/config"]
        Y1["application.yaml"]
        Y2["model.yaml"]
        Y3["inference.yaml"]
        Y4["logging.yaml"]
        Y5["security.yaml"]
        Y6["database.yaml"]
    end

    subgraph GIS["application/gis"]
        RG["reverse_geocoding<br/>ReverseGeocoder"]
        SR["spatial_resolver<br/>SpatialResolver"]
        HC["historical_context<br/>HistoricalContextResolver"]
        LRC["resolver<br/>LocationResolver"]
        LRC --> RG
        LRC --> SR
        LRC --> HC
    end

    subgraph INF["application/inference"]
        SVC["services<br/>PredictionService"]
        ENG["engine<br/>InferenceEngine"]
        LDR["loaders<br/>ModelLoader"]
        CAC["cache<br/>PredictionCache"]
        EXP["explainability<br/>PredictionExplainer"]
        VER["versioning<br/>ModelVersionResolver"]
        VAL["validation<br/>InferencePackageValidator"]
        SVC --> ENG
        ENG --> LDR
        SVC --> CAC
        SVC --> EXP
        LDR --> VER
        VAL -.-> LDR
    end

    subgraph HIST["application/history"]
        ST["store<br/>PredictionHistoryStore"]
    end

    subgraph PKG["inference package + models"]
        PK["application/inference_package<br/>manifest.py + artifacts"]
        MD["application/models<br/>cropfusion.pt / cropfusion_*.pt"]
    end

    subgraph SH["shared/"]
        SH1["config · enums · schemas"]
        SH2["versioning · validation · interfaces"]
    end

    GIS --> INF
    HIST --> INF
    PKG --> INF
    CFG -. "load_yaml_config" .-> INF
    INF --> SH
    GIS --> SH
    HIST --> SH

    style SH fill:#e3f2fd
    style INF fill:#e8f5e9
    style GIS fill:#fff3e0
    style HIST fill:#fff3e0
    style PKG fill:#fce4ec
    style CFG fill:#f3e5f5
```

- `application/inference`, `application/gis`, `application/history` and the
  package/models contracts depend only on `shared/` (never `training/`).
- The existing `application/backend` (API + embedded engine) is unchanged and
  outside this diagram.
