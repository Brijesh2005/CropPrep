# R1.3 Dependency Rules

```mermaid
flowchart LR
    subgraph Training["training/  — Training Platform"]
        DM["dataset_manager"]
        STAM["stam"]
        PPT["preprocessing"]
        MOD["models"]
        TD["training"]
        EXP["explainability"]
        MLO["mlops"]
    end

    subgraph App["application/  — Prediction Platform"]
        BACK["backend (FastAPI)"]
        FRONT["frontend (React)"]
    end

    subgraph Shared["shared/  — contract layer"]
        P["config · enums · exceptions · interfaces<br/>logging · schemas · serialization · types<br/>utils · validation · versioning"]
    end

    subgraph Base["base"]
        STD["Python stdlib + site-packages"]
    end

    DM --> P
    STAM --> P
    PPT --> P
    MOD --> P
    TD --> P
    EXP --> P
    MLO --> P
    BACK --> P
    P --> STD

    BACK -. "delegates execution to training algorithms<br/>(ModelFactory · STAM · Preprocessor) — out of R1.3 scope" .-> MOD
    BACK -.-> STAM
    BACK -.-> PPT
    BACK -.-> DM
    BACK -.-> EXP

    style Shared fill:#e3f2fd
    style Training fill:#fff3e0
    style App fill:#fce4ec
    style Base fill:#efebe9
```

Rules that hold and are verified:

1. `training/*` imports only `shared` (plus stdlib/third-party).
2. `application/backend/app/core` imports `shared` for utilities; it never
   imports private `training` helpers.
3. `shared` imports nothing from either platform.
4. The dashed lines are **execution delegation** (the backend calls training
   algorithms at runtime), not utility imports — intentional and unchanged.
