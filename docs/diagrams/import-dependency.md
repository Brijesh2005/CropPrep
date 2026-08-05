# Import Dependency Diagram

```mermaid
flowchart LR
    subgraph ApplicationPkg["application/backend/app (app.*)"]
        CORE["app.core"]
        API["app.api"]
        SVC["app.services"]
        MOD["app.modules"]
        ENG["app.modules.prediction (inference engine)"]
    end

    subgraph DatabasePkg["application/database (database.*)"]
        DB["database.repositories\n database.seeds"]
    end

    subgraph TrainingPkg["training/*"]
        TM["training.models"]
        TP["training.preprocessing"]
        TT["training.training"]
        TE["training.explainability"]
        TDM["training.dataset_manager"]
        TS["training.stam"]
        TQ["training.quality"]
        TM2["training.mlops"]
    end

    subgraph SharedPkg["shared/*"]
        SH["shared.schemas · dto · interfaces · validation"]
    end

    CORE --> CORE_P["core.paths (sys.path bootstrap)\n adds backend root, application root, repo root"]
    API --> SVC
    SVC --> MOD
    MOD --> ENG
    ENG --> TM
    ENG --> TP
    ENG --> TE
    SVC --> DB
    CORE --> DB

    TM2 --> TQ
    TM2 --> TM
    TQ --> TM

    TM --> SH
    TP --> SH
    TDM --> SH
    TQ --> SH
    SVC --> SH
    DB --> SH

    style TM2 fill:#eef,stroke:#66a
    style SH fill:#efe,stroke:#6a6
