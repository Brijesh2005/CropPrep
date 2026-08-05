# Dependency Diagram

```mermaid
flowchart LR
    subgraph Training["Training Platform (training/)"]
        MODELS["training/models"]
        PRE["training/preprocessing"]
        ENG["training/training"]
        EXPL["training/explainability"]
        DM["training/dataset_manager"]
        STAM["training/stam"]
        QA["training/quality"]
        MLOPS["training/mlops"]
    end

    subgraph Application["Prediction Platform (application/)"]
        BACKEND["application/backend (app.*)"]
        DB["application/database (database.*)"]
        GIS["application/gis"]
        INF["application/inference"]
        MON["application/monitoring"]
    end

    subgraph Shared["Shared (shared/)"]
        SCH["shared/schemas"]
        DTO["shared/dto"]
        IF["shared/interfaces"]
        VAL["shared/validation"]
        SER["shared/serialization"]
    end

    MLOPS --> QA
    ENG --> MODELS
    ENG --> PRE
    EXPL --> MODELS
    PRE --> DM
    STAM --> DM
    PRE --> STAM

    BACKEND --> DB
    BACKEND --> GIS
    INF --> BACKEND

    Training --> Shared
    Application --> Shared
