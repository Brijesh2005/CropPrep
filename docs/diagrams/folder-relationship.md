# Folder Relationship Diagram

```mermaid
flowchart TB
    subgraph L1["Level 1 — platform roots"]
        TRAIN["training/"]
        APP["application/"]
        SHARED["shared/"]
    end

    subgraph L2["Level 2 — top-level platform folders"]
        TRAIN2["models · preprocessing · training · explainability\n dataset_manager · stam · quality · mlops\n feature_engineering · evaluation · experiments\n export · hyperparameter_search · kaggle\n config · tests"]
        APP2["backend · frontend · database · gis\n monitoring · docker · config · tests\n inference · authentication · history\n admin · models · inference_package"]
        SHARED2["schemas · dto · enums · interfaces\n validation · utils · exceptions\n config · constants · serialization · tests"]
    end

    subgraph L3["Level 3 — examples of nested packages"]
        B3["application/backend/app\n (FastAPI package: api · core · modules · services …)"]
        M3["training/models\n (networks · losses · exporter …)"]
        Q3["training/quality\n (drift · fairness · optimization · monitoring)"]
    end

    TRAIN --> TRAIN2
    APP --> APP2
    SHARED --> SHARED2
    APP2 --> B3
    TRAIN2 --> M3
    TRAIN2 --> Q3

    TRAIN2 -. "depends on" .-> SHARED
    APP2 -. "depends on" .-> SHARED
