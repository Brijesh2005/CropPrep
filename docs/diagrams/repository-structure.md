# Repository Structure Diagram

```mermaid
flowchart TB
    ROOT["cropfusion/ (repo root)"]

    ROOT --> TRAIN["training/ — Training Platform"]
    ROOT --> APP["application/ — Prediction Platform"]
    ROOT --> SHARED["shared/ — Contracts"]
    ROOT --> DOCS["docs/"]
    ROOT --> REL["releases/"]
    ROOT --> SCR["scripts/"]
    ROOT --> GH[".github/"]
    ROOT --> MISC["datasets/ · Tabular_Datasets/ · cropfusion/ · root configs"]

    TRAIN --> T1["models/"]
    TRAIN --> T2["preprocessing/"]
    TRAIN --> T3["training/"]
    TRAIN --> T4["explainability/"]
    TRAIN --> T5["dataset_manager/"]
    TRAIN --> T6["stam/"]
    TRAIN --> T7["quality/"]
    TRAIN --> T8["mlops/"]
    TRAIN --> T9["feature_engineering/ · evaluation/ · experiments/ · export/ · hyperparameter_search/ · config/ · tests/"]
    TRAIN --> T10["kaggle/"]

    APP --> A1["backend/"]
    A1 --> A1A["app/ (FastAPI package)"]
    A1 --> A1B["datasets/ (runtime)"]
    APP --> A2["frontend/"]
    APP --> A3["database/"]
    APP --> A4["gis/"]
    APP --> A5["monitoring/"]
    APP --> A6["docker/"]
    APP --> A7["config/"]
    APP --> A8["tests/"]
    APP --> A9["inference/ · authentication/ · history/ · admin/ · models/ · inference_package/"]

    SHARED --> S1["schemas/ · dto/ · enums/ · interfaces/"]
    SHARED --> S2["validation/ · utils/ · exceptions/ · config/ · constants/ · serialization/"]
    SHARED --> S3["tests/"]
