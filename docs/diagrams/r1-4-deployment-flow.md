# R1.4 Deployment Flow

```mermaid
flowchart TB
    subgraph B["Dockerfile.backend (unchanged)"]
        B1["python:3.12-slim"]
        B2["+ training/ + shared/"]
        B3["+ backend + database"]
        B4["uvicorn app.main:app"]
        B1 --> B2 --> B3 --> B4
    end

    subgraph S["Dockerfile.inference.standalone (new)"]
        S1["python:3.12-slim"]
        S2["+ shared/ only (no training/)"]
        S3["+ inference · gis · history"]
        S4["+ inference_package · models · config"]
        S5["placeholder entrypoint (verify skeleton)"]
        S1 --> S2 --> S3 --> S4 --> S5
    end

    subgraph RUN["inference worker (future)"]
        R1["mount models/ + inference_package/ read-only"]
        R2["validate package → load → serve"]
        R1 --> R2
    end

    B -->|"current"| API["FastAPI + embedded engine"]
    S -->|"future"| RUN

    style S fill:#fff3e0
    style RUN fill:#e8f5e9
```

- `Dockerfile.inference.standalone` ships no training packages; the exported
  artifacts are mounted at run time.
- Build: `docker build -f application/docker/Dockerfile.inference.standalone -t cropfusion/inference-standalone .`
