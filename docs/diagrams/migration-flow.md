# Migration Flow Diagram

```mermaid
flowchart LR
    OLD["Monolithic layout"] --> M1["Move ai/* → training/*"]
    OLD --> M2["Move services/* → training/* (spatial_alignment → stam)"]
    OLD --> M3["Move quality/, mlops/ → training/"]
    OLD --> M4["Move backend/ → application/ (app/ + database/ + datasets/)"]
    OLD --> M5["Move frontend/, gis/, tests/ → application/"]
    OLD --> M6["Move deployment/, nginx/, Docker*, compose → application/docker + application/monitoring"]
    OLD --> M7["Move research/ → docs/research/, .env examples → application/config/"]

    M1 --> R1["Rewrite imports: ai. → training., services. → training.,\n quality. → training.quality., mlops. → training.mlops."]
    M2 --> R1
    M3 --> R1

    R1 --> B["Bootstrap: paths.py + conftest.py\n add training/, application/, shared/ to sys.path"]
    M4 --> B

    B --> INFRA["Update infra: pyproject · pytest.ini · Makefile ·\n .gitignore · Dockerfiles · compose · GitHub Actions"]
    INFRA --> SKEL["Create shared/ skeleton + training/ & application/ placeholders"]
    SKEL --> DOCS["Reorganise docs/ · READMEs · guides · diagrams · migration report"]
    DOCS --> VER["Verify: compile · stale-import grep · sanity imports"]
    VER --> DONE["Two-platform repository ✔"]
