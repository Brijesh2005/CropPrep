# R1.3 Configuration Resolution

```mermaid
flowchart TB
    ENV["Environment variables<br/>&lt;PREFIX&gt;&lt;SECTION&gt;__&lt;KEY&gt;<br/>DM_SCAN__WORKERS=16"] --> PARSER["shared.config.parse_env<br/>(JSON/bool/number parsing)"]

    YAML["YAML config file<br/>(platform pydantic settings)"] --> LOAD["shared.config.load_yaml_config"]
    PARSER --> LOAD

    LOAD --> MERGE["shared.config.deep_merge<br/>(env overrides YAML)"]
    MERGE --> CASI["shared.config.apply_case_insensitive<br/>(match pydantic field names)"]
    CASI --> VALIDATE["pydantic model_validate<br/>(typed, extra=forbid)"]
    VALIDATE --> SETTINGS["Settings instance"]

    SRC["any value needing YAML-safe conversion<br/>(Path · torch tensor · numpy scalar)"] --> YSAFE["shared.utils.yaml_safe<br/>(.item() for tensors/scalars)"]
    YSAFE --> YAML

    style ENV fill:#e8f5e9
    style YAML fill:#e8f5e9
    style SETTINGS fill:#e3f2fd
    style YSAFE fill:#fce4ec
```

Consumers:

| Consumer | Prefix | Loader |
| --- | --- | --- |
| `training/dataset_manager/config.py` | `DM_` | `load_settings()` |
| `training/training/config.py` | `TD_` | `load_training_config()` |
| `training/stam/config.py` | `ST_` | `load_stam_config()` |
| `training/preprocessing/config.py` | `PPT_` | `load_preprocessing_config()` |
| `training/explainability/config.py` | `EXP_` | `load_explainability_config()` |
| `application/backend/app/core/config.py` | `BACKEND_` | `Settings` bootstrap |

All six share `parse_env → deep_merge → apply_case_insensitive` from
`shared.config`; each adds only its own pydantic model. Before R1.3 the four
`_yaml_safe` copies and six private helper copies lived in these files.
