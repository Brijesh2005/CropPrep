# R1.3 Migration Flow

Before (R1.2) — duplicated helpers across platforms:

```mermaid
flowchart TB
    subgraph Before["R1.2 — before"]
        DMc["dataset_manager/config.py<br/>_parse_env · _apply_case_insensitive<br/>_normalise_key · deep_merge · _yaml_safe"]
        TRc["training/config.py<br/>_parse_env · _apply_case_insensitive<br/>_yaml_safe"]
        STc["stam/config.py<br/>_parse_env · _apply_case_insensitive"]
        PPc["preprocessing/config.py<br/>_parse_env · _apply_case_insensitive"]
        EXc["explainability/config.py<br/>_parse_env · _apply_case_insensitive<br/>_yaml_safe"]
        BKc["application/backend/app/core/config.py<br/>_yaml_safe (imported training internals)"]
        LG["dataset_manager/logger.py<br/>stam/logger.py · preprocessing/logger.py<br/>duplicated JSON/compact formatters"]
        EN["dataset_manager/models.py<br/>local IndexType · Resolution · Severity · ..."]
        EXP["6 platform exceptions.py<br/>standalone base classes"]
    end
    style Before fill:#fce4ec
```

After (R1.3) — one source of truth in `shared/`:

```mermaid
flowchart LR
    subgraph After["R1.3 — after"]
        SC["shared/config/loader.py<br/>parse_env · apply_case_insensitive<br/>normalise_key · deep_merge · load_yaml_config"]
        SU["shared/utils/yaml.py<br/>yaml_safe (superset incl. .item())"]
        SL["shared/logging/formatters.py<br/>JsonFormatter · CompactFormatter · ColoredFormatter"]
        SE["shared/enums/<br/>IndexType · Resolution · Severity · ..."]
        SX["shared/exceptions/CropFusionError<br/>+ domain bases (CF-* codes)"]
    end

    DMc --> SC
    TRc --> SC
    STc --> SC
    PPc --> SC
    EXc --> SC
    BKc --> SC
    BKc --> SU
    LG --> SL
    EN --> SE
    EXP --> SX

    style After fill:#e8f5e9
```

Steps taken:

1. Extracted the duplicated config primitives into `shared.config.loader` and
   `shared.utils.yaml_safe`; rewired all six consumers to import them.
2. Extracted the JSON/compact log formatters into `shared.logging.formatters`;
   the three platform loggers now import from `shared`.
3. Moved the canonical enums into `shared.enums`;
   `training/dataset_manager/models.py` re-exports them (no local definitions).
4. Re-based the six platform exception hierarchies onto
   `shared.exceptions.CropFusionError`, preserving each platform's `code`
   prefix and adding `suggested_resolution`.
5. Added shared **tests** (101) and verified every platform suite stays green.
