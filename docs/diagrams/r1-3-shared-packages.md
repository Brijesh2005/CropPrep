# R1.3 Shared Package Layout

```mermaid
flowchart LR
    subgraph SHARED["shared/  (version 0.1.0)"]
        CFG["config<br/>deep_merge · parse_env<br/>apply_case_insensitive<br/>load_yaml_config"]
        CNS["constants<br/>dirs · extensions · CRS<br/>env prefixes · providers"]
        ENM["enums<br/>IndexType · Resolution · Severity<br/>CropType · Season · Status · ..."]
        EXC["exceptions<br/>CropFusionError<br/>+ 30 domain errors (CF-*)"]
        INT["interfaces<br/>Provider · Repository · Cache<br/>Storage · Exporter · ..."]
        LOG["logging<br/>setup_logging · profiles<br/>JSON/compact/colored formatters"]
        SCH["schemas<br/>Dataset · Image · Prediction<br/>Validation · Config · Training"]
        SER["serialization<br/>registry · json/yaml/pickle<br/>parquet/csv/numpy/torch"]
        TYP["types"]
        UTL["utils<br/>yaml_safe · sha256_file<br/>is_geotiff · classify_*"]
        VAL["validation<br/>Validator · registry<br/>csv/image/metadata/config/version"]
        VER["versioning<br/>SemanticVersion · VersionInfo<br/>Dataset/Model/Inference/App"]
        DTO["dto"]
    end

    SHARED --> UTL
    SHARED --> ENM
    SHARED --> EXC

    style SHARED fill:#e3f2fd
    style UTL fill:#e8f5e9
    style ENM fill:#e8f5e9
    style EXC fill:#e8f5e9
```

- `utils`, `enums` and `exceptions` are the leaf-most packages; everything else
  in `shared/` builds on them.
- `config`, `serialization`, `validation`, `logging` and `versioning` import
  `enums`/`exceptions`/`utils` but never each other's private internals.
