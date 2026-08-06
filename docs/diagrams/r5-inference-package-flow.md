# R5 Inference Package Flow

```mermaid
flowchart TD
    MODEL["Trained CropFusionModel"]
    BATCH["sample_batch (or built from model.config)"]
    INFCFG["InferenceConfig<br/>INF_* env > YAML > defaults"]
    META["metadata<br/>framework · framework_version"]

    MODEL --> EX["ModelExporter.export_bundle"]
    BATCH --> EX
    INFCFG --> EX
    EX --> PT["cropfusion.pt (self-describing)"]
    EX --> TS["cropfusion.torchscript.pt (optional)"]
    EX --> ONNX["cropfusion.onnx (optional)"]
    EX --> SIDE["model_config.yaml · metrics.json · metadata.json · checksums.json"]

    PT --> VER["versioning<br/>semver bump · fingerprint conflict check"]
    PT --> BUILD["PackageBuilder.build"]
    TS --> BUILD
    ONNX --> BUILD
    SIDE --> BUILD
    INFCFG --> BUILD
    META --> BUILD
    VER --> BUILD

    BUILD --> PKG["package_dir/ cropfusion-<version>/<br/>14 required artifacts + manifest.json<br/>+ requirements.txt · api.py · README.md"]
    PKG --> VAL["validate_package(package_dir)"]
    VAL --> INT["integrity — SHA-256 vs checksums.json"]
    VAL --> MAN["manifest — version · formats · artifacts"]
    VAL --> COM["compatibility — fingerprint · param count"]
    VAL --> SMK["smoke_test — batch from model.config · forward"]
    INT --> RESULT["validation_report.md/.json<br/>all green ⇒ usable package"]
    MAN --> RESULT
    COM --> RESULT
    SMK --> RESULT

    style EX fill:#e8f5e9
    style BUILD fill:#e8f5e9
    style VAL fill:#fff3e0
    style RESULT fill:#e3f2fd
```

`manifest.json` lists artifact checksums plus the formats; both
`checksums.json` and `manifest.json` exclude themselves from the checksum map.
The `pytorch` format is always required; `torchscript` / `onnx` are optional
per `exporter.formats`.
