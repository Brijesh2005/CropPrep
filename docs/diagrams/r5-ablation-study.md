# R5 Ablation Study

```mermaid
flowchart LR
    BASE["base ModelConfig"]
    REG["DEFAULT_VARIANTS registry<br/>7 variants"]
    LOAD["evaluation loader"]

    BASE --> BUILD["build_variant_config(variant)"]
    REG --> BUILD
    BUILD --> A["build_variant_model<br/>config surgery + apply_variant_surgery"]

    subgraph variants
        A --> V1["without_tabtransformer"]
        A --> V2["without_efficientnet<br/>(+use_temporal_stream=false)"]
        A --> V3["without_temporal_encoder<br/>(_TemporalPooling swap)"]
        A --> V4["without_cross_attention"]
        A --> V5["without_adaptive_gate"]
        A --> V6["without_confidence_fusion<br/>(gating + residual off)"]
        A --> V7["without_temporal_branch"]
    end

    V1 --> SW["AblationStudy.run(loader, variants)"]
    V2 --> SW
    V3 --> SW
    V4 --> SW
    V5 --> SW
    V6 --> SW
    V7 --> SW
    LOAD --> SW

    SW --> EV["evaluate each variant<br/>full EvaluationOutcome"]
    EV --> BM["benchmark<br/>params · inference ms"]
    EV --> C["compare to full model<br/>compare_metric · compare_mode"]
    BM --> C
    C --> BEST["best_variant"]
    C --> REP["ablation_report.md/.json<br/>parameter table · metric table · winner"]

    style SW fill:#e8f5e9
    style REP fill:#e3f2fd
```

`variants=None` sweeps all seven; a subset can be passed explicitly. Both
`build_variant_config` and `apply_variant_surgery` are public so a single
variant can be built and inspected without running the sweep.
