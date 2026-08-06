# R2.4 Cross-Modal Fusion Engine

```mermaid
flowchart LR
    subgraph Engine["CrossModalFusionEngine (fusion_engine.py)"]
        direction TB
        CA["CrossAttention<br/>[B, D_img] → [B, D_cross]<br/>return_attention → weights [B, 1, 1]"]
        AGF["AdaptiveGatedFusion<br/>gates: image · tabular · fusion<br/>(+ temporal when enabled)"]
        RF["residual_fusion?<br/>fused + image_token + tabular_token"]
        SE["SharedMultimodalEncoder<br/>→ [B, out_dim]"]

        CA --> AGF --> RF --> SE
    end

    IMG["image_embedding [B, D_img]"] --> CA
    TAB["tabular_embedding [B, D_tab]"] --> CA
    TAB --> AGF
    TMP["temporal_embedding [B, D_img]<br/>(fusion.use_temporal_stream)"] --> AGF

    SE --> OUT["FusionOutput<br/>shared_embedding · fused · cross_output<br/>tokens · gates"]

    style Engine fill:#e8f5e9
    style OUT fill:#e3f2fd
```

Config-driven ablations (all in `ModelConfig`):

| ablation | switch |
| --- | --- |
| no cross-attention | `cross_attention.enabled: false` (cross path = image identity) |
| no gated fusion | `gated_fusion.enabled: false` (concat into shared encoder) |
| no residual fusion | `fusion.residual_fusion: false` |
| fourth temporal stream | `fusion.use_temporal_stream: true` |
