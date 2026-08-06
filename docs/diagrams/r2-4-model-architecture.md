# R2.4 CropFusion Model Architecture

```mermaid
flowchart TB
    subgraph Input["Phase 4 batch dict"]
        TAB["tabular [B, F]"]
        NDVI["ndvi [B, T, 1, H, W]"]
        EVI["evi [B, T, 1, H, W]"]
        MASK["temporal_mask [B, T]"]
    end

    subgraph Tabular["Tabular branch"]
        TT["TabTransformer<br/>(cls token + OOV)"]
        TEB["tabular_embedding [B, D_tab]"]
    end

    subgraph Image["Image branch"]
        N["NdviEncoder<br/>(EfficientNetV2-S)"]
        E["EviEncoder<br/>(EfficientNetV2-S)"]
        IF["ImageFusion<br/>(per-timestep)"]
        TE["TemporalTransformer<br/>(masked, cls)"]
        IEB["image_embedding [B, D_img]"]
    end

    subgraph Fusion["CrossModalFusionEngine"]
        CA["CrossAttention<br/>(Q=image, K=V=tabular)"]
        AGF["AdaptiveGatedFusion<br/>(image · tabular · fusion [· temporal] gates)"]
        SE["SharedMultimodalEncoder<br/>(cls-pooled transformer)"]
    end

    subgraph Heads["MultiTaskHeads"]
        CH["CropHead → crop_logits [B, C]"]
        YH["YieldHead → yield_pred [B, 1]"]
    end

    TAB --> TT --> TEB --> AGF
    NDVI --> N --> IF
    EVI --> E --> IF --> TE --> IEB --> CA
    TEB --> CA
    CA --> AGF
    MASK --> TE
    AGF --> SE --> CH
    SE --> YH

    style Fusion fill:#e8f5e9
    style Tabular fill:#fff3e0
    style Image fill:#fff3e0
    style Heads fill:#e3f2fd
```

Single-modality models bypass the engine: `tabular → SharedMultimodalEncoder`
or `image → SharedMultimodalEncoder` (the `shared_encoder` property hides which
path is active).
