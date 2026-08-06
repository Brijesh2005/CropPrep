# CrossModalFusionEngine

The **fusion unit** of CropFusion: one `nn.Module` that owns every step between
the modality embeddings and the shared multimodal representation, so the whole
cross-modal pathway is a swappable, ablable unit.

## Responsibilities

`training/models/fusion_engine.py` — `CrossModalFusionEngine` owns:

1. **Cross attention** — the image embedding attends to the tabular embedding
   (`Q = image, K = V = tabular`) via `CrossAttention`.
2. **Adaptive gated fusion** — per-sample image / tabular / fusion gates (and an
   optional fourth **temporal** gate) via `AdaptiveGatedFusion`.
3. **Shared multimodal encoder** — a CLS-pooled transformer stack producing the
   final `[B, out_dim]` representation via `SharedMultimodalEncoder`.

```
image_embedding [B, D_img] ─┐
                            ▼
tabular_embedding [B, D_tab] ─► CrossAttention ─► gated fusion ─► shared encoder ─► [B, out_dim]
                                    │                    ▲
                              [B, out_dim]         (optional temporal stream)
```

## The FusionOutput contract

`CrossModalFusionEngine.forward(...)` returns a `FusionOutput` dataclass (also
exported from the package root):

| field | shape | meaning |
| --- | --- | --- |
| `shared_embedding` | `[B, out_dim]` | final multimodal representation (feeds all task heads) |
| `fused` | `[B, D_gate]` | gated fusion output (pre shared-encoder) |
| `cross_output` | `[B, D_cross]` | cross-attention output |
| `image_token` / `tabular_token` / `temporal_token` | `[B, D_gate]` | projected modality streams |
| `gates` | dict | per-sample gate values for explainability |

## Gates

`gates` always contains `image_gate`, `tabular_gate` and `fusion_gate`
(`[B, 1]`, bounded `[0, 1]`) when gated fusion is enabled. These are the
per-sample "modality weights" consumed by the explainability layer. When
`fusion.use_temporal_stream` is on, a `temporal_gate` is added. With
`return_attention=True`, `gates["cross_attention"]` carries the head-averaged
cross-attention weights `[B, 1, 1]`.

## Configuration (`ModelConfig.fusion`)

| field | default | effect |
| --- | --- | --- |
| `residual_fusion` | `true` | add the projected modality streams back to the gated output before the shared encoder (gated-plus-original token). Off = pure gated ablation. |
| `use_temporal_stream` | `false` | feed the raw temporal stream into the gated fusion as a fourth stream (adds `temporal_gate`). Requires the image branch. |

The engine is only built for multimodal models. Single-modality models skip it
and use a plain `SharedMultimodalEncoder` over their one stream; the model's
`shared_encoder` property hides which path is active.

## Ablations

Everything below is config-only — no code changes:

| Ablation | config |
| --- | --- |
| no cross-attention | `cross_attention.enabled: false` |
| no gated fusion (concat) | `gated_fusion.enabled: false` |
| no residual fusion | `fusion.residual_fusion: false` |
| temporal stream on | `fusion.use_temporal_stream: true` |
| both attention + gates off | set both `enabled: false` |

## Key APIs

```python
from training.models import CrossModalFusionEngine, FusionOutput, ModelFactory

model = ModelFactory.create(config)
engine: CrossModalFusionEngine = model.fusion_engine      # multimodal models
out: FusionOutput = engine(img, tab, temporal_embedding=None, return_attention=False)
shared = out.shared_embedding                             # [B, out_dim]
```

Backward compatibility: `model.cross_attention`, `model.gated_fusion` and
`model.shared_encoder` still resolve to the engine-owned components, so existing
callers keep working unchanged.
