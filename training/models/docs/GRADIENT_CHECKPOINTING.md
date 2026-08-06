# Gradient checkpointing

Gradient (activation) checkpointing trades compute for memory: instead of
keeping every transformer-block activation alive for the backward pass, the
forward recomputes each block's activations on the fly during backprop. This
roughly halves the activation footprint of the transformer stacks at the cost of
one extra forward per block.

## Where it applies

CropFusion's three transformer stacks each expose `set_gradient_checkpointing`:

| module | attribute |
| --- | --- |
| `TabTransformer` | `model.tab_encoder` |
| `TemporalTransformer` | `model.temporal_transformer` |
| `SharedMultimodalEncoder` | `model.shared_encoder` |

The timm CNN backbones are **not** checkpointed (their activations are
significantly cheaper per element and their Block modules are not recomputed).

Checkpointing only activates in **training mode**; eval and export run the
normal path, so it never affects inference or traced graphs.

## Enabling

Three equivalent ways:

```python
# 1. Config (applied at creation)
cfg.runtime.gradient_checkpointing = True
model = ModelFactory.create_with_runtime(cfg)

# 2. Model method
model.enable_gradient_checkpointing(True)

# 3. Runtime helper / factory
from training.models import enable_gradient_checkpointing
enable_gradient_checkpointing(model, True)
# or ModelFactory.enable_gradient_checkpointing(model, True)
```

The flag is recorded on `model.config.runtime.gradient_checkpointing` and
surfaces in `model.metadata`, so a checkpoint records whether it was trained
with checkpointing on.

## Implementation notes

- Uses `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
- `TemporalTransformer` passes its key-padding mask through as a keyword
  argument (`src_key_padding_mask=...`) — passing it positionally would land on
  `src_mask` and change the attention behaviour.
- Gradients flow through every trainable parameter exactly as without
  checkpointing; the recomputed values are numerically identical.

## When to use it

Enable when activation memory is the bottleneck (long temporal sequences, large
patches, or big batches). Disable for maximal throughput on memory-rich
hardware.
