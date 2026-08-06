# Model runtime (precision, device, compile, parallelism)

`training/models/runtime.py` applies **execution** settings to a built
`CropFusionModel`. It never changes the architecture — it is the deployment
counterpart to `ModelConfig.runtime` (`RuntimeConfig`).

```
RuntimeConfig
   │  precision (float32 / float16 / bfloat16)  → apply_precision
   │  device (cpu / cuda / cuda:N / mps)        → move_to_device
   │  gradient_checkpointing                    → enable_gradient_checkpointing
   │  compile + compile_mode                    → compile_model
   │  data_parallel                             → wrap_data_parallel
   │  distributed + local_rank                  → wrap_distributed
   ▼
apply_runtime(model)  — one call, fixed order
```

## RuntimeConfig fields

| field | default | effect |
| --- | --- | --- |
| `precision` | `float32` | parameter / activation dtype; `float16` / `bfloat16` for AMP |
| `device` | `null` | explicit device (`null` = auto CUDA-or-CPU) |
| `compile` | `false` | wrap in `torch.compile` |
| `compile_mode` | `default` | `default` / `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` |
| `gradient_checkpointing` | `false` | recompute transformer activations during backprop |
| `data_parallel` | `false` | `torch.nn.DataParallel` (single node, multi GPU) |
| `distributed` | `false` | `torch.distributed` DistributedDataParallel |
| `local_rank` | `null` | local rank for DDP (defaults to `get_rank()`) |

## Functions

### Precision

- `dtype_from_precision("bfloat16")` → `torch.bfloat16`;
  `precision_from_dtype` reverses it.
- `apply_precision(model, "bfloat16")` converts every floating-point parameter
  and buffer to the dtype, but keeps **LayerNorm / BatchNorm in float32** for
  numeric stability. Also records `config.runtime.precision`.
- `amp_context("bfloat16", "cpu")` is a context manager running a block under
  `torch.autocast`. `float32` is a no-op. An unsupported precision/device
  combination raises `ModelConfigurationError` instead of failing mid-forward.

> **Note:** when you convert weights to float16 you must also feed float16
> tensors (or run under `amp_context`) — plain float32 inputs against float16
> weights raise a dtype mismatch, as in any torch model.

### Device & compile

- `move_to_device(model, "cuda:0")` — explicit placement.
- `compile_model(model, mode="default", backend=None)` — `torch.compile`.
  `backend="eager"` is useful in tests (near-no-op); `inductor` is the default.
  Fails with `MissingDependencyError` when `torch.compile` is unavailable.

### Parallelism

- `enable_gradient_checkpointing(model, enabled=True)` — toggles recomputation
  on every stack exposing `set_gradient_checkpointing` (TabTransformer,
  TemporalTransformer, shared encoder). Only active in training mode, so eval
  and export are unaffected.
- `wrap_data_parallel(model)` — raises `ModelConfigurationError` when no CUDA
  device exists (a silent CPU fallback would mask the misconfiguration).
- `wrap_distributed(model, local_rank=...)` — requires an initialised process
  group; moves the model to the local rank's device.

### One-call entry points

- `apply_runtime(model, runtime=None)` — applies everything in a fixed order
  (precision → device → checkpointing → compile → data-parallel wrappers).
- `ModelFactory.create_with_runtime(config)` — build + configure in one call.
- Convenience methods on the model itself: `model.to_precision(...)`,
  `model.to_device(...)`, `model.enable_gradient_checkpointing(...)`,
  `model.compile(...)`.

## Errors

All failures raise `ModelError` subclasses with stable codes
(`MDL-CONFIG-001`, `MDL-DEP-001`); see `training/models/exceptions.py`.
