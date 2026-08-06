# torch.compile Wiring

The base `Trainer` (and therefore `CropFusionTrainer`) can wrap the model with
`torch.compile` before the DDP wrapper, so compilation happens once and the
optimizer / checkpoint manager operate on the compiled graph.

## Configuration

```yaml
general:
  compile: true
  compile_mode: default            # default | reduce-overhead | max-autotune | max-autotune-no-cudagraphs
  compile_backend: eager           # Optional; inductor default, eager for testing
```

Env override: `TRN_GENERAL__COMPILE`, `TRN_GENERAL__COMPILE_MODE`,
`TRN_GENERAL__COMPILE_BACKEND`.

* `compile: false` (default) — no wrapper, identical behaviour to earlier
  releases.
* `compile_backend: "eager"` — a near-no-op wrapper useful for exercising the
  wiring without a full compile (used by the test-suite).
* On PyTorch builds without `torch.compile`, requesting it raises a clean
  `MissingDependencyError` instead of silently continuing.

## What happens

1. `Trainer.__init__` keeps the original model as `self.raw_model`.
2. When `general.compile` is set, `self.raw_model` is replaced by
   `torch.compile(raw_model, mode=..., backend=...)`.
3. `self.model` is then the DDP wrapper (or the compiled model on single
   device), exactly as before.
4. The optimizer, `ModelCheckpoint` and resume path all reference
   `self.raw_model`, so checkpoint state stays weights-only-loadable.

## Usage

```python
from training.training import CropFusionTrainer, TrainingConfig

config = TrainingConfig(
    general={"compile": True, "compile_mode": "reduce-overhead"},
    train={"epochs": 50},
)

trainer = CropFusionTrainer(model, train_loader, config)
result = trainer.train()
```

Everything else — AMP, gradient handling, schedulers, checkpoints, callbacks,
reports — is untouched. Dropping `compile: true` back to `false` is the
fastest way to confirm a discrepancy came from compilation.
