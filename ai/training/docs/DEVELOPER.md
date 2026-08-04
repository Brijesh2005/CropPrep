# Developer Guide

## Architecture & dependencies

```
ai/training (ports: interfaces.py)
   ├── config.py        TrainingConfig — one pydantic section per subsystem
   ├── losses.py        MultiTaskLoss + GradNormController
   ├── optimizers.py    build_optimizer factory + self-contained Lion
   ├── schedulers.py    build_scheduler -> SchedulerHandle
   ├── metrics.py       MetricsTracker / accumulators
   ├── validator.py     Validator + fold generators
   ├── checkpoint.py    TrainingCheckpointManager (wraps Phase 5 manager)
   ├── callbacks.py     Callback implementations
   ├── logger.py        ExperimentLogger (CSV/JSON/git/config)
   ├── trainer.py       Trainer (the engine)
   ├── evaluator.py     Evaluator + Benchmark
   ├── experiment.py    Experiment orchestrator
   ├── ablation.py      AblationRunner
   ├── benchmark.py     Benchmark
   └── visualizer.py    Visualizer
```

The package depends on the **Phase 5 model** (`ai.models`) and the **Phase 4
preprocessing** (`ai.preprocessing`). It never reads files directly — it
consumes ready-made Phase 4 batches.

## Design principles

* **Dependency injection** — `Trainer`, `Validator`, `Evaluator`, `Experiment`
  take their collaborators (model, loaders, loss, optimizer, scheduler,
  callbacks, logger, checkpoint manager) as constructor arguments. Nothing is
  constructed from hidden globals.
* **Ports & adapters** — `Callback`, `FoldGenerator`, `SchedulerHandle` and
  `Weighter` are the extension points in `interfaces.py`.
* **Config-first** — every behaviour is expressible through
  `TrainingConfig`; the code reads the config, never the other way.
* **Graceful degradation** — AMP, DDP, TensorBoard, W&B, sklearn and
  matplotlib all fall back to a sensible CPU / no-op path when unavailable.
* **Backward compatibility** — the Phase 5 model gained **additive** ablation
  toggles (`enable_ndvi`, `enable_evi`, `cross_attention.enabled`,
  `gated_fusion.enabled`). The full model remains the default; all Phase 5
  behaviour and tests are unchanged.

## Adding a callback

```python
from ai.training import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # logs: dict with train_loss, val_loss, lr, per-task metrics ...
        pass
```

Register it by passing `callbacks=[MyCallback()]` to `Trainer`, or add it to
`Trainer._build_callbacks` if it should be config-driven.

## Adding a loss

Extend `ai.models.losses` (per-task criteria) or add a new weighting strategy
to `MultiTaskLoss` / `GradNormController` in `ai/training/losses.py`. A new
task head plugs in through `model.add_head(...)` (Phase 5) and its target key
through the input map.

## Distributed training

`Trainer` wraps the model with `DistributedDataParallel` when
`torch.distributed` is initialized and `world_size > 1`. Launch with:

```bash
torchrun --nproc-per-node=2 -m my_script
```

The utilities in `ai/training/utils.py` (`setup_distributed`,
`cleanup_distributed`, `all_gather_tensor`, `broadcast_dict`, `is_primary`)
handle rank detection, metric reduction and artifact writing (only rank 0
writes shared files). On a machine without CUDA (or without
`RANK`/`WORLD_SIZE`), everything degrades to a single CPU process.

## Testing

```bash
python -m pytest ai/training -q
```

Coverage: **85%** (trainer 86%, experiment 91%, validator 93%, losses 89%).
The fast suite uses tiny tabular-only models; integration tests build a real
STAM chain over the synthetic dataset.

## Conventions

* Python 3.12+, PyTorch 2.x, PEP 8, type hints, docstrings, SOLID.
* Errors raise `TrainingError` subclasses with stable `TR-<AREA>-<NNN>` codes.
* New code ships with tests; no TODOs or placeholder implementations.
