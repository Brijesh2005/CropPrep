# Ablation Study (Phase R5)

`AblationStudy` sweeps the seven R5 variants over one base `ModelConfig` and
compares each variant's **performance, parameter count and inference speed**
against the full model — the quantitative story behind "which component
earns its parameters".

## The seven variants

| variant | what is removed / disabled |
| --- | --- |
| `without_tabtransformer` | tabular branch (tabular-only model — image only) |
| `without_efficientnet` | image branch (backbone + temporal stream — tabular only) |
| `without_temporal_encoder` | temporal transformer → mask-aware mean pooling (structural surgery) |
| `without_cross_attention` | cross-attention block (gated fusion stays) |
| `without_adaptive_gate` | adaptive gated fusion → concatenated streams |
| `without_confidence_fusion` | gating off **and** residual re-injection disabled |
| `without_temporal_branch` | the temporal stream (fourth gate) |

Two variants (image-only / tabular-only) need model-side constraints:
`without_efficientnet` also sets `fusion.use_temporal_stream=false` (the
temporal gate requires the image branch), and `without_temporal_encoder`
replaces `model.temporal_transformer` with a `_TemporalPooling` module of the
same input/output width so nothing downstream breaks.

## Running a study

```python
from training.evaluation import AblationStudy, EvaluationConfig

config = EvaluationConfig(
    ablation={
        "compare_metric": "crop/f1",
        "compare_mode": "max",
        "benchmark_iterations": 5,
        "benchmark_warmup": 2,
    }
)
report = AblationStudy(base_model, config).run(loader, variants=None)  # all 7

report.results[name]        # metrics + parameter_count/delta + inference_ms + speedup_vs_full
report.comparison           # multimodal comparison table
report.best_variant         # winner under compare_metric / compare_mode
```

`variants=None` runs the full `DEFAULT_VARIANTS` registry; a subset can be
passed explicitly. `build_variant_config` / `apply_variant_surgery` are public
so a variant can be built and inspected without running the sweep.

## Report

`generate_ablation_reports` writes `ablation_report.md` / `.json`: the
parameter/speed table, the metric comparison table and the winning variant.
