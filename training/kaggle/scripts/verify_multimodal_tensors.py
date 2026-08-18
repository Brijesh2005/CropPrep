"""R5.2.1 Task F: Multimodal tensor trace through the full fusion pipeline.

Traces tensors at every boundary in the CropFusion forward pass:
  1. Tabular embedding (TabTransformer)
  2. NDVI/EVI encoder outputs (EfficientNetV2-S)
  3. Image fusion (NdviEvi fusion)
  4. Temporal transformer output
  5. Cross-attention (Q=image, K/V=tabular)
  6. Adaptive gated fusion
  7. Shared encoder output
  8. Crop head logits
  9. Yield head predictions
 10. Gradient flow through all components

Reports: shape, dtype, min/max, mean, NaN/Inf counts, gradient norm per component.

Run from Kaggle training kernel (needs imagery)::

    python training/kaggle/scripts/verify_multimodal_tensors.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/multimodal_trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.models.config import ModelConfig  # noqa: E402
from training.models.factory import ModelFactory  # noqa: E402
from training.models.cropfusion import CropFusionModel  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config, split_observations  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402
from training.training.config import load_training_config  # noqa: E402
from training.training.losses import MultiTaskLoss, build_class_weights  # noqa: E402


def _stats(t: torch.Tensor, name: str) -> dict[str, Any]:
    t_f = t.detach().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "min": float(t_f.min().item()),
        "max": float(t_f.max().item()),
        "mean": float(t_f.mean().item()),
        "std": float(t_f.std().item()),
        "nan": int(torch.isnan(t_f).sum().item()),
        "inf": int(torch.isinf(t_f).sum().item()),
        "zero_frac": float((t_f == 0).float().mean().item()),
        "finite": bool(torch.isfinite(t_f).all().item()),
    }


def _trace_forward(model: CropFusionModel, inputs: dict[str, Any]) -> dict[str, Any]:
    """Trace through each component of CropFusionModel."""
    trace: dict[str, Any] = {}
    hooks: dict[str, torch.Tensor] = {}

    # Hook into each component
    def _hook(name: str):
        def fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                hooks[name] = out
            elif isinstance(out, dict):
                hooks[name] = out.get("fused", out.get("output", list(out.values())[0]))
            elif hasattr(out, "shared_embedding"):
                hooks[name] = out.shared_embedding
            elif hasattr(out, "crop_logits") and out.crop_logits is not None:
                hooks[name] = out.crop_logits
        return fn

    registered = []

    # Register hooks
    if model.tab_encoder is not None:
        registered.append(model.tab_encoder.register_forward_hook(_hook("tab_encoder")))
    if model.ndvi_encoder is not None:
        registered.append(model.ndvi_encoder.register_forward_hook(_hook("ndvi_encoder")))
    if model.evi_encoder is not None:
        registered.append(model.evi_encoder.register_forward_hook(_hook("evi_encoder")))
    if model.image_fusion is not None:
        registered.append(model.image_fusion.register_forward_hook(_hook("image_fusion")))
    if model.temporal_transformer is not None:
        registered.append(model.temporal_transformer.register_forward_hook(_hook("temporal_transformer")))
    if model.cross_attention is not None:
        registered.append(model.cross_attention.register_forward_hook(_hook("cross_attention")))
    if model.gated_fusion is not None:
        registered.append(model.gated_fusion.register_forward_hook(_hook("gated_fusion")))
    if model.shared_encoder is not None:
        registered.append(model.shared_encoder.register_forward_hook(_hook("shared_encoder")))

    # Forward
    with torch.enable_grad():
        out = model(inputs)

    # Record stats for each hook
    for name, tensor in hooks.items():
        trace[name] = _stats(tensor, name)
        print(f"  {name:30s} shape={list(tensor.shape):30s} "
              f"nan={trace[name]['nan']} inf={trace[name]['inf']} "
              f"min={trace[name]['min']:.6g} max={trace[name]['max']:.6g}")

    # Record output heads
    if out.crop_logits is not None:
        trace["crop_logits"] = _stats(out.crop_logits, "crop_logits")
        print(f"  {'crop_logits':30s} shape={list(out.crop_logits.shape):30s} "
              f"nan={trace['crop_logits']['nan']} inf={trace['crop_logits']['inf']}")
    if out.yield_pred is not None:
        trace["yield_pred"] = _stats(out.yield_pred, "yield_pred")
        print(f"  {'yield_pred':30s} shape={list(out.yield_pred.shape):30s} "
              f"nan={trace['yield_pred']['nan']} inf={trace['yield_pred']['inf']}")
    if out.shared_representation is not None:
        trace["shared_repr"] = _stats(out.shared_representation, "shared_repr")

    # Cleanup
    for h in registered:
        h.remove()

    return trace, out


def _trace_backward(
    model: CropFusionModel,
    inputs: dict[str, Any],
    targets: dict[str, Any],
    loss_fn: MultiTaskLoss,
    device: torch.device,
) -> dict[str, Any]:
    """Run forward + backward and report gradient norms per component."""
    print("\n--- Gradient Flow Trace ---")
    model.train()
    grad_norms: dict[str, Any] = {}

    with torch.enable_grad():
        out = model(inputs)
        out_dict = {}
        if out.crop_logits is not None:
            out_dict["crop"] = out.crop_logits
        if out.yield_pred is not None:
            out_dict["yield"] = out.yield_pred
        total, per_task = loss_fn(out_dict, targets)
        total.backward()

    # Collect gradient norms per component
    components = {
        "tab_encoder": model.tab_encoder,
        "ndvi_encoder": model.ndvi_encoder,
        "evi_encoder": model.evi_encoder,
        "image_fusion": model.image_fusion,
        "temporal_transformer": model.temporal_transformer,
        "cross_attention": model.cross_attention,
        "gated_fusion": model.gated_fusion,
        "shared_encoder": model.shared_encoder,
        "heads": model.heads,
    }

    for name, module in components.items():
        if module is None:
            continue
        norm = 0.0
        nan_grads = 0
        total_params = 0
        for p in module.parameters():
            if p.grad is not None:
                norm += p.grad.data.norm(2).item() ** 2
                nan_grads += int(torch.isnan(p.grad).sum().item())
                total_params += p.numel()
        norm = norm ** 0.5
        grad_norms[name] = {"norm": norm, "nan_grads": nan_grads, "params": total_params}
        status = "OK" if nan_grads == 0 and norm > 0 else ("DEAD" if norm == 0 else "NaN!")
        print(f"  {name:30s} grad_norm={norm:10.4f} nan_grads={nan_grads:4d} {status}")

    return {
        "loss": float(total.item()),
        "per_task": {k: float(v.item()) for k, v in per_task.items()},
        "gradient_norms": grad_norms,
        "any_nan_grads": any(g["nan_grads"] > 0 for g in grad_norms.values()),
        "any_zero_grads": any(g["norm"] == 0 for g in grad_norms.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-verify-multimodal-tensors",
        description="R5.2.1 Task F: trace tensors through full CropFusion pipeline",
    )
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--config",
                        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"))
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MULTIMODAL TENSOR TRACE (device={device}) ===")

    # Load observations
    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    obs = [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]
    print(f"Loaded {len(obs)} accepted observations")

    # Preprocessor
    pre = Preprocessor(load_preprocessing_config(args.config))
    train_obs, _, _ = split_observations(obs, pre.config.split)
    accepted, _ = pre.filter(train_obs)
    pre.fit(accepted)

    # Model
    mc = ModelConfig.from_preprocessor(pre)
    model = ModelFactory.create(mc)
    model.to(device)

    # Loss
    cfg = load_training_config(str(_REPO_ROOT / "training" / "config" / "training.yaml"))
    counts = torch.tensor([64.0, 7.0, 1.0, 1.0, 1.0])
    loss_fn = MultiTaskLoss(
        cfg.loss,
        class_weights={
            "crop": build_class_weights(cfg.loss, mc.heads.crop.num_classes, counts)
        },
    )
    loss_fn.to(device)

    # Build batch
    batch_samples = accepted[: args.batch_size]
    tabular = torch.stack([pre.tabular.transform(o).float() for o in batch_samples]).to(device)
    labels = [pre.label.transform(o) for o in batch_samples]
    crops = torch.stack([c for c, _y in labels]).to(device)
    yields = torch.stack([y for _c, y in labels]).to(device)

    # Image tensors (random for tabular-only; real with imagery on Kaggle)
    if mc.uses_image:
        seq_len = mc.temporal.max_len
        img_size = mc.image_encoder.input_size or 224
        ndvi = torch.randn(args.batch_size, seq_len, 1, img_size, img_size, device=device) * 0.1
        evi = torch.randn(args.batch_size, seq_len, 1, img_size, img_size, device=device) * 0.1
        mask = torch.ones(args.batch_size, seq_len, device=device)
        if seq_len > 1:
            mask[:, -1] = 0.0
        inputs = {"tabular": tabular, "ndvi": ndvi, "evi": evi, "temporal_mask": mask}
    else:
        inputs = {"tabular": tabular}

    targets = {"crop": crops, "yield": yields}

    # Phase 1: Forward trace
    print("\n--- Phase 1: Forward Trace ---")
    forward_trace, _ = _trace_forward(model, inputs)

    # Phase 2: Gradient trace
    grad_trace = _trace_backward(model, inputs, targets, loss_fn, device)

    # Summary
    nan_components = [k for k, v in forward_trace.items() if isinstance(v, dict) and not v.get("finite", True)]
    zero_grad_components = [k for k, v in grad_trace["gradient_norms"].items() if v["norm"] == 0]

    print(f"\n=== SUMMARY ===")
    print(f"  Components with NaN/Inf: {nan_components or 'NONE'}")
    print(f"  Components with zero gradients: {zero_grad_components or 'NONE'}")
    print(f"  Gradient NaN detected: {grad_trace['any_nan_grads']}")
    print(f"  Total loss: {grad_trace['loss']:.6f}")
    print(f"  Per-task: {grad_trace['per_task']}")

    all_pass = not nan_components and not grad_trace["any_nan_grads"]
    print(f"\n  RESULT: {'PASS' if all_pass else 'FAIL'}")

    report = {
        "forward_trace": forward_trace,
        "gradient_trace": grad_trace,
        "nan_components": nan_components,
        "zero_grad_components": zero_grad_components,
        "all_finite": all_pass,
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "multimodal_trace_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nWrote {out / 'multimodal_trace_report.json'}")

    return 0 if all_pass else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
