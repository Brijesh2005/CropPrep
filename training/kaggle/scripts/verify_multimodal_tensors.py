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

The script fixes the R5.2 review findings on the earlier verify script:

* **Real split** — the batch is drawn from the **frozen taluk train split**
  recorded in each observation's provenance (``provenance.split``), never a
  re-split of the whole accepted corpus.
* **Real patches** — imagery tensors come from the real patch extractor
  (``STAM.get_patch`` through the Phase 4 preprocessor), not ``randn`` stubs.
  When imagery is unavailable the script **aborts loudly** (exit 2) instead of
  silently substituting fake tensors.
* **Fitted class counts** — loss class weights are built from the fitted label
  encoder counts over the train split, not the hard-coded ``[64, 7, 1, 1, 1]``.
* **Temporal length** — the sequence length is ``temporal.max_observations``
  (8 per config), not the model's ``max_len`` (16).
* **Checkpointing** — gradient checkpointing is enabled via the unified
  ``apply_gradient_checkpointing`` path so the OOM-avoiding per-timestep
  checkpointing is what the trace exercises.

Run from Kaggle training kernel (imagery attached)::

    python training/kaggle/scripts/verify_multimodal_tensors.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/multimodal_trace

Run on a research box with no imagery to confirm the loud downgrade::

    python training/kaggle/scripts/verify_multimodal_tensors.py --corpus <path>
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.models.config import ModelConfig  # noqa: E402
from training.models.cropfusion import CropFusionModel  # noqa: E402
from training.models.factory import ModelFactory  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.preprocessing.dataset import CropFusionDataset  # noqa: E402
from training.preprocessing.dataloader import build_dataloader  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402
from training.training.config import load_training_config  # noqa: E402
from training.training.diagnostics import assert_image_batch_shape, profile_batch  # noqa: E402
from training.training.losses import MultiTaskLoss, build_class_weights  # noqa: E402
from training.training.utils import apply_gradient_checkpointing  # noqa: E402


def _shape_str(shape: Any) -> str:
    """Render a tensor shape list as ``[4, 8, 1, 224, 224]`` for format columns.

    A list must be stringified BEFORE an alignment width spec — applying
    ``:30s`` to a raw ``list`` raises ``TypeError: unsupported format string
    passed to list.__format__`` (the R5.3 formatter bug).
    """
    return "[" + ", ".join(str(int(d)) for d in shape) + "]"


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


def _trace_forward(model: CropFusionModel, inputs: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """Trace through each component of CropFusionModel (no autograd graph)."""
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
    out = model(inputs)

    # Record stats for each hook
    for name, tensor in hooks.items():
        trace[name] = _stats(tensor, name)
        shape_str = _shape_str(tensor.shape)
        print(f"  {name:30s} shape={shape_str:26s} "
              f"nan={trace[name]['nan']} inf={trace[name]['inf']} "
              f"min={trace[name]['min']:.6g} max={trace[name]['max']:.6g}")

    # Record output heads
    if out.crop_logits is not None:
        trace["crop_logits"] = _stats(out.crop_logits, "crop_logits")
        shape_str = _shape_str(out.crop_logits.shape)
        print(f"  {'crop_logits':30s} shape={shape_str:26s} "
              f"nan={trace['crop_logits']['nan']} inf={trace['crop_logits']['inf']}")
    if out.yield_pred is not None:
        trace["yield_pred"] = _stats(out.yield_pred, "yield_pred")
        shape_str = _shape_str(out.yield_pred.shape)
        print(f"  {'yield_pred':30s} shape={shape_str:26s} "
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
    *,
    use_amp: bool = False,
) -> dict[str, Any]:
    """Run forward + backward and report gradient norms per component.

    ``use_amp`` mirrors the R5.2 trainer's FP16 GradScaler path on CUDA — the
    exact environment where per-timestep checkpointing + cuDNN interplay matter
    — so the gradient trace exercises the same precision/casting conditions as
    a real training step.
    """
    print("\n--- Gradient Flow Trace ---")
    model.train()
    grad_norms: dict[str, Any] = {}

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    with autocast_ctx:
        out = model(inputs)
        out_dict = {}
        if out.crop_logits is not None:
            out_dict["crop"] = out.crop_logits
        if out.yield_pred is not None:
            out_dict["yield"] = out.yield_pred
        total, per_task = loss_fn(out_dict, targets)
        scaler.scale(total).backward()

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


def _load_observations(corpus_path: Path) -> list[AgriculturalObservation]:
    """Load accepted observations from an ObservationCorpus JSON."""
    raw = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
    return [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]


def _split_from_provenance(
    observations: list[AgriculturalObservation],
) -> tuple[list[AgriculturalObservation], list[AgriculturalObservation], list[AgriculturalObservation]]:
    """Use the frozen taluk split recorded in ``provenance.split``.

    This is the authoritative split: the corpus was built from the frozen CSV
    with a location-aware (taluk) train/val/test split. Re-splitting at verify
    time (the old behaviour) produced the 8601/0/1518 contradiction and would
    let train/val leak across the still-served 2025 images.
    """
    train: list[AgriculturalObservation] = []
    val: list[AgriculturalObservation] = []
    test: list[AgriculturalObservation] = []
    for obs in observations:
        split = obs.provenance.get("split", "unknown")
        if split == "train":
            train.append(obs)
        elif split == "val":
            val.append(obs)
        elif split == "test":
            test.append(obs)
        else:
            train.append(obs)
    return train, val, test


def _resolve_extractor(args: argparse.Namespace) -> Any:
    """Resolve the real patch extractor (STAM.get_patch) or raise loudly."""
    if args.extractor_module:
        module = importlib.import_module(args.extractor_module)
        extractor = module.build_extractor()  # type: ignore[attr-defined]
        print(f"Extractor from module: {args.extractor_module}")
        return extractor

    try:
        from training.dataset_manager import DatasetManager, load_settings
        from training.stam import STAM
        from training.stam.config import load_stam_config
    except ImportError as exc:  # pragma: no cover
        print(
            "\n[FATAL] Cannot import DatasetManager/STAM for the patch extractor:\n"
            f"  {exc}\n"
            "  A patch extractor is REQUIRED for real image tensors.\n"
            "  Provide --extractor-module or run with imagery available.\n"
        )
        raise SystemExit(2)

    manager = DatasetManager(load_settings(Path(args.dataset_config)))
    stam = STAM(manager, load_stam_config(Path(args.stam_config)))
    stam.initialize()
    print(
        f"Extractor: STAM.get_patch (matcher={stam.matcher.spatial_stats()}, "
        f"seasons={stam.season_resolver.names()})"
    )
    return stam.get_patch


def _fit_class_counts(pre: Preprocessor, train_obs: list[AgriculturalObservation],
                     num_classes: int) -> torch.Tensor:
    """Fitted class counts from the train split (label-encoder order)."""
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for obs in train_obs:
        crop, _yield = pre.label.transform(obs)
        counts[int(crop)] += 1.0
    return counts


def _build_first_batch(
    pre: Preprocessor,
    train_obs: list[AgriculturalObservation],
    extractor: Any,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the first real train batch exactly as the training loop would.

    Uses ``split="val"`` so no train augmentation is applied to the trace.
    Image tensors come from the real patch extractor through
    ``preprocessor.transform`` — no random stubs.
    """
    dataset = CropFusionDataset.build(
        pre, train_obs, split="val", extractor=extractor
    )
    loader = build_dataloader(
        dataset,
        pre.config,
        split="val",
        batch_size=batch_size,
        workers=0,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))

    inputs = {
        key: batch[key].to(device, non_blocking=True)
        for key in ("tabular", "ndvi", "evi", "temporal_mask")
    }
    targets = {
        "crop": batch["crop_label"].to(device, non_blocking=True),
        "yield": batch["yield_label"].to(device, non_blocking=True),
    }
    return inputs, targets, profile_batch(batch)


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
    parser.add_argument("--dataset-config",
                        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"))
    parser.add_argument("--stam-config",
                        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"))
    parser.add_argument("--extractor-module", default=None,
                        help="Python module exposing build_extractor() -> callable")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="disable gradient checkpointing (default: enabled)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MULTIMODAL TENSOR TRACE (device={device}) ===")

    # Load observations + authoritative frozen split.
    observations = _load_observations(Path(args.corpus))
    train_obs, val_obs, test_obs = _split_from_provenance(observations)
    print(f"Frozen split (provenance): train={len(train_obs)} "
          f"val={len(val_obs)} test={len(test_obs)}")
    if not train_obs:
        print("\n[FATAL] The frozen train split is empty — refusing to continue.\n")
        return 2

    # Preprocessor fit on the TRAIN split only.
    pre = Preprocessor(load_preprocessing_config(args.config))
    if pre.config.image.normalize == "standard":
        extractor = _resolve_extractor(args)
        pre.fit(train_obs, extractor=extractor)
    else:
        pre.fit(train_obs)

    # Model.
    mc = ModelConfig.from_preprocessor(pre)
    model = ModelFactory.create(mc)
    model.to(device)
    if not args.no_checkpointing:
        apply_gradient_checkpointing(model, True)
        print(f"Gradient checkpointing: enabled (per-timestep encoders)")

    # Loss with fitted class counts (real distribution, not a stub).
    cfg = load_training_config(str(_REPO_ROOT / "training" / "config" / "training.yaml"))
    counts = _fit_class_counts(pre, train_obs, mc.heads.crop.num_classes)
    print(f"Fitted class counts (train): {counts.tolist()}")
    loss_fn = MultiTaskLoss(
        cfg.loss,
        class_weights={
            "crop": build_class_weights(cfg.loss, mc.heads.crop.num_classes, counts)
        },
    )
    loss_fn.to(device)

    # Real batch + mandated shape assertion.
    if mc.uses_image:
        extractor = _resolve_extractor(args)
        inputs, targets, batch_profile = _build_first_batch(
            pre, train_obs, extractor, args.batch_size, device
        )
        # R5.2 mandate: every image tensor is [B, T, C, 224, 224]; assert it.
        assert_image_batch_shape(
            inputs,
            mc.image_encoder.input_size or 224,
            error_type=AssertionError,
            detail="verify_multimodal_tensors first batch",
        )
        print("\n--- First-Batch Multimodal Profile ---")
        print(f"  T (max_observations) = {pre.config.temporal.max_observations}")
        for key in ("tabular", "ndvi", "evi", "temporal_mask", "crop_label", "yield_label"):
            if key in batch_profile:
                print(f"  {key:16s} {batch_profile[key]}")
        print(f"  ndvi_frames: {batch_profile.get('ndvi_frames')}")
        print(f"  evi_frames:  {batch_profile.get('evi_frames')}")
    else:
        # Tabular-only model: no imagery involved.
        tabular = torch.stack([pre.tabular.transform(o).float() for o in train_obs[: args.batch_size]]).to(device)
        labels = [pre.label.transform(o) for o in train_obs[: args.batch_size]]
        crops = torch.stack([c for c, _y in labels]).to(device)
        yields = torch.stack([y for _c, y in labels]).to(device)
        inputs = {"tabular": tabular}
        targets = {"crop": crops, "yield": yields}
        batch_profile = {"tabular": {"shape": list(tabular.shape), "finite": True}}

    # Phase 1: Forward trace (inference mode — no activation graph retained,
    # so the measured memory is the peak of a real inference batch).
    print("\n--- Phase 1: Forward Trace (inference_mode) ---")
    with torch.inference_mode():
        forward_trace, _ = _trace_forward(model, inputs)

    # Phase 2: Gradient trace (autograd graph; FP16 AMP on CUDA mirrors the
    # trainer so per-timestep checkpointing is exercised under the same
    # precision/casting conditions as a real training step).
    use_amp = device.type == "cuda"
    if use_amp:
        print("\n[AMP] FP16 autocast + GradScaler active for the gradient trace "
              "(mirrors trainer config.general.amp)")
    grad_trace = _trace_backward(
        model, inputs, targets, loss_fn, device, use_amp=use_amp
    )

    # GPU memory report (no duplicate model instances were created).
    memory: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.synchronize()
        memory = {
            "device": torch.cuda.get_device_name(0),
            "total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20, 1),
            "peek_reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
            "peek_allocated_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "batch_size": args.batch_size,
        }
        print("\n--- GPU Memory ---")
        print(f"  device:      {memory['device']}")
        print(f"  total:       {memory['total_mb']} MB")
        print(f"  peak resvd:  {memory['peek_reserved_mb']} MB")
        print(f"  peak alloc:  {memory['peek_allocated_mb']} MB")
        print(f"  batch:       {memory['batch_size']}")
        torch.cuda.empty_cache()

    # Summary
    nan_components = [k for k, v in forward_trace.items() if isinstance(v, dict) and not v.get("finite", True)]
    zero_grad_components = [k for k, v in grad_trace["gradient_norms"].items() if v["norm"] == 0]

    print(f"\n=== SUMMARY ===")
    print(f"  Frozen split: train={len(train_obs)} val={len(val_obs)} test={len(test_obs)}")
    print(f"  Components with NaN/Inf: {nan_components or 'NONE'}")
    print(f"  Components with zero gradients: {zero_grad_components or 'NONE'}")
    print(f"  Gradient NaN detected: {grad_trace['any_nan_grads']}")
    print(f"  Total loss: {grad_trace['loss']:.6f}")
    print(f"  Per-task: {grad_trace['per_task']}")

    all_pass = not nan_components and not grad_trace["any_nan_grads"]
    print(f"\n  RESULT: {'PASS' if all_pass else 'FAIL'}")

    report = {
        "split": {"train": len(train_obs), "val": len(val_obs), "test": len(test_obs),
                  "source": "provenance.split (frozen taluk split)"},
        "batch_profile": batch_profile,
        "class_counts": counts.tolist(),
        "sequence_len": str(pre.config.temporal.max_observations),
        "checkpointing": not args.no_checkpointing,
        "amp_enabled_trace": use_amp,
        "gpu_memory": memory,
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