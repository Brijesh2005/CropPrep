"""R5.2.1 Task B: AMP stability test — controlled precision experiment.

Tests 4 precision modes on the CropFusion model:
  1. FP32 (baseline)
  2. FP16 AMP (current config)
  3. AMP disabled (FP32 with no autocast)
  4. BF16 AMP (only if hardware supports it)

For each mode:
  - Runs forward + backward on a real data sample
  - Reports loss, gradient norms, NaN/Inf status
  - Determines hardware capability
  - Recommends the stable precision mode

P100 (compute capability 6.0): FP32 + FP16 only (no BF16)
V100 (compute capability 7.0): FP32 + FP16 only (no BF16)
A100 (compute capability 8.0): FP32 + FP16 + BF16
T4  (compute capability 7.5): FP32 + FP16 only (no BF16)

Run from repo root::

    python training/kaggle/scripts/amp_stability_test.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/amp_stability
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from training.models.config import ModelConfig  # noqa: E402
from training.models.factory import ModelFactory  # noqa: E402
from training.preprocessing import (  # noqa: E402
    Preprocessor,
    load_preprocessing_config,
    split_observations,
)
from training.stam.observation import AgriculturalObservation  # noqa: E402
from training.training.config import load_training_config  # noqa: E402
from training.training.losses import MultiTaskLoss, build_class_weights  # noqa: E402


def _check_hardware() -> dict[str, Any]:
    """Determine GPU capabilities."""
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["bf16_supported"] = cap[0] >= 8  # Ampere+
        info["fp16_supported"] = cap[0] >= 6  # Pascal+
        info["total_memory_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2))
    else:
        info["bf16_supported"] = False
        info["fp16_supported"] = False
    return info


def _make_batch(
    pre: Preprocessor, obs: list[Any], model: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a single batch from observations."""
    tabular = torch.stack([pre.tabular.transform(o).float() for o in obs])
    labels = [pre.label.transform(o) for o in obs]
    yields = torch.stack([y for _c, y in labels])
    crops = torch.stack([c for c, _y in labels])

    inputs: dict[str, Any] = {"tabular": tabular}

    if model.use_image:
        seq_len = model.config.temporal.max_len
        image_size = model.config.image_encoder.input_size or 224
        ndvi = torch.randn(len(obs), seq_len, 1, image_size, image_size) * 0.1
        evi = torch.randn(len(obs), seq_len, 1, image_size, image_size) * 0.1
        mask = torch.ones(len(obs), seq_len)
        if seq_len > 1:
            mask[:, -1] = 0.0
        inputs["ndvi"] = ndvi
        inputs["evi"] = evi
        inputs["temporal_mask"] = mask

    targets = {"crop": crops, "yield": yields}
    return inputs, targets


def _run_precision_experiment(
    mode: str,
    model: Any,
    inputs: dict[str, Any],
    targets: dict[str, Any],
    loss_fn: Any,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Run one precision experiment and report results."""
    result: dict[str, Any] = {"mode": mode}

    model.train()
    inputs_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()}
    targets_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in targets.items()}

    autocast_ctx = (
        torch.autocast("cuda", dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else contextlib.nullcontext()
    )

    try:
        with autocast_ctx:
            out = model(inputs_d)
            out_dict = {}
            if out.crop_logits is not None:
                out_dict["crop"] = out.crop_logits
            if out.yield_pred is not None:
                out_dict["yield"] = out.yield_pred
            total, per_task = loss_fn(out_dict, targets_d)

        result["loss"] = float(total.item())
        result["per_task"] = {k: float(v.item()) for k, v in per_task.items()}
        result["loss_finite"] = bool(torch.isfinite(total).item())

        total.backward()

        # Check gradient norms
        grad_norms = {}
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = float(param.grad.data.norm(2).item())
                grad_norms[name] = norm
                total_norm += norm ** 2
        total_norm = total_norm ** 0.5

        result["total_grad_norm"] = round(total_norm, 6)
        result["any_grad_nan"] = any(
            torch.isnan(p.grad).any().item()
            for p in model.parameters()
            if p.grad is not None
        )
        result["any_grad_inf"] = any(
            torch.isinf(p.grad).any().item()
            for p in model.parameters()
            if p.grad is not None
        )

        # Check output stats
        result["yield_pred_stats"] = {
            "min": float(out.yield_pred.min().item()),
            "max": float(out.yield_pred.max().item()),
            "mean": float(out.yield_pred.mean().item()),
            "nan": int(torch.isnan(out.yield_pred).sum().item()),
            "inf": int(torch.isinf(out.yield_pred).sum().item()),
        }
        if out.crop_logits is not None:
            result["crop_logits_stats"] = {
                "min": float(out.crop_logits.min().item()),
                "max": float(out.crop_logits.max().item()),
                "mean": float(out.crop_logits.mean().item()),
                "nan": int(torch.isnan(out.crop_logits).sum().item()),
                "inf": int(torch.isinf(out.crop_logits).sum().item()),
            }

        result["passed"] = (
            result["loss_finite"]
            and not result["any_grad_nan"]
            and not result["any_grad_inf"]
        )

    except Exception as exc:
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    model.zero_grad()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-amp-stability-test")
    parser.add_argument("--corpus", default=None, help="Path to corpus JSON (optional; skip if not provided)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--config", default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"))
    args = parser.parse_args(argv)

    hardware = _check_hardware()
    print("=== AMP STABILITY TEST ===")
    print(f"GPU: {hardware.get('device_name', 'N/A')}")
    print(f"Compute capability: {hardware.get('compute_capability', 'N/A')}")
    print(f"FP16 supported: {hardware['fp16_supported']}")
    print(f"BF16 supported: {hardware['bf16_supported']}")
    print(f"CUDA available: {hardware['cuda_available']}")

    device = torch.device("cuda" if hardware["cuda_available"] else "cpu")

    if not args.corpus:
        print("\n=== SKIPPED: --corpus not provided (run run_pipeline.py first) ===")
        return 0

    # Load corpus
    raw = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    obs = [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]
    print(f"\nAccepted observations: {len(obs)}")

    # Preprocessor
    pre = Preprocessor(load_preprocessing_config(args.config))
    train_obs, _, _ = split_observations(obs, pre.config.split)
    accepted, _ = pre.filter(train_obs)
    pre.fit(accepted)

    # Model (tabular-only for local testing; image branch needs rasters)
    mc = ModelConfig.from_preprocessor(pre)
    mc.image_encoder.backbone = None  # tabular-only (no raster locally)
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
    inputs, targets = _make_batch(pre, accepted[: args.batch_size], model)

    # Run experiments
    experiments: list[dict[str, Any]] = []

    # 1. FP32 baseline
    print("\n--- Experiment 1: FP32 ---")
    model_fp32 = ModelFactory.create(mc).to(device)
    model_fp32.load_state_dict(model.state_dict())
    r1 = _run_precision_experiment("fp32", model_fp32, inputs, targets, loss_fn, device)
    experiments.append(r1)
    print(f"  Loss: {r1.get('loss', 'N/A'):.6f} | Finite: {r1.get('passed', False)}")
    del model_fp32
    torch.cuda.empty_cache()

    # 2. FP16 AMP (current config)
    print("\n--- Experiment 2: FP16 AMP ---")
    model_fp16 = ModelFactory.create(mc).to(device)
    model_fp16.load_state_dict(model.state_dict())
    r2 = _run_precision_experiment(
        "fp16_amp", model_fp16, inputs, targets, loss_fn, device,
        amp_dtype=torch.float16,
    )
    experiments.append(r2)
    print(f"  Loss: {r2.get('loss', 'N/A'):.6f} | Finite: {r2.get('passed', False)}")
    del model_fp16
    torch.cuda.empty_cache()

    # 3. AMP disabled (explicit FP32)
    print("\n--- Experiment 3: AMP disabled (FP32) ---")
    model_noamp = ModelFactory.create(mc).to(device)
    model_noamp.load_state_dict(model.state_dict())
    r3 = _run_precision_experiment(
        "no_amp", model_noamp, inputs, targets, loss_fn, device, amp_dtype=None,
    )
    experiments.append(r3)
    print(f"  Loss: {r3.get('loss', 'N/A'):.6f} | Finite: {r3.get('passed', False)}")
    del model_noamp
    torch.cuda.empty_cache()

    # 4. BF16 AMP (if supported)
    if hardware["bf16_supported"]:
        print("\n--- Experiment 4: BF16 AMP ---")
        model_bf16 = ModelFactory.create(mc).to(device)
        model_bf16.load_state_dict(model.state_dict())
        r4 = _run_precision_experiment(
            "bf16_amp", model_bf16, inputs, targets, loss_fn, device,
            amp_dtype=torch.bfloat16,
        )
        experiments.append(r4)
        print(f"  Loss: {r4.get('loss', 'N/A'):.6f} | Finite: {r4.get('passed', False)}")
        del model_bf16
        torch.cuda.empty_cache()
    else:
        print("\n--- Experiment 4: BF16 AMP --- SKIPPED (hardware does not support BF16)")
        experiments.append({
            "mode": "bf16_amp",
            "passed": False,
            "skipped": True,
            "reason": f"GPU {hardware.get('device_name', 'N/A')} (compute {hardware.get('compute_capability', 'N/A')}) does not support BF16",
        })

    # Recommendation
    stable_modes = [e["mode"] for e in experiments if e.get("passed")]
    print(f"\n=== STABLE MODES: {stable_modes or 'NONE'} ===")

    recommendation = "fp32"  # safest fallback
    if "fp16_amp" in stable_modes:
        recommendation = "fp16_amp"
    if "bf16_amp" in stable_modes:
        recommendation = "bf16_amp"  # preferred when available

    print(f"RECOMMENDATION: {recommendation}")
    print(f"  Current training.yaml amp_dtype: float16")
    print(f"  Current performance.yaml dtype: bf16")
    if not hardware["bf16_supported"] and recommendation == "fp16_amp":
        print(f"  NOTE: performance.yaml specifies bf16 but GPU does not support it.")
        print(f"         training.yaml correctly uses float16.")

    report = {
        "hardware": hardware,
        "experiments": experiments,
        "recommendation": recommendation,
        "stable_modes": stable_modes,
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "amp_stability_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[amp_stability_test] wrote {out / 'amp_stability_report.json'}")

    return 0 if stable_modes else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
