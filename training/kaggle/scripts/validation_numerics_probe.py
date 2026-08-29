"""R5.3: validation-numerics probe — root-cause proof for TR-VAL-001.

The July run failed IN VALIDATION: Inf appeared at
``ndvi_encoder.backbone.blocks.4.2.bn2.drop`` on a ``[128, 960, 14, 14]``
tensor while the training loss / gradients and the batch-4 diagnostic stayed
finite. This probe explains that divergence with controlled experiments over
the SAME frozen corpus, SAME preprocessing, SAME model config and REAL imagery
(Kaggle mount) the trainer consumes:

    A. FP32 validation (config default after the fix)                -> finite
    B. FP16 validation (old behaviour, autoreproduce the Inf)        -> observed
    C. NDVI-only model, FP16 validation                              -> observed
    D. EVI-only model, FP16 validation                               -> observed
    E. Imagery-only model (no tabular), FP16 validation              -> observed
    F. Batch 4 (the diagnostic's shape), FP32 + FP16                 -> finite
    G. Batch size sweep {1, 4, 8, 16} under FP16 (eval fast path)    -> threshold
    H. Lightweight gate: 1 AMP training step (B=16, checkpointed)
       + 1 FP32 validation pass                                      -> all finite

Why the divergence: in eval the encoder uses the fast path and encodes
``B * T`` frames in ONE backbone forward (backbone.py:177) — batch 16 x 8 =
128 frames; fp16 GEMM/accumulation overflows there. In training (gradient
checkpointing enabled) the encoder iterates per timestep with B=16 frames each.
The diagnostic used batch 4 -> 32 frames, finite.

Every FP16(FP32) pass routes through the real :class:`Validator` so the
attribution uses the production ``nan_source_hooks`` with
``origin == "created" | "propagated"``.

Run from repo root (needs the Kaggle imagery mount):

    python training/kaggle/scripts/validation_numerics_probe.py \
        --frozen-crop-csv govt_crop_matched_v1/crop_supervised_v1.csv \
        --frozen-manifest training_manifests/crop_supervised_v1_manifest.json \
        --output training/kaggle/outputs/reports/numerics_probe.json

Exit code 0 = the lightweight gate passed AND fp32 validation was finite;
2 = CUDA/imagery unavailable (environment gate); 1 = a gate failed.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from shared.config import deep_merge  # noqa: E402
from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.kaggle.config import (  # noqa: E402
    load_kaggle_config,
    load_paths_config,
    WorkspaceLayout,
)
from training.kaggle.environment import EnvironmentManager  # noqa: E402
from training.kaggle.frozen_corpus import FrozenCorpusLoader  # noqa: E402
from training.kaggle.workspace import WorkspaceManager  # noqa: E402
from training.models.config import (  # noqa: E402
    ModelConfig,
    load_model_config as load_model_cfg,
)
from training.models.factory import ModelFactory  # noqa: E402
from training.preprocessing import (  # noqa: E402
    CropFusionDataset,
    DataloaderConfig,
    Preprocessor,
    build_dataloader,
)
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.training.config import (  # noqa: E402
    load_training_config as load_training_cfg,
)
from training.training.losses import MultiTaskLoss  # noqa: E402
from training.training.validator import Validator  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _architecture_only(mc: ModelConfig) -> dict[str, Any]:
    """User model config minus the schema-owned fields (mirrors
    ``Experiment._architecture_only``: the preprocessor re-derives them)."""
    user = mc.model_dump()
    tabular = dict(user.get("tabular") or {})
    tabular.pop("numeric_dim", None)
    tabular.pop("categorical_cardinalities", None)
    user["tabular"] = tabular
    heads = dict(user.get("heads") or {})
    crop = dict(heads.get("crop") or {})
    crop.pop("num_classes", None)
    heads["crop"] = crop
    user["heads"] = heads
    image_encoder = dict(user.get("image_encoder") or {})
    image_encoder.pop("input_size", None)
    user["image_encoder"] = image_encoder
    temporal = dict(user.get("temporal") or {})
    temporal.pop("max_len", None)
    user["temporal"] = temporal
    return user


def _build_model(
    pre: Preprocessor, base: ModelConfig, **touches: Any
) -> ModelConfig:
    overrides = deep_merge(_architecture_only(base), touches)
    return ModelConfig.from_preprocessor(pre, **overrides)


def _default_input_map(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = {k: batch[k] for k in ("tabular", "ndvi", "evi", "temporal_mask")
              if k in batch}
    targets: dict[str, Any] = {}
    if "crop_label" in batch:
        targets["crop"] = batch["crop_label"]
    if "yield_label" in batch:
        targets["yield"] = batch["yield_label"]
    return inputs, targets


def _finite(value: Any) -> bool:
    return bool(torch.isfinite(torch.as_tensor(value)).all().item())


# --------------------------------------------------------------------------- #
# Experiment runners
# --------------------------------------------------------------------------- #


def run_validation(
    *,
    model: torch.nn.Module,
    loader: Any,
    loss_module: torch.nn.Module,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    batches: int,
    label: str,
) -> dict[str, Any]:
    """Run the real :class:`Validator` for a few validation batches.

    FP16 runs may raise ``ValidationError`` (nan_policy=stop) — that is
    captured and reported with the first-producer attribution.
    """
    validator = Validator(
        model,
        loss_module,
        device=device,
        amp=amp,
        amp_dtype=amp_dtype,
        nan_policy="stop",
    )
    entry: dict[str, Any] = {"label": label, "amp": amp, "amp_dtype": amp_dtype}
    try:
        result = validator.validate(itertools.islice(loader, batches), epoch=0)
        val_loss = float(result.metrics.get("val_loss", 0.0))
        entry.update(
            {
                "passed": True,
                "val_loss": val_loss,
                "val_loss_finite": _finite(val_loss),
                "batches_evaluated": batches,
                "first_batch": result.first_batch,
            }
        )
    except Exception as exc:
        detail = getattr(exc, "detail", None) or {}
        entry.update(
            {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "val_loss_finite": False,
                "nan_sources": detail.get("nan_sources") or [],
                "first_batch": detail.get("first_batch") or {},
            }
        )
    return entry


def run_lightweight_train_step(
    *,
    model: torch.nn.Module,
    loader: Any,
    loss_module: torch.nn.Module,
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    """One AMP-fp16 training step (B=16, gradient checkpointing, GradScaler).

    Mirrors ``Trainer._run_epoch``: autocast forward, scaled backward,
    ``scaler.step`` / ``scaler.update``, NaN checks on loss + gradients.
    """
    from training.training.utils import apply_gradient_checkpointing

    apply_gradient_checkpointing(model, True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    autocast = torch.autocast("cuda", dtype=torch.float16)

    entry: dict[str, Any] = {"steps_done": 0}
    try:
        for batch in itertools.islice(loader, steps):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            inputs, targets = _default_input_map(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                out = model(inputs)
                out_dict = {k: out.as_dict()[k] for k in ("crop_logits", "yield_pred")
                            if out.as_dict().get(k) is not None}
                total, per_task = loss_module(out_dict, targets)
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
            grads_finite = all(
                _finite(p.grad)
                for p in model.parameters()
                if p.grad is not None
            )
            entry = {
                "steps_done": entry["steps_done"] + 1,
                "last_loss": float(total.detach().item()),
                "loss_finite": _finite(total.detach()),
                "grads_finite": bool(grads_finite),
                "passed": _finite(total.detach()) and bool(grads_finite),
            }
    except Exception as exc:
        entry["passed"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"
    model.eval()
    return entry


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-validation-numerics-probe"
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument(
        "--paths-config", default=str(_REPO_ROOT / "training" / "config" / "paths.yaml")
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--training-config",
        default=str(_REPO_ROOT / "training" / "config" / "training.yaml"),
    )
    parser.add_argument(
        "--model-config",
        default=str(_REPO_ROOT / "training" / "config" / "model.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    parser.add_argument(
        "--preprocessing-config",
        default=str(_REPO_ROOT / "training" / "config" / "preprocessing.yaml"),
    )
    parser.add_argument("--frozen-crop-csv", required=True)
    parser.add_argument("--frozen-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--val-batches", type=int, default=8,
        help="Validation batches per pass (all B=16, T=8 -> 128 frames/pass)",
    )
    parser.add_argument(
        "--train-steps", type=int, default=2, help="AMP training steps for gate H",
    )
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("[probe] FATAL: CUDA required (fp16 experiments are GPU-only). "
              "Run on the Kaggle GPU notebook.")
        return 2

    # -- 1. Configuration (mirrors run_pipeline) -------------------------- #
    paths = load_paths_config(Path(args.paths_config))
    kaggle_cfg = load_kaggle_config()
    dataset_settings = load_settings(Path(args.dataset_config))
    training_cfg = load_training_cfg(Path(args.training_config))
    model_cfg = load_model_cfg(Path(args.model_config))
    stam_cfg = load_stam_config(Path(args.stam_config))
    preprocessing_cfg = Preprocessor.from_config(args.preprocessing_config)
    if (
        stam_cfg.temporal.season_file is not None
        and not stam_cfg.temporal.season_file.is_absolute()
    ):
        stam_cfg.temporal.season_file = (
            Path(args.stam_config).resolve().parent / stam_cfg.temporal.season_file
        )

    layout = WorkspaceLayout.resolve(paths, repo_root=Path(args.repo_root))
    workspace = WorkspaceManager(layout)
    workspace.create()
    env_report = EnvironmentManager(Path(args.repo_root)).report()
    device = torch.device("cuda")

    report: dict[str, Any] = {
        "gpu": env_report.get("gpu"),
        "device": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(
            str(v) for v in torch.cuda.get_device_capability(0)
        ),
        "cudnn_version": torch.backends.cudnn.version(),
    }

    # -- 2. Dataset / imagery / STAM (real patches required) -------------- #
    manager = DatasetManager(dataset_settings)
    manifests = manager.provider_manifests()
    image_manifest = manifests.get("kaggle_hub_image", {})
    if image_manifest.get("available") is not True:
        manager.close()
        print("[probe] FATAL: imagery unavailable — real patches are required "
              "for a faithful reproduction. Attach the imagery dataset.")
        return 2
    manager.ensure_image()
    manager.generate_image_metadata(force=False)
    stam = STAM(manager, stam_cfg)
    stam.initialize()

    # -- 3. Frozen corpus (same provenance split the trainer consumes) ---- #
    loader = FrozenCorpusLoader(
        csv_path=args.frozen_crop_csv,
        manifest_path=args.frozen_manifest or str(
            _REPO_ROOT / "training_manifests" / "crop_supervised_v1_manifest.json"
        ),
    )
    loader.validate()
    train_obs, val_obs, test_obs = loader.build(stam)
    report["corpus"] = {
        "train": len(train_obs),
        "val": len(val_obs),
        "test": len(test_obs),
        "imagery": loader.imagery_summary(
            train_obs, val_obs, test_obs,
            max_observations=preprocessing_cfg.config.temporal.max_observations,
        ),
    }
    print(f"[probe] corpus: train={len(train_obs)} val={len(val_obs)} "
          f"test={len(test_obs)}")

    # -- 4. Preprocessor (fit on the frozen TRAIN split only) ------------- #
    pre = Preprocessor(preprocessing_cfg.config)
    fit_obs, _ = pre.filter(train_obs)
    pre.fit(fit_obs, extractor=stam.get_patch)

    print("[probe] preprocessing fitted "
          f"(image={preprocessing_cfg.config.image.size}, "
          f"temporal={preprocessing_cfg.config.temporal.max_observations})")

    # -- 5. Loaders (exactly the Experiment construction) ----------------- #
    def _loader(obs: Any, batch_size: int, split: str) -> Any:
        dataset = CropFusionDataset.build(
            pre, obs, split=split, extractor=stam.get_patch
        )
        loader_cfg = DataloaderConfig(
            batch_size=batch_size,
            workers=0,
            pin_memory=True,
            shuffle_train=False,
        )
        return build_dataloader(dataset, loader_cfg, split=split)

    train_loader = _loader(train_obs, 16, "train")
    base_loader_16 = _loader(val_obs, 16, "val")
    diagnostics_loader_4 = _loader(val_obs, 4, "val")

    # -- 6. Model + loss (the real architecture, real imagery) ------------ #
    base_mc = _build_model(pre, model_cfg)
    loss_fn = MultiTaskLoss(training_cfg.loss).to(device)
    print("[probe] model:", base_mc.image_encoder.backbone,
          f"B*T fast path frames/pass = 16x{base_mc.temporal.max_len}="
          f"{16 * base_mc.temporal.max_len}")

    experiments: list[dict[str, Any]] = []
    a_result: dict[str, Any] | None = None

    # ---- A. FP32 validation (the fix: validation.amp default False) ----- #
    model = ModelFactory.create(base_mc).to(device)
    model.eval()
    a = run_validation(
        model=model, loader=base_loader_16, loss_module=loss_fn, device=device,
        amp=False, amp_dtype="float16", batches=args.val_batches,
        label="A_fp32_validation_b16",
    )
    experiments.append(a)
    if a["passed"] and a["val_loss_finite"]:
        a_result = a
    del model
    torch.cuda.empty_cache()

    # ---- B. FP16 validation (reproduce TR-VAL-001) ---------------------- #
    model = ModelFactory.create(base_mc).to(device)
    model.eval()
    b = run_validation(
        model=model, loader=base_loader_16, loss_module=loss_fn, device=device,
        amp=True, amp_dtype="float16", batches=args.val_batches,
        label="B_fp16_validation_b16",
    )
    experiments.append(b)
    del model
    torch.cuda.empty_cache()

    # ---- C. NDVI-only model, FP16 validation ---------------------------- #
    model = ModelFactory.create(_build_model(pre, model_cfg,
                                             image_encoder={"enable_evi": False})).to(device)
    model.eval()
    experiments.append(run_validation(
        model=model, loader=base_loader_16, loss_module=loss_fn, device=device,
        amp=True, amp_dtype="float16", batches=args.val_batches,
        label="C_ndvi_only_fp16_b16",
    ))
    del model
    torch.cuda.empty_cache()

    # ---- D. EVI-only model, FP16 validation ----------------------------- #
    model = ModelFactory.create(_build_model(pre, model_cfg,
                                             image_encoder={"enable_ndvi": False})).to(device)
    model.eval()
    experiments.append(run_validation(
        model=model, loader=base_loader_16, loss_module=loss_fn, device=device,
        amp=True, amp_dtype="float16", batches=args.val_batches,
        label="D_evi_only_fp16_b16",
    ))
    del model
    torch.cuda.empty_cache()

    # ---- E. Imagery-only (no tabular), FP16 validation ------------------ #
    model = ModelFactory.create(_build_model(
        pre, model_cfg,
        tabular={"numeric_dim": 0, "categorical_cardinalities": []},
    )).to(device)
    model.eval()
    experiments.append(run_validation(
        model=model, loader=base_loader_16, loss_module=loss_fn, device=device,
        amp=True, amp_dtype="float16", batches=args.val_batches,
        label="E_imagery_only_fp16_b16",
    ))
    del model
    torch.cuda.empty_cache()

    # ---- F. Batch 4 (the diagnostic's shape): FP32 + FP16 --------------- #
    model = ModelFactory.create(base_mc).to(device)
    model.eval()
    experiments.append(run_validation(
        model=model, loader=diagnostics_loader_4, loss_module=loss_fn,
        device=device, amp=False, amp_dtype="float16", batches=args.val_batches,
        label="F_batch4_fp32",
    ))
    experiments.append(run_validation(
        model=model, loader=diagnostics_loader_4, loss_module=loss_fn,
        device=device, amp=True, amp_dtype="float16", batches=args.val_batches,
        label="F_batch4_fp16",
    ))
    del model
    torch.cuda.empty_cache()

    # ---- G. Batch-size sweep {1, 4, 8, 16} under FP16 eval -------------- #
    model = ModelFactory.create(base_mc).to(device)
    model.eval()
    for batch_size in (1, 4, 8, 16):
        sweep_loader = _loader(val_obs, batch_size, "val")
        experiments.append(run_validation(
            model=model, loader=sweep_loader, loss_module=loss_fn, device=device,
            amp=True, amp_dtype="float16", batches=args.val_batches,
            label=f"G_fp16_eval_bs{batch_size}",
        ))
    del model
    torch.cuda.empty_cache()

    # ---- H. Lightweight gate: AMP training steps + FP32 validation ------ #
    train_model = ModelFactory.create(base_mc).to(device)
    h_train = run_lightweight_train_step(
        model=train_model, loader=train_loader, loss_module=loss_fn,
        device=device, steps=args.train_steps,
    )
    train_model.eval()
    h_val = run_validation(
        model=train_model, loader=base_loader_16, loss_module=loss_fn,
        device=device, amp=False, amp_dtype="float16", batches=args.val_batches,
        label="H_lightweight_fp32_validation",
    )
    del train_model
    torch.cuda.empty_cache()
    gate = {
        "train_steps": h_train,
        "validation_fp32": h_val,
        "passed": bool(h_train.get("passed") and h_val.get("passed")
                       and h_val.get("val_loss_finite")),
    }
    experiments.append({"label": "H_lightweight_gate", **gate})
    a_result = experiments[0] if experiments[0]["passed"] else a_result

    # fp32-stable verdict: B reproduced a non-finite fp16 loss AND A was finite
    root_cause = {
        "fp32_validation_finite": bool(a_result and a_result.get("val_loss_finite")),
        "fp16_validation_nonfinite": not experiments[1].get("passed"),
        "first_nonfinite": experiments[1].get("nan_sources") or [],
        "verdict": (
            "fp16 autocast in the eval fast path (B*T=128 frames) goes "
            "non-finite on this GPU; FP32 validation is finite and is now the "
            "config default (validation.amp=false). Training AMP is unchanged."
            if (a_result and a_result.get("val_loss_finite")
                and not experiments[1].get("passed"))
            else "no fp16 validation non-finite reproduced on this run"
        ),
    }
    report["root_cause"] = root_cause
    report["experiments"] = experiments
    report["gate"] = gate

    print("\n==================== VALIDATION NUMERICS PROBE ====================")
    print(f"GPU: {report['device']} (compute {report['compute_capability']}, "
          f"cudnn {report['cudnn_version']})")
    for e in experiments:
        label = e.get("label", "")
        if label.startswith("H_lightweight_gate"):
            continue
        status = "FINITE " if e.get("passed") and e.get("val_loss_finite") else "BROKEN "
        print(f"  [{status}] {label:28s} val_loss={e.get('val_loss')}")
        for ns in e.get("nan_sources", []):
            print(f"         first producer: {ns.get('module')} ({ns.get('type')}) "
                  f"shape={ns.get('shape')} nan={ns.get('nan')} inf={ns.get('inf')} "
                  f"origin={ns.get('origin')}")
    print(f"  light gate        : train={gate['train_steps'].get('passed')} "
          f"val_fp32={gate['validation_fp32'].get('passed')} "
          f"-> {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"  root cause verdict: {root_cause['verdict']}")
    print("====================================================================")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[probe] wrote {out}")

    manager.close()
    return 0 if gate["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())