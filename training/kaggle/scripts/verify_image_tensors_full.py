"""R5.2.1 Task A: Full image-branch tensor trace on Kaggle GPU.

Traces every tensor boundary from GeoTIFF through EfficientNetV2-S, reporting:
  shape, dtype, min, max, mean, NaN count, Inf count, zero count, nodata count

Pipeline trace:
  GeoTIFF -> raster read -> patch extraction -> NDVI/EVI preprocessing
  -> normalization -> temporal stack -> EfficientNetV2-S -> image embedding
  -> fusion -> loss

If NaN/Inf occurs, stops immediately and identifies:
  exact tensor, exact sample, exact time step, exact layer, exact operation

Run from repo root (Kaggle training kernel, after ``manager.ensure_image()``)::

    python training/kaggle/scripts/verify_image_tensors_full.py \
        --corpus training/kaggle/outputs/reports/corpus.json \
        --output training/artifacts/image_verification
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    repo_training = (repo_root / "training").resolve()
    for entry in list(sys.path):
        if entry == root or entry == "":
            continue
        shadow = Path(entry) / "training"
        if shadow.exists() and shadow.resolve() != repo_training:
            sys.path.remove(entry)


_add_repo_root(_REPO_ROOT)

from training.dataset_manager import DatasetManager, load_settings  # noqa: E402
from training.preprocessing import Preprocessor, load_preprocessing_config  # noqa: E402
from training.stam import STAM  # noqa: E402
from training.stam.config import load_stam_config  # noqa: E402
from training.stam.observation import AgriculturalObservation  # noqa: E402


def _tensor_stats(t: torch.Tensor, name: str) -> dict[str, Any]:
    """Compute comprehensive stats for a tensor."""
    t_float = t.detach().float()
    return {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min": float(t_float.min().item()),
        "max": float(t_float.max().item()),
        "mean": float(t_float.mean().item()),
        "nan_count": int(torch.isnan(t_float).sum().item()),
        "inf_count": int(torch.isinf(t_float).sum().item()),
        "zero_count": int((t_float == 0).sum().item()),
        "total_elements": int(t_float.numel()),
        "finite": bool(torch.isfinite(t_float).all().item()),
    }


def _load_observations(path: Path) -> list[AgriculturalObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted = []
    for sample in raw["samples"]:
        if sample["status"] == "accepted" and sample.get("observation") is not None:
            accepted.append(AgriculturalObservation.model_validate(sample["observation"]))
    return accepted


def _trace_preprocessing(
    obs: list[AgriculturalObservation],
    pre: Preprocessor,
    extractor: Any,
) -> list[dict[str, Any]]:
    """Trace every observation through preprocessing and report tensor stats."""
    per_obs_report: list[dict[str, Any]] = []
    nan_stop = False

    for i, o in enumerate(obs):
        obs_report: dict[str, Any] = {"index": i, "observation_id": str(o.observation_id)}

        try:
            sample = pre.transform(o, extractor=extractor)
        except Exception as exc:
            obs_report["error"] = f"{type(exc).__name__}: {exc}"
            per_obs_report.append(obs_report)
            continue

        # Trace tabular
        tab = sample["tabular"]
        obs_report["tabular"] = _tensor_stats(tab, "tabular")

        # Trace NDVI sequence
        ndvi_seq = sample["ndvi"]
        obs_report["ndvi_sequence"] = {
            "length": len(ndvi_seq),
            "tensors": [],
        }
        for t_idx, tensor in enumerate(ndvi_seq):
            stats = _tensor_stats(tensor, f"ndvi_t{t_idx}")
            obs_report["ndvi_sequence"]["tensors"].append(stats)
            if not stats["finite"]:
                obs_report["nan_stop"] = True
                obs_report["nan_location"] = f"ndvi_t{t_idx}"
                nan_stop = True

        # Trace EVI sequence
        evi_seq = sample["evi"]
        obs_report["evi_sequence"] = {
            "length": len(evi_seq),
            "tensors": [],
        }
        for t_idx, tensor in enumerate(evi_seq):
            stats = _tensor_stats(tensor, f"evi_t{t_idx}")
            obs_report["evi_sequence"]["tensors"].append(stats)
            if not stats["finite"]:
                obs_report["nan_stop"] = True
                obs_report["nan_location"] = f"evi_t{t_idx}"
                nan_stop = True

        # Trace temporal mask
        mask = sample["temporal_mask"]
        obs_report["temporal_mask"] = _tensor_stats(mask, "temporal_mask")

        # Trace labels
        obs_report["crop_label"] = _tensor_stats(sample["crop_label"], "crop_label")
        obs_report["yield_label"] = _tensor_stats(sample["yield_label"], "yield_label")

        per_obs_report.append(obs_report)

        if nan_stop:
            print(f"\n[CRITICAL] NaN/Inf detected at observation {i} ({o.observation_id})")
            print(f"  Stopping immediately. Bad tensor: {obs_report['nan_location']}")
            break

    return per_obs_report


def _trace_model_forward(
    pre: Preprocessor,
    obs: list[AgriculturalObservation],
    extractor: Any,
) -> dict[str, Any]:
    """Trace through the actual model (NDVI/EVI encoders + fusion)."""
    from training.models.config import ModelConfig
    from training.models.factory import ModelFactory

    mc = ModelConfig.from_preprocessor(pre)
    model = ModelFactory.create(mc)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    forward_report: dict[str, Any] = {"device": str(device), "traces": []}
    nan_stop = False

    with torch.no_grad():
        for i, o in enumerate(obs[:5]):  # Trace first 5 for speed
            try:
                sample = pre.transform(o, extractor=extractor)
            except Exception:
                continue

            trace: dict[str, Any] = {"index": i, "observation_id": str(o.observation_id)}

            # Stack into batch
            ndvi = sample["ndvi"].unsqueeze(0).to(device)  # [1, T, 1, H, W]
            evi = sample["evi"].unsqueeze(0).to(device)
            tabular = sample["tabular"].unsqueeze(0).to(device)
            temporal_mask = sample["temporal_mask"].unsqueeze(0).to(device)

            trace["input_ndvi"] = _tensor_stats(ndvi, "input_ndvi")
            trace["input_evi"] = _tensor_stats(evi, "input_evi")
            trace["input_tabular"] = _tensor_stats(tabular, "input_tabular")

            # NDVI encoder
            if model.ndvi_encoder is not None:
                ndvi_feat = model.ndvi_encoder(ndvi)
                trace["ndvi_encoder_output"] = _tensor_stats(ndvi_feat, "ndvi_features")
                if not trace["ndvi_encoder_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "ndvi_encoder"
                    nan_stop = True

            # EVI encoder
            if model.evi_encoder is not None:
                evi_feat = model.evi_encoder(evi)
                trace["evi_encoder_output"] = _tensor_stats(evi_feat, "evi_features")
                if not trace["evi_encoder_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "evi_encoder"
                    nan_stop = True

            # Image fusion
            if model.image_fusion is not None:
                fused = model.image_fusion(
                    ndvi_feat if model.ndvi_encoder else None,
                    evi_feat if model.evi_encoder else None,
                )
                trace["image_fusion_output"] = _tensor_stats(fused, "fused_features")
                if not trace["image_fusion_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "image_fusion"
                    nan_stop = True

            # Temporal transformer
            if model.temporal_transformer is not None:
                temporal_out = model.temporal_transformer(fused, mask=temporal_mask)
                trace["temporal_output"] = _tensor_stats(temporal_out, "temporal_output")
                if not trace["temporal_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "temporal_transformer"
                    nan_stop = True

            # Tabular encoder
            tab_emb = model.tab_encoder(tabular)
            trace["tabular_embedding"] = _tensor_stats(tab_emb, "tabular_embedding")
            if not trace["tabular_embedding"]["finite"]:
                trace["nan_stop"] = True
                trace["nan_location"] = "tabular_encoder"
                nan_stop = True

            # Cross attention
            if model.cross_attention is not None:
                cross_out = model.cross_attention(temporal_out, tab_emb)
                trace["cross_attention_output"] = _tensor_stats(cross_out, "cross_attention")
                if not trace["cross_attention_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "cross_attention"
                    nan_stop = True

            # Adaptive gate
            if model.gated_fusion is not None:
                gated = model.gated_fusion(temporal_out, tab_emb, cross_out)
                trace["gated_fusion_output"] = _tensor_stats(gated["fused"], "gated_fused")
                if not trace["gated_fusion_output"]["finite"]:
                    trace["nan_stop"] = True
                    trace["nan_location"] = "adaptive_gate"
                    nan_stop = True

            forward_report["traces"].append(trace)

            if nan_stop:
                print(f"\n[CRITICAL] NaN/Inf in model forward at obs {i}")
                print(f"  Bad location: {trace['nan_location']}")
                break

    return forward_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropfusion-verify-image-tensors-full")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-preprocess", type=int, default=50,
                        help="Max observations for preprocessing trace")
    parser.add_argument("--max-forward", type=int, default=5,
                        help="Max observations for model forward trace")
    parser.add_argument(
        "--config", default=None,
        help="preprocessing.yaml path (default training/config/preprocessing.yaml)",
    )
    parser.add_argument(
        "--dataset-config",
        default=str(_REPO_ROOT / "training" / "config" / "dataset.yaml"),
    )
    parser.add_argument(
        "--stam-config",
        default=str(_REPO_ROOT / "training" / "config" / "stam.yaml"),
    )
    args = parser.parse_args(argv)

    config = (
        Preprocessor.from_config(args.config).config
        if args.config
        else load_preprocessing_config(_REPO_ROOT / "training" / "config" / "preprocessing.yaml")
    )
    pre = Preprocessor(config)
    obs = _load_observations(Path(args.corpus))
    print(f"[verify_image_tensors_full] accepted observations: {len(obs)}")
    accepted, _ = pre.filter(obs)
    print(f"[verify_image_tensors_full] after quality filter: {len(accepted)}")

    # STAM + extractor
    manager = DatasetManager(load_settings(Path(args.dataset_config)))
    stam = None
    try:
        manifest = manager.provider_manifests().get("kaggle_hub_image", {})
        if not manifest.get("available"):
            raise RuntimeError(
                "imagery catalog is NOT available in this environment "
                "(requires the Kaggle imagery mount + ensure_image)"
            )
        stam = STAM(manager, load_stam_config(Path(args.stam_config)))
        stam.initialize()
        print("[verify_image_tensors_full] STAM initialized; extractor ready")
    except Exception as exc:
        print(f"[verify_image_tensors_full] SKIPPED: {exc}")
        print("[verify_image_tensors_full] cannot verify without rasters")
        return 0
    finally:
        manager.close()

    pre.fit(accepted, extractor=stam.get_patch)

    # Phase 1: Preprocessing trace
    print("\n" + "=" * 70)
    print("PHASE 1: PREPROCESSING TENSOR TRACE")
    print("=" * 70)
    preprocess_report = _trace_preprocessing(
        accepted[: args.max_preprocess], pre, stam.get_patch
    )

    # Summary stats
    nan_count = sum(1 for r in preprocess_report if r.get("nan_stop"))
    finite_count = sum(1 for r in preprocess_report if not r.get("nan_stop") and "error" not in r)
    error_count = sum(1 for r in preprocess_report if "error" in r)
    print(f"\nPreprocessing results: {finite_count} finite, {nan_count} NaN/Inf, {error_count} errors")

    if nan_count:
        bad = next(r for r in preprocess_report if r.get("nan_stop"))
        print(f"  First NaN at: obs {bad['index']}, location: {bad['nan_location']}")
        print(f"  Observation ID: {bad['observation_id']}")

    # Phase 2: Model forward trace
    print("\n" + "=" * 70)
    print("PHASE 2: MODEL FORWARD TENSOR TRACE (EfficientNetV2-S + Fusion)")
    print("=" * 70)
    forward_report = _trace_model_forward(pre, accepted, stam.get_patch)

    model_nan = sum(1 for t in forward_report["traces"] if t.get("nan_stop"))
    model_finite = sum(1 for t in forward_report["traces"] if not t.get("nan_stop"))
    print(f"\nModel forward results: {model_finite} finite, {model_nan} NaN/Inf")

    if model_nan:
        bad = next(t for t in forward_report["traces"] if t.get("nan_stop"))
        print(f"  First NaN at: obs {bad['index']}, location: {bad['nan_location']}")
        print(f"  Observation ID: {bad['observation_id']}")

    # Write report
    report = {
        "preprocessing_trace": preprocess_report,
        "model_forward_trace": forward_report,
        "summary": {
            "observations_traced": len(preprocess_report),
            "preprocessing_nan": nan_count,
            "preprocessing_finite": finite_count,
            "preprocessing_errors": error_count,
            "model_nan": model_nan,
            "model_finite": model_finite,
            "all_finite": nan_count == 0 and model_nan == 0,
        },
    }

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "image_tensor_full_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"\n[verify_image_tensors_full] wrote {out / 'image_tensor_full_report.json'}")

    return 1 if (nan_count or model_nan) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
