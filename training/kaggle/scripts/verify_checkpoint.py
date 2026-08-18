"""R5.2 Task 10: checkpoint provenance + integrity audit.

Verifies the dk-bridge checkpoint (latest.pt, 684MB) and cross-checks the
v2.0.0 release model's provenance:
  * loads with the repo's CheckpointManager (weights_only=True)
  * metadata: epoch / step / metrics / model config snapshot
  * every weight tensor in model_state_dict is finite
  * metric provenance: the release package metrics vs the two Kaggle runs
    (dk-bridge = crop/support 0, R2 -78k; train-v6 = crop/support 11,
    R2 0.9748). NOTE: the shipped release model is the TRAIN-V6 model, NOT
    the dk-bridge model the release notes claim.
  * release package model checksum vs manifest
  * released TorchScript model forward-pass finiteness

Run from repo root::

    python training/kaggle/scripts/verify_checkpoint.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    repo_root = repo_root.resolve()
    root = str(repo_root)
    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)


_add_repo_root(_REPO_ROOT)

from training.models.checkpoint import CheckpointManager  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ckpt = Path(
        r"D:\CropPrep\kaggle_runs\train-dk-bridge\checkpoint\CropPrep"
        r"\training\kaggle\outputs\cropfusion_training\checkpoints\latest.pt"
    )
    print(f"checkpoint: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")

    state = CheckpointManager.load(ckpt)
    print("\n=== checkpoint keys ===")
    for key, value in state.items():
        if isinstance(value, dict):
            print(f"  {key}: dict[{len(value)}]")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")

    meta = state.get("metadata") or {}
    print("\n=== metadata ===")
    for k, v in meta.items():
        if isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, default=str)[:300]}")
        else:
            print(f"  {k}: {v}")

    metrics = state.get("metrics") or {}
    print("\n=== metrics ===")
    print(json.dumps(metrics, indent=2, default=str)[:2000])

    model_state = state.get("model_state_dict") or {}
    print(f"\n=== model_state_dict: {len(model_state)} tensors ===")
    non_finite = []
    n_params = 0
    total_bytes = 0
    for name, tensor in model_state.items():
        if not torch.is_tensor(tensor):
            continue
        n_params += tensor.numel()
        total_bytes += tensor.numel() * tensor.element_size()
        if not torch.isfinite(tensor).all():
            non_finite.append((name, int(torch.isnan(tensor).sum()),
                               int(torch.isinf(tensor).sum())))
    print(f"  params: {n_params:,} | approx size: {total_bytes / 1e6:.1f} MB")
    print(f"  non-finite tensors: {len(non_finite)}")
    for name, nan, inf in non_finite[:10]:
        print(f"    {name}: nan={nan} inf={inf}")

    # Config snapshot vs shipped model.yaml
    cfg = state.get("config") or {}
    print("\n=== config snapshot (first-level keys) ===")
    if isinstance(cfg, dict):
        for k in list(cfg)[:15]:
            print(f"  {k}: {str(cfg[k])[:120]}")

    # Cross-check the release package model checksum.
    release_model = Path(
        r"D:\CropPrep\releases\v2.0.0\cropfusion_release-v2.0.0\model\cropfusion.pt"
    )
    manifest = json.loads(
        (release_model.parent.parent / "version" / "manifest.json").read_text(encoding="utf-8")
    )
    checksums = json.loads(
        (release_model.parent.parent / "version" / "checksum.json").read_text(encoding="utf-8")
    )
    print("\n=== release model provenance ===")
    print("  manifest released_at:", manifest.get("released_at"))
    print("  manifest model_version:", manifest.get("model_version"))
    print("  expected sha256:", checksums.get("files", {}).get("model/cropfusion.pt"))
    print("  actual sha256:  ", _sha256(release_model))

    # Compare release metrics against both runs to pin provenance.
    release_metrics = json.loads(
        (release_model.parent.parent / "reports" / "metrics.json").read_text(encoding="utf-8")
    )
    dk_pipeline = Path(r"D:\CropPrep\kaggle_runs\train-dk-bridge\reports\CropPrep"
                       r"\training\kaggle\outputs\reports\pipeline.json")
    v6_pipeline = Path(r"D:\CropPrep\kaggle_runs\train-v6\CropPrep"
                       r"\training\kaggle\outputs\reports\pipeline.json")
    print("\n=== release metrics vs run provenance ===")
    print("  release crop/support:", release_metrics["metrics"].get("crop/support"),
          "acc:", release_metrics["metrics"].get("crop/accuracy"))
    print("  release yield/r2:", release_metrics["metrics"].get("yield/r2"),
          "support:", release_metrics["metrics"].get("yield/support"))
    print("  release registered:", release_metrics.get("registered", {}).get("created_at"))
    if dk_pipeline.exists():
        dk = json.loads(dk_pipeline.read_text(encoding="utf-8"))
        eval_ = dk["training"]["report"]["evaluation"]["metrics"]
        print("  dk-bridge  crop/support:", eval_.get("crop/support"),
              "yield/support:", eval_.get("yield/support"),
              "yield/r2:", eval_.get("yield/r2"))
    if v6_pipeline.exists():
        v6 = json.loads(v6_pipeline.read_text(encoding="utf-8"))
        eval_ = v6["training"]["report"]["evaluation"]["metrics"]
        print("  train-v6   crop/support:", eval_.get("crop/support"),
              "yield/support:", eval_.get("yield/support"),
              "yield/r2:", eval_.get("yield/r2"))

    # Forward-pass sanity of the shipped TorchScript model.
    try:
        sys.path.insert(0, str(_REPO_ROOT / "application"))
        from inference_package.release.loader import ReleasePackageLoader  # noqa: E402
        pkg = ReleasePackageLoader(str(release_model.parent.parent)).load()
        with torch.inference_mode():
            b, seq = 2, 4
            out = pkg.model(
                tabular=torch.randn(b, 128),
                ndvi=torch.randn(b, seq, 1, 224, 224),
                evi=torch.randn(b, seq, 1, 224, 224),
                temporal_mask=torch.ones(b, seq),
            )
        for i, o in enumerate(out):
            print(f"  forward[{i}] shape={tuple(o.shape)} finite={bool(torch.isfinite(o).all())}")
    except Exception as exc:  # noqa: BLE001 - report forward-check failure, keep audit result
        print("  forward-pass check failed:", type(exc).__name__, exc)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
