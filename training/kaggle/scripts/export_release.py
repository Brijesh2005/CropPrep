"""Kaggle release exporter — package a trained model for serving (orchestration).

Exports a trained checkpoint to TorchScript + ONNX via
:class:`ModelExporter`, snapshots the model config, and writes a release
manifest JSON. The actual export *engines* live in ``training/models`` — this
script only wires the checkpoint.

The example inputs used to trace the graphs are derived from the checkpoint
model's own config (:meth:`CropFusionModel.sample_batch`), guaranteeing the
exported artifacts accept exactly the schema the model was trained on.

Run on Kaggle::

    !python training/kaggle/scripts/export_release.py \\
        --checkpoint /kaggle/working/runs/run-name/checkpoint_epoch0100.pt \\
        --output /kaggle/working/release

Run on a research machine::

    python training/kaggle/scripts/export_release.py \\
        --repo-root . --checkpoint runs/run-name/checkpoint_epoch0100.pt \\
        --output artifacts/release
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _add_repo_root(repo_root: Path) -> None:
    """Force the repository root to the front of ``sys.path``."""
    import sys

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
            print(f"[export_release] removing shadowing sys.path entry: {entry}")
            sys.path.remove(entry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cropfusion-export", description="Export a trained model for serving"
    )
    parser.add_argument("--repo-root", default=str(_REPO_ROOT))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output", default=str(_REPO_ROOT / "artifacts" / "release")
    )
    args = parser.parse_args(argv)

    _add_repo_root(Path(args.repo_root))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from training.models import ModelExporter
    from training.models.factory import ModelFactory

    model = ModelFactory.from_checkpoint(Path(args.checkpoint))

    exporter = ModelExporter(model)
    torchscript_path = exporter.export_torchscript(output_dir / "model.torchscript.pt")
    onnx_path = exporter.export_onnx(output_dir / "model.onnx")

    config_path = output_dir / "model.yaml"
    model.save_config(config_path)

    manifest = {
        "released_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(args.checkpoint)),
        "model_config": model.config.model_dump(mode="json"),
        "artifacts": {
            "torchscript": str(torchscript_path),
            "onnx": str(onnx_path),
            "config": str(config_path),
        },
    }
    manifest_path = output_dir / "release.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
