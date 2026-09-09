"""Kaggle entry point — Phase 4 image benchmark (thin wrapper).

Run inside a Kaggle Notebook on the GPU accelerator with the dataset
``shathanandabhatn/crop-yield-forecasting-karnataka-dakshina-kannada``
attached:

    !python kaggle/train_image.py

This only:

  1. makes the repo importable (canonical code lives in ``training/``),
  2. prints the ACTUAL structure of the attached Kaggle dataset so the
     discovery code's assumptions can be audited against reality,
  3. invokes the canonical ``training.train_image.py`` with the Kaggle
     defaults (auto-detected mount, output to ``/kaggle/working/outputs/image``).

After it finishes, evaluate the selected model ONCE on the untouched test
split (same notebook):

    !python kaggle/train_image.py --evaluate --final-test

then copy ``/kaggle/working/outputs/image`` -> ``models/image/`` locally.
The notebook run config used (GPU name, pytorch/timm versions, dataset
version/datetime) must be recorded into docs/PHASE_4_IMAGE_RESULTS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_MOUNT = Path("/kaggle/input/crop-yield-forecasting-karnataka-dakshina-kannada")


def _dump_data_inventory() -> None:
    """Best-effort tree + first-file previews of the mounted dataset."""
    print("Kaggle dataset mount:", DATA_MOUNT)
    print("dataset present:", DATA_MOUNT.is_dir())
    if not DATA_MOUNT.is_dir():
        return
    print("\nDATASET INVENTORY (structure the discovery code will see):")
    n_files = 0
    for i, p in enumerate(sorted(DATA_MOUNT.rglob("*"))):
        if i >= 400:
            print("... (truncated)")
            break
        if p.is_file():
            n_files += 1
            print(f"  FILE {p.relative_to(DATA_MOUNT)}  ({p.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--evaluate")]
    evaluate = any(a.startswith("--evaluate") for a in sys.argv[1:])
    _dump_data_inventory()

    cmd = ["--repo-root", str(REPO_ROOT),
           "--image-dir", str(DATA_MOUNT),
           "--output", "/kaggle/working/outputs/image"]
    cmd += args
    sys.argv = ["train_image.py"] + cmd
    if evaluate:
        from training.evaluate_image import main as eval_main
        sys.argv[0] = "evaluate_image.py"
        eval_main()
    else:
        from training.train_image import main as train_main
        train_main()


if __name__ == "__main__":
    main()