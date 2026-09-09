"""evaluate — evaluate all models on the untouched test set (SKELETON).

Phase 2 implementation. Phase 1 provides the interface/structure only.
No training/evaluation in Phase 1.

Intended flow:
    load tabular-only, image-only, and fusion model results
        -> run the SAME metric set on the SAME untouched spatial test set (Sullia)
        -> produce a comparison table proving fusion > tabular-only, image-only
        -> persist an evaluation report (metrics.json)

Constraint: identical split/validation/metrics for every model so the
fusion-vs-baseline comparison is honest (docs/NEW_TRAINING_ARCHITECTURE.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate_all_models(
    repo_root: Path,
    master_dataset_dir: Path | None = None,
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Evaluate tabular/image/fusion models. Phase 2: implement. Phase 1: raise NotImplemented."""
    raise NotImplementedError("evaluate is a Phase 2 implementation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all models (skeleton).")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset", default=None, help="Master dataset dir (optional).")
    parser.add_argument("--results", default=None, help="Training results dir (optional).")
    parser.add_argument("--output", default=None, help="Output dir (optional).")
    args = parser.parse_args()

    evaluate_all_models(
        repo_root=Path(args.repo_root),
        master_dataset_dir=Path(args.dataset) if args.dataset else None,
        results_dir=Path(args.results) if args.results else None,
        output_dir=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
