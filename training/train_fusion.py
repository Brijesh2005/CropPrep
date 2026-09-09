"""train_fusion — train the fusion model (SKELETON).

Phase 2 implementation. Phase 1 provides the interface/structure only.
No training in Phase 1.

Intended flow:
    compare to the fixed architecture (docs/NEW_TRAINING_ARCHITECTURE.md):
        tabular branch  -> tabular encoder embedding
        image branch    -> image encoder embedding
        concatenate learned embeddings -> fusion MLP
        heads: crop classification (softmax) + yield regression
        joint/fine-tuned training on train (Belthangady/Mangalore/Bantwal)
        tune on val (Puttur), evaluate on untouched test (Sullia)

Constraint: the fusion model MUST experimentally outperform both tabular-only
and image-only baselines using the same validations and the untouched spatial
test set. No manipulation of test labels or splits (docs/NEW_TRAINING_ARCHITECTURE.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_fusion_model(
    repo_root: Path,
    master_dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    yield_target: str | None = None,
) -> dict:
    """Train the fusion model. Phase 2: implement. Phase 1: raise NotImplemented."""
    raise NotImplementedError("train_fusion is a Phase 2 implementation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fusion model (skeleton).")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset", default=None, help="Master dataset dir (optional).")
    parser.add_argument("--output", default=None, help="Output dir (optional).")
    parser.add_argument(
        "--yield-target",
        default=None,
        help="Yield regression column. PLACEHOLDER — resolve later.",
    )
    args = parser.parse_args()

    train_fusion_model(
        repo_root=Path(args.repo_root),
        master_dataset_dir=Path(args.dataset) if args.dataset else None,
        output_dir=Path(args.output) if args.output else None,
        yield_target=args.yield_target,
    )


if __name__ == "__main__":
    main()
