"""Extract / persist the imagery-embedding cache for the feature-fusion campaign.

What is cached
--------------
The legitimate, locally-reproducible "imagery" representation: the R5.10 DK
grid vegetation-composite vectors are converted to the per-(field, grid-year)
standardized tensor [N, T, F] plus temporal validity mask [N, T], keyed by
``field_id`` + ``grid_year``. Normalization uses train-split statistics only
(identical to ``prepare_imagery``); the cache is a pure re-materialization of
the frozen R5.10 input, so it is fully reproducible and gitignored.

On a platform with real Sentinel-2 patches this stage is where patch ids are
expanded to an on-disk embedding bank (e.g. EfficientNet pool vectors keyed by
patch_id); the training script loads the same cache interface.

Layout
------
    artifacts/feature_fusion/image_embeddings/
        image_embeddings.npz        X [N,4,12], mask [N,4], field_id, ...
        image_embeddings_meta.json  cols / timesteps / split hash / source
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.kaggle.scripts.feature_fusion_utils import (  # noqa: E402
    CACHE_DIR, IMG_CACHE_FILE, IMG_CACHE_META, load_image_cache, save_image_cache,
)
from training.kaggle.scripts.final_cropfusion import (  # noqa: E402
    build_population, load_ftd, prepare_imagery,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="reload the cache and print a digest")
    args = ap.parse_args(argv)

    if args.verify:
        c = load_image_cache()
        print(f"cache: X={c['X'].shape} mask={c['mask'].shape} "
              f"fields={len(c['field_id'])}")
        print(f"meta: {IMG_CACHE_META.read_text()[:400]}...")
        return 0

    ftd = load_ftd()
    pop = build_population(ftd)
    img = prepare_imagery(pop)
    save_image_cache(pop, img)
    meta = json.loads(IMG_CACHE_META.read_text())
    print(f"wrote {IMG_CACHE_FILE} -> X={img['X'].shape}, "
          f"mask=>={img['mask'].shape}, split_hash={meta['split_hash']}")
    print(f"dir: {CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())