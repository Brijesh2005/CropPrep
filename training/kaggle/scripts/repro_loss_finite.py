"""R5.2 Task 4 reproduction: does the training loss stay finite on real data?

Builds the tabular-only model (image disabled locally — no raster available),
runs the real MultiTaskLoss over every accepted observation, and reports
finiteness + loss composition. Also reproduces the temporal split composition.

Run: python training/kaggle/scripts/repro_loss_finite.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

CORPUS = Path(
    r"D:\CropPrep\kaggle_runs\train-dk-bridge\reports\CropPrep\training\kaggle\outputs\reports\corpus.json"
)


def main() -> int:
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    obs = [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]
    print(f"accepted observations: {len(obs)}")

    pre = Preprocessor(
        load_preprocessing_config(r"D:\CropPrep\training\config\preprocessing.yaml")
    )
    train, val, test = split_observations(obs, pre.config.split)
    print(
        f"temporal split -> train={len(train)} val={len(val)} test={len(test)}"
    )
    print("train years:", sorted({o.temporal.year for o in train}))
    print("val years:  ", sorted({o.temporal.year for o in val}))
    print("test years: ", sorted({o.temporal.year for o in test}))
    print(
        "crop labels per split:",
        sum(o.crop is not None for o in train),
        sum(o.crop is not None for o in val),
        sum(o.crop is not None for o in test),
    )
    print(
        "yield present per split:",
        sum(o.yield_value is not None for o in train),
        sum(o.yield_value is not None for o in val),
        sum(o.yield_value is not None for o in test),
    )

    accepted, _ = pre.filter(train)
    pre.fit(accepted)

    mc = ModelConfig.from_preprocessor(pre)
    mc.image_encoder.backbone = None  # tabular-only (no raster locally)
    model = ModelFactory.create(mc)
    model.eval()

    cfg = load_training_config(r"D:\CropPrep\training\config\training.yaml")
    counts = torch.tensor([64.0, 7.0, 1.0, 1.0, 1.0])
    loss = MultiTaskLoss(
        cfg.loss,
        class_weights={
            "crop": build_class_weights(cfg.loss, mc.heads.crop.num_classes, counts)
        },
    )

    with torch.no_grad():
        for name, split_obs in (("train", train), ("val", val), ("test", test)):
            tabular = torch.stack(
                [pre.tabular.transform(o).float() for o in split_obs]
            )
            labels = [pre.label.transform(o) for o in split_obs]
            yields = torch.stack([y for _c, y in labels])
            crops = torch.stack([c for c, _y in labels])
            print(f"\n--- {name}: n={len(split_obs)}")
            print(
                f"  tabular {tuple(tabular.shape)} finite={torch.isfinite(tabular).all().item()} "
                f"min={float(tabular.min()):.3f} max={float(tabular.max()):.3f}"
            )
            print(
                f"  yield_label finite={torch.isfinite(yields).all().item()} "
                f"min={float(yields.min()):.4f} max={float(yields.max()):.4f}"
            )
            print(f"  crop_label codes>=0: {(crops >= 0).sum().item()} / {crops.numel()}")
            out = model({"tabular": tabular})
            yield_pred = out.yield_pred
            print(
                f"  yield_pred finite={torch.isfinite(yield_pred).all().item()} "
                f"min={float(yield_pred.min()):.4f} max={float(yield_pred.max()):.4f}"
            )
            if out.crop_logits is not None:
                print(
                    f"  crop_logits finite={torch.isfinite(out.crop_logits).all().item()}"
                )
            total, per = loss(
                {"crop": out.crop_logits, "yield": yield_pred},
                {"crop": crops, "yield": yields},
            )
            print(
                f"  total finite={bool(torch.isfinite(total).item())} "
                f"| per_task={ {k: round(float(v), 6) for k, v in per.items()} }"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
