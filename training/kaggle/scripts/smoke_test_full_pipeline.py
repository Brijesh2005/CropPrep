"""R5.2 smoke test: dataset -> ModelInput -> model -> loss -> eval.

End-to-end run over the REAL dk-bridge corpus (275 accepted observations)
without any raster imagery (image branch disabled locally; tabular + labels
use the real pipelines):

  1. Load accepted observations from corpus.json.
  2. Temporal split (train/val/test) via the real splitter.
  3. Preprocessor.filter + fit on train, transform all splits.
  4. Build the real CropFusionModel via ModelFactory (tabular branch).
  5. Run the real MultiTaskLoss over every split -> finiteness + shapes.
  6. Run the real Evaluator on the test split -> metrics, no NaN/Inf/errors.

PASS/FAIL criteria: every tensor finite, shapes match the model contract,
evaluation completes and reports metrics. Any NaN/Inf or shape mismatch fails
the smoke test (fail-loudly, matching the R5.2 nan_policy=stop posture).

Run: python training/kaggle/scripts/smoke_test_full_pipeline.py
"""

from __future__ import annotations

import json
import math
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
from training.training.evaluator import Evaluator  # noqa: E402
from training.training.losses import MultiTaskLoss, build_class_weights  # noqa: E402
from training.training.validator import Validator  # noqa: E402

CORPUS = Path(
    r"D:\CropPrep\kaggle_runs\train-dk-bridge\reports\CropPrep"
    r"\training\kaggle\outputs\reports\corpus.json"
)

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name} {detail}")
        failures.append(name)


def main() -> int:
    print("=== R5.2 SMOKE TEST: dataset -> ModelInput -> model -> loss -> eval ===")

    # -- 1. Dataset ---------------------------------------------------- #
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    obs = [
        AgriculturalObservation.model_validate(s["observation"])
        for s in raw["samples"]
        if s["status"] == "accepted" and s.get("observation")
    ]
    print(f"\n[1] dataset: {len(obs)} accepted observations")
    check("accepted>=100 (corpus not empty)", len(obs) >= 100, f"n={len(obs)}")

    # -- 2. Temporal split ---------------------------------------------- #
    pre = Preprocessor(
        load_preprocessing_config(r"D:\CropPrep\training\config\preprocessing.yaml")
    )
    train, val, test = split_observations(obs, pre.config.split)
    print(f"[2] temporal split -> train={len(train)} val={len(val)} test={len(test)}")
    check("all splits non-empty",
          bool(train) and bool(val) and bool(test),
          f"{len(train)}/{len(val)}/{len(test)}")

    # -- 3. Preprocess -------------------------------------------------- #
    accepted, _ = pre.filter(train)
    pre.fit(accepted)
    print("[3] preprocessor fit on train")

    def make_batch(split_obs):
        tabular = torch.stack([pre.tabular.transform(o).float() for o in split_obs])
        labels = [pre.label.transform(o) for o in split_obs]
        yields = torch.stack([y for _c, y in labels])
        crops = torch.stack([c for c, _y in labels])
        return {"tabular": tabular}, {"crop": crops, "yield": yields}

    # -- 4. Model -------------------------------------------------------- #
    mc = ModelConfig.from_preprocessor(pre)
    mc.image_encoder.backbone = None  # tabular-only locally (no raster)
    model = ModelFactory.create(mc)
    model.eval()
    print("\n[4] model built (tabular branch; image disabled locally):")
    print(f"     tabular_feature_dim={mc.tabular_feature_dim} "
          f"num_classes={mc.heads.crop.num_classes}")

    # -- 5. Loss ---------------------------------------------------------- #
    cfg = load_training_config(r"D:\CropPrep\training\config\training.yaml")
    counts = torch.tensor([64.0, 7.0, 1.0, 1.0, 1.0])
    loss = MultiTaskLoss(
        cfg.loss,
        class_weights={
            "crop": build_class_weights(cfg.loss, mc.heads.crop.num_classes, counts)
        },
    )
    print(f"[5] loss: {type(loss).__name__} (nan_policy default: stop)")

    with torch.no_grad():
        for name, split_obs in (("train", train), ("val", val), ("test", test)):
            inputs, targets = make_batch(split_obs)
            check(f"{name} tabular finite",
                  torch.isfinite(inputs["tabular"]).all().item())
            check(f"{name} yield label finite",
                  torch.isfinite(targets["yield"]).all().item())
            out = model(inputs)
            check(f"{name} yield_pred finite", torch.isfinite(out.yield_pred).all().item())
            check(f"{name} yield_pred shape",
                  tuple(out.yield_pred.shape) == (len(split_obs), 1),
                  str(tuple(out.yield_pred.shape)))
            if out.crop_logits is not None:
                check(f"{name} crop_logits finite",
                      torch.isfinite(out.crop_logits).all().item())
                check(f"{name} crop_logits shape",
                      tuple(out.crop_logits.shape) == (len(split_obs), mc.heads.crop.num_classes),
                      str(tuple(out.crop_logits.shape)))
            total, per = loss(
                {"crop": out.crop_logits, "yield": out.yield_pred}, targets
            )
            check(f"{name} total loss finite", torch.isfinite(total).item(),
                  f"total={float(total)}")
            for k, v in per.items():
                check(f"{name} {k} loss finite", torch.isfinite(v).item(),
                      f"{k}={float(v)}")

    # -- 6. Evaluator on test --------------------------------------------- #
    print("\n[6] Evaluator over test split:")
    test_inputs, test_targets = make_batch(test)
    eval_batch = dict(test_inputs)
    eval_batch["crop_label"] = test_targets["crop"]
    eval_batch["yield_label"] = test_targets["yield"]

    class _DummyLoader:
        def __init__(self, batch):
            self._b = batch

        def __iter__(self):
            yield self._b

    validator = Validator(
        model,
        loss,
        metrics_config=cfg.metrics,
        nan_policy="stop",
    )
    val_result = validator.validate(_DummyLoader(eval_batch), epoch=0)
    print(f"     validator val_loss={val_result.val_loss} samples={val_result.samples}")
    print(f"     val per_task={ {k: round(float(v), 6) for k, v in val_result.per_task_losses.items()} }")
    check("validator val_loss finite", math.isfinite(val_result.val_loss),
          f"val_loss={val_result.val_loss}")
    check("validator ran >0 samples", val_result.samples == len(test),
          f"samples={val_result.samples}")

    ev = Evaluator(
        model,
        metrics_config=cfg.metrics,
        input_map=validator.input_map,
    )
    result = ev.evaluate(_DummyLoader(eval_batch), loss_module=loss)
    metrics = result.metrics
    print(f"     test_loss={metrics.get('test_loss')}")
    print(f"     yield/rmse={metrics.get('yield/rmse')} r2={metrics.get('yield/r2')}")
    print(f"     crop/support={metrics.get('crop/support')} acc={metrics.get('crop/accuracy')}")
    print(f"     multi_task_score={result.multi_task_score}")
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            check(f"eval metric {key} finite", False, str(value))
    check("eval test_loss finite",
          torch.isfinite(torch.tensor(metrics.get("test_loss", float("nan")))).item())
    check("eval yield/rmse finite",
          torch.isfinite(torch.tensor(metrics.get("yield/rmse", float("nan")))).item())

    # -- Summary ------------------------------------------------------------ #
    summary = {
        "passed": not failures,
        "failures": failures,
        "dataset": {"accepted": len(obs)},
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "model": {
            "tabular_feature_dim": mc.tabular_feature_dim,
            "num_classes": mc.heads.crop.num_classes,
            "image_branch": False,
        },
        "eval": {
            "test_loss": metrics.get("test_loss"),
            "yield_rmse": metrics.get("yield/rmse"),
            "yield_r2": metrics.get("yield/r2"),
            "yield_support": metrics.get("yield/support"),
            "crop_support": metrics.get("crop/support"),
            "crop_accuracy": metrics.get("crop/accuracy"),
            "multi_task_score": result.multi_task_score,
        },
        "validator": {"val_loss": val_result.val_loss, "samples": val_result.samples},
    }
    out_dir = _REPO_ROOT / "training" / "artifacts" / "smoke_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_test_result.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nresult written: {out_dir / 'smoke_test_result.json'}")
    print(f"=== SMOKE TEST {'PASSED' if not failures else 'FAILED'} "
          f"({len(failures)} failure(s)) ===")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
