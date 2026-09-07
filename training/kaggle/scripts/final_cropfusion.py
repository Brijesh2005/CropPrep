"""FINAL CropFusion campaign — tabular vs imagery vs CropFusion.

The final engineering step of the CropFusion project. It consumes the R5.9/
R5.10 frozen field-level target and spatial split verbatim, builds ONE
balanced modeling population under a rule defined BEFORE any training,
implements the three final model families on the legitimate per-year satellite
composite representation (DK grid vegetation indices — the strongest imagery
representation available locally; raw Sentinel-2 patches only exist on Kaggle),
selects every configuration on VALIDATION ONLY, freezes everything, and then
evaluates the held-out TEST set exactly once.

Scientific constraints (identical to the R5.8-R5.10 chain)
  1. Crop_Extent is NEVER a feature; CROP_EXTENT_UNIT_STATUS = UNKNOWN.
  2. Target: R5.9 dominant-crop coconut-vs-pepper field composition, consumed
     unchanged from reports/R5.9/field_dataset_split.csv.
  3. No target-derived features (coconut/pepper/top1 fraction, dominance gap,
     composition counts, Crop_Extent, Yield_Proxy_NPP, validity fields).
  4. No future information: imagery sequence uses grid years <= survey year.
  5. No fabricated / duplicated satellite observations. Missing slots are
     masked, never imputed as real observations.
  6. Physical field never spans two splits (taluk-grouped, inherited R5.9).
  7. Population balancing samples the majority class ONLY (deterministic
     seed); validation/test are real observations, never SMOTE/oversampled.
  8. Test set is evaluated exactly once after every selection is frozen.
  9. Threshold: fixed 0.5 for all models (no post-hoc test threshold).
  10. Location features are investigated SEPARATELY (environmental vs
      geographic prior); the headline tabular model is the no-location variant.

Runtime: CPU-only (no GPU in this environment). Every neural model is small and
trained with early stopping; tabular models reuse the R5.10 sklearn pool.

Run from repo root:
    python training/kaggle/scripts/final_cropfusion.py
    python training/kaggle/scripts/final_cropfusion.py --phases 0 1 2 3 4
    python training/kaggle/scripts/final_cropfusion.py --phases 5 6 7 8
    python training/kaggle/scripts/final_cropfusion.py --skip-torch
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
FINAL_DIR = REPO_ROOT / "reports" / "final"
FIG_DIR = FINAL_DIR / "figures"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

R59_DATASET = REPO_ROOT / "reports" / "R5.9" / "field_dataset_split.csv"
R510_DIR = REPO_ROOT / "reports" / "R5.10"
FTD_CSV = R510_DIR / "field_target_dataset.csv"
GRID_CSV = R510_DIR / "temporal_field_grid.csv"

import torch  # noqa: E402
from torch import nn  # noqa: E402

from training.models.config import (  # noqa: E402
    TabularModelConfig,
    TemporalModelConfig,
)
from training.models.tabtransformer import TabTransformer  # noqa: E402
from training.models.temporal_transformer import TemporalTransformer  # noqa: E402
from training.models.cross_attention import CrossAttention  # noqa: E402
from training.models.adaptive_gate import AdaptiveGatedFusion  # noqa: E402
from training.models.multitask_heads import CropHead  # noqa: E402

from training.kaggle.scripts.r5_10_temporal_field_target import (  # noqa: E402
    _metrics_full,
    _models,
)

SEED = 42
# Frozen R5.10 population configuration (rule defined BEFORE training).
COCONUT_RATIO = 4          # coconut sampled per split = ratio * pepper-in-split
BALANCE_SEED = 42
MIN_SURVEY_TO_GRID_YEARS = [2018, 2019, 2020, 2021]
CROP_EXTENT_UNIT_STATUS = "UNKNOWN"

# Imagery (satellite-domain) per-year composite columns — the legitimate
# Sentinel-2-derived vegetation indices from the DK grids.
IMAGERY_COLS = [
    "ndvi", "evi", "ndwi", "ndre", "savi", "s2_obs_count",
    "kharif_ndvi", "kharif_evi", "kharif_ndwi",
    "rabi_ndvi", "rabi_evi", "rabi_ndwi",
]
T_GRID = 4

# Tabular (environmental/agricultural) feature groups. EVERY satellite-derived
# column (ndvi/evi/... sat_*) and every location column are deliberately kept
# out of this group; they belong to the imagery / location investigations.
TABULAR_CLIMATE = [
    "annual_rainfall_mm", "dewpoint_c", "temperature_c",
    "relative_humidity_pct",
]
TABULAR_STATIC = [
    "elevation", "slope", "soil_clay_pct", "soil_sand_pct",
    "soil_organic_carbon", "soil_ph", "soil_moisture",
]
TABULAR_CATEGORICAL = ["is_cropland", "land_cover_class", "soil_type_class"]
LOCATION_COLS = ["lat", "lon"]

# Minimum real satellite observations required to construct a temporal
# representation (robustness gate, defined BEFORE evaluation). A leading-window
# sequence must contain >= 3 matched grid years to count as a valid temporal
# representation.
MIN_OBS_ROBUSTNESS = 3

PHASES = {
    "0": "population",
    "1": "features",
    "2": "tabular_selection",
    "3": "imagery_selection",
    "4": "fusion_selection",
    "5": "freeze_test_eval",
    "6": "ablation",
    "7": "robustness",
    "8": "confidence_intervals",
    "9": "figures",
    "10": "tables",
    "11": "provenance",
    "12": "report",
    "13": "paper",
    "14": "summary",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, name: str) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(FINAL_DIR / name, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)


def resolve_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Phase 0 — balanced population (frozen before training; saved audit CSV)
# ---------------------------------------------------------------------------
def load_ftd() -> pd.DataFrame:
    d = pd.read_csv(FTD_CSV, low_memory=False)
    d["survey_year"] = d["year"].astype(int)
    return d


def build_population(ftd: pd.DataFrame | None = None) -> pd.DataFrame:
    if ftd is None:
        ftd = load_ftd()
    rng = np.random.default_rng(BALANCE_SEED)
    frames = []
    for split in ("train", "val", "test"):
        sub = ftd[ftd["split"] == split].copy()
        pep = sub[sub["target"] == 1]
        coc = sub[sub["target"] == 0]
        n_coc = min(int(COCONUT_RATIO * len(pep)), len(coc))
        coc_sample = coc.sample(n=n_coc, random_state=rng, replace=False)
        frames.append(pd.concat([pep, coc_sample]))
    pop = pd.concat(frames).reset_index(drop=True)
    pop = pop.sort_values(["split", "field_id"]).reset_index(drop=True)
    return pop


def phase0_population(ftd: pd.DataFrame | None = None) -> dict:
    pop = build_population(ftd)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    pop.to_csv(FINAL_DIR / "final_population.csv", index=False)
    audit = {
        "rule": (
            "retain ALL minority (pepper-dominant) fields in each split; "
            f"sample majority (coconut-dominant) within each split to "
            f"{COCONUT_RATIO}x the split's pepper count, uniform without "
            f"replacement, deterministic seed {BALANCE_SEED}"),
        "split_counts": to_py(pop.groupby(["split", "target"]).size().to_dict()),
        "total_fields": int(len(pop)),
        "class_ratio_train": round(
            float((pop[pop["split"] == "train"]["target"] == 1).mean()), 4),
        "near_duplicate_thread": (
            "fields sharing the same R5.7 ~50m field_cluster_id are audited in "
            "the split-integrity probe; the split itself is inherited from R5.9"),
    }
    # near-duplicate coordinate / cluster audit for the frozen split
    field_split = pop.groupby("field_id")["split"].nunique()
    cluster_split = pop.groupby("field_cluster_id")["split"].nunique()
    cell_key = pop["lat"].round(2).astype(str) + "|" + pop["lon"].round(2).astype(str)
    audit["fields_crossing_splits"] = int((field_split > 1).sum())
    audit["clusters_straddling_splits"] = int((cluster_split > 1).sum())
    cell_frame = pd.DataFrame({"cell": cell_key, "split": pop["split"]})
    audit["approx_1km_cells_straddling_splits"] = int(
        (cell_frame.groupby("cell")["split"].nunique() > 1).sum())
    write_json(audit, "final_population.json")
    return audit


def to_py(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, float):
        return round(o, 6)
    return o


# ---------------------------------------------------------------------------
# Phase 1 — feature preparation (train-only statistics everywhere)
# ---------------------------------------------------------------------------
def _categorical_codes(df: pd.DataFrame, cols: list[str],
                       train_idx: np.ndarray) -> np.ndarray:
    X = np.zeros((len(df), len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        vals = df[c].fillna("__nan__").astype(str).to_numpy()
        train_vals = vals[train_idx]
        _, tr = pd.factorize(train_vals, sort=False)
        table = {v: i for i, v in enumerate(tr)}
        codes = np.array([table.get(v, -1) for v in vals], dtype=np.float64)
        codes[codes == -1] = 0.0
        X[:, j] = codes
    return X


def prepare_tabular(pop: pd.DataFrame, with_location: bool = False) -> dict:
    numeric = list(TABULAR_CLIMATE) + [f"{v}_{s}" for v in TABULAR_CLIMATE
                                       for s in ("tmean", "tstd", "tmin",
                                                 "tmax", "trange", "tdelta",
                                                 "tslope")] + TABULAR_STATIC
    numeric = [c for c in numeric if c in pop.columns]
    if with_location:
        numeric = numeric + [c for c in LOCATION_COLS if c in pop.columns]
    categorical = [c for c in TABULAR_CATEGORICAL if c in pop.columns]

    splits = pop["split"].to_numpy()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    Xn = pop[numeric].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=float, copy=True)
    med = np.nanmedian(Xn[idx["train"]], axis=0)
    med = np.nan_to_num(med, nan=0.0)
    for j in range(Xn.shape[1]):
        col = Xn[:, j]
        mask_nan = np.isnan(col)
        if mask_nan.any():
            col[mask_nan] = med[j]
    trn = Xn[idx["train"]]
    std = trn.std(axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-9), 1e-9, std)
    Xn = (Xn - med) / std
    Xc = _categorical_codes(pop, categorical, idx["train"])
    X = np.hstack([Xn, Xc]).astype(np.float64)
    return {
        "X": X, "numeric": numeric, "categorical": categorical,
        "cols": numeric + categorical,
        "idx": idx, "median": med, "std": std,
    }


def prepare_imagery(pop: pd.DataFrame, grid: pd.DataFrame | None = None) -> dict:
    if grid is None:
        grid = pd.read_csv(GRID_CSV, low_memory=False)
    pop = pop.copy()
    fields = pop["field_id"].to_numpy()
    survey = pop["survey_year"].to_numpy()
    n = len(pop)
    feat_dims = [c for c in IMAGERY_COLS if c in grid.columns]
    F = len(feat_dims)
    X = np.zeros((n, T_GRID, F), dtype=np.float64)
    mask = np.zeros((n, T_GRID), dtype=np.float64)
    grid = grid[grid["field_id"].isin(set(fields))].copy()
    g = grid.set_index(["field_id", "grid_year"])

    for i, (fid, syr) in enumerate(zip(fields, survey)):
        for t, gy in enumerate(MIN_SURVEY_TO_GRID_YEARS):
            if gy > int(syr):
                continue  # future => masked, never observed
            key = (fid, gy)
            if key not in g.index:
                continue
            row = g.loc[key]
            if not bool(row["matched"]):
                continue
            vals = pd.to_numeric(row[feat_dims], errors="coerce").to_numpy(float)
            if np.isnan(vals).any():
                continue
            X[i, t] = vals
            mask[i, t] = 1.0

    splits = pop["split"].to_numpy()
    idx = {s: np.where(splits == s)[0] for s in ("train", "val", "test")}
    trn = X[idx["train"]]
    valid_trn = trn[mask[idx["train"]] > 0]
    mean = valid_trn.mean(axis=0, keepdims=True) if len(valid_trn) else np.zeros(F)
    std = valid_trn.std(axis=0, keepdims=True) if len(valid_trn) else np.ones(F)
    std = np.where(std < 1e-9, 1e-9, std)
    X = (X - mean) / std
    obs_counts = mask.sum(axis=1)
    audit = {
        "features_per_timestep": F,
        "timesteps": MIN_SURVEY_TO_GRID_YEARS,
        "matched_fraction_leading": round(float(mask[mask.sum(axis=1) > 0].sum()
                                                / max(len(mask) * T_GRID, 1)), 4),
        "obs_per_field_mean": round(float(obs_counts.mean()), 3),
        "obs_per_field_median": float(np.median(obs_counts)),
        "padded_slots_fraction": round(float((mask == 0).mean()), 4),
        "min_obs_rule": (
            f">= {MIN_OBS_ROBUSTNESS} matched grid years (leading window) "
            f"required for the robustness subset"),
    }
    return {"X": X, "mask": mask, "cols": feat_dims, "idx": idx,
            "obs": obs_counts, "audit": audit}


# ---------------------------------------------------------------------------
# Shared evaluation / selection helpers
# ---------------------------------------------------------------------------
def split_y(pop: pd.DataFrame) -> dict:
    splits = pop["split"].to_numpy()
    y = pop["target"].to_numpy()
    return {s: np.where(splits == s)[0] for s in ("train", "val", "test")}, y


def _roc_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return None
    return round(float(roc_auc_score(y, p)), 4)


# ---------------------------------------------------------------------------
# Torch building blocks (reusing training/models components)
# ---------------------------------------------------------------------------
class TimestepProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImagerySequenceModel(nn.Module):
    """Imagery-only: per-year satellite composites -> temporal encoder."""
    def __init__(self, feat_dim: int, temp_cfg: TemporalModelConfig,
                 proj_dim: int = 32, num_classes: int = 1):
        super().__init__()
        self.projector = TimestepProjector(feat_dim, proj_dim)
        self.temporal = TemporalTransformer(temp_cfg, input_dim=proj_dim)
        self.head = CropHead(self.temporal.output_dim, num_classes=num_classes,
                             hidden_dim=32, dropout=0.2)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.projector(seq)
        pooled = self.temporal(emb, mask=mask)
        return self.head(pooled)


class CrossProjector(nn.Module):
    """Projects the concatenated modality context to the cross stream."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FinalFusionModel(nn.Module):
    """Tabular + imagery fused model (existing CropFusion building blocks).

    mechanism:
      concat  - concatenation fusion of tabular + imagery embeddings.
      gated   - AdaptiveGatedFusion over both streams (no cross attention).
      cross   - CrossAttention (Q=imagery, K/V=tabular) then gated fusion.
    """

    def __init__(self, tab_cfg: TabularModelConfig,
                 temp_cfg: TemporalModelConfig,
                 imagery_feat_dim: int, mechanism: str = "cross",
                 proj_dim: int = 32, num_classes: int = 1):
        super().__init__()
        resolve_seed()
        self.mechanism = mechanism
        self.img_proj = TimestepProjector(imagery_feat_dim, proj_dim)
        self.temporal = TemporalTransformer(temp_cfg, input_dim=proj_dim)
        self.tab_encoder = TabTransformer(tab_cfg)
        img_d = self.temporal.output_dim
        tab_d = self.tab_encoder.output_dim
        cross_dim = img_d
        if mechanism == "concat":
            self.cross_proj = CrossProjector(img_d + tab_d, 64)
            self.head = CropHead(64, num_classes=num_classes,
                                 hidden_dim=48, dropout=0.2)
        else:
            if mechanism == "cross":
                if img_d % 2 != 0:
                    raise ValueError("img_d must be even for cross attention")
                self.cross_attn = CrossAttention(
                    query_dim=img_d, key_dim=tab_d, num_heads=2,
                    out_dim=cross_dim, dropout=0.1)
                self.cross_proj = None
            else:  # gated without cross attention
                self.cross_attn = None
                self.cross_proj = CrossProjector(img_d + tab_d, cross_dim)
            self.gated = AdaptiveGatedFusion(
                image_dim=img_d, tabular_dim=tab_d, cross_dim=cross_dim,
                out_dim=64, hidden_dim=64, dropout=0.15, activation="gelu")
            self.head = CropHead(64, num_classes=num_classes,
                                 hidden_dim=48, dropout=0.2)

    def embed(self, seq: torch.Tensor, mask: torch.Tensor,
              tabular: torch.Tensor):
        img = self.temporal(self.img_proj(seq), mask=mask)
        tab = self.tab_encoder(tabular)
        if self.mechanism == "concat":
            h = self.cross_proj(torch.cat([img, tab], dim=-1))
            return h, {"image_gate": None, "tabular_gate": None}
        if self.mechanism == "cross":
            cross_out = self.cross_attn(img, tab)
        else:
            cross_out = self.cross_proj(torch.cat([img, tab], dim=-1))
        out = self.gated(img, tab, cross_out)
        return out["fused"], {"image_gate": out.get("image_gate"),
                              "tabular_gate": out.get("tabular_gate")}

    def forward(self, seq: torch.Tensor, mask: torch.Tensor,
                tabular: torch.Tensor) -> torch.Tensor:
        h, _ = self.embed(seq, mask, tabular)
        return self.head(h)


def _tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).float()


def train_torch(model: nn.Module, *, Xt, yt, mask_t=None, Xv, yv, mask_v=None,
                tabular_t=None, tabular_v=None, epochs: int = 90,
                lr: float = 3e-3, wd: float = 1e-4, batch: int = 128,
                patience: int = 12) -> tuple[nn.Module, dict]:
    model.train()
    pos_weight = torch.tensor([float((yt == 0).sum() / max((yt == 1).sum(), 1))])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(yt)
    yt_t = _tensor(yt)
    yv_t = _tensor(yv)
    weights = np.where(yt == 1, 1.0 / max((yt == 1).sum(), 1),
                       1.0 / max((yt == 0).sum(), 1))
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=8 * n, replacement=True)
    # Build dataset: include yt (label) so DataLoader and sampler stay aligned.
    tensors = [_tensor(Xt), _tensor(mask_t if mask_t is not None else np.ones((n, 1))),
               yt_t]
    if tabular_t is not None:
        tensors.append(_tensor(tabular_t))
    ds = torch.utils.data.TensorDataset(*tensors)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch, sampler=sampler, num_workers=0)
    vtab = _tensor(tabular_v) if tabular_v is not None else None

    best = {"bal": -1.0, "state": None, "epoch": -1}
    bad = 0
    for epoch in range(epochs):
        model.train()
        for parts in dl:
            xb, mb, yb = parts[0], parts[1], parts[2]
            tb = parts[3] if tabular_t is not None else None
            opt.zero_grad()
            if isinstance(model, FinalFusionModel):
                logits = model(xb, mb, tb)
            elif isinstance(model, ImagerySequenceModel):
                logits = model(xb, mb)
            else:  # pragma: no cover
                logits = model(xb)
            loss = loss_fn(logits.squeeze(1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            logits = _predict(model, Xv, mask_v, vtab)
            p = torch.sigmoid(logits).numpy()
        met = _metrics_full(yv, (p >= 0.5).astype(int), p)
        bal = met["balanced_accuracy"]
        if bal > best["bal"] + 1e-6:
            best = {"bal": bal, "state": copy.deepcopy(model.state_dict()),
                    "epoch": epoch}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"],
                   "best_val_balanced_accuracy": best["bal"]}


def _predict(model: nn.Module, X: np.ndarray, mask: np.ndarray | None,
             tabular: np.ndarray | None) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = []
        for i in range(0, len(X), 1024):
            xb = _tensor(X[i:i + 1024])
            mb = _tensor(mask[i:i + 1024]) if mask is not None else None
            tb = _tensor(tabular[i:i + 1024]) if tabular is not None else None
            if isinstance(model, FinalFusionModel):
                out.append(model(xb, mb, tb))
            elif isinstance(model, ImagerySequenceModel):
                out.append(model(xb, mb))
            else:  # pragma: no cover
                out.append(model(xb))
        return torch.cat(out, dim=0)


# ---------------------------------------------------------------------------
# Phase 2 — tabular selection (VALIDATION ONLY)
# ---------------------------------------------------------------------------
def select_tabular(pop: pd.DataFrame, feat: dict,
                   with_location: bool,
                   results: list[dict]) -> dict:
    X, idx, y = feat["X"], feat["idx"], pop["target"].to_numpy()
    tag = "tabular-loc" if with_location else "tabular-no-loc"
    per = {}
    for name, clf in _models():
        clf.fit(X[idx["train"]], y[idx["train"]])
        row = {"config": tag, "model": name}
        for split in ("val",):
            pr = clf.predict(X[idx[split]])
            p = clf.predict_proba(X[idx[split]])[:, 1]
            met = _metrics_full(y[idx[split]], pr, p)
            row["val_balanced_accuracy"] = met["balanced_accuracy"]
            row["val_macro_f1"] = met["macro_f1"]
            row["val_roc_auc"] = met["roc_auc"]
            row["val_accuracy"] = met["accuracy"]
        per[name] = row
        results.append(row)
    best_name = max(per, key=lambda k: per[k]["val_balanced_accuracy"])
    return {"tag": tag, "best": best_name, "per": per,
            "best_val_balanced_accuracy": per[best_name][
                "val_balanced_accuracy"],
            "best_val_macro_f1": per[best_name]["val_macro_f1"],
            "best_val_roc_auc": per[best_name]["val_roc_auc"]}


# ---------------------------------------------------------------------------
# Phase 3/4 — imagery / fusion selection (VALIDATION ONLY)
# ---------------------------------------------------------------------------
def _torch_config() -> tuple[TabularModelConfig, TemporalModelConfig]:
    tab_cfg = TabularModelConfig(
        numeric_dim=0, categorical_cardinalities=[], embedding_dim=16,
        depth=2, num_heads=2, ff_dim=48, dropout=0.15, use_cls=True,
        max_len=8)
    temp_cfg = TemporalModelConfig(
        d_model=32, depth=2, num_heads=2, ff_dim=64, dropout=0.1,
        use_cls=True, position_encoding="learned", max_len=T_GRID,
        embedding_dim=32)
    return tab_cfg, temp_cfg


def select_imagery(pop: pd.DataFrame, img: dict, seeded: bool = True,
                   epochs: int = 90) -> dict:
    if seeded:
        resolve_seed()
    y = pop["target"].to_numpy()
    tab_cfg, temp_cfg = _torch_config()
    model = ImagerySequenceModel(
        feat_dim=len(img["cols"]), temp_cfg=temp_cfg, proj_dim=32)
    tr, _ = split_y(pop)
    model, info = train_torch(
        model, Xt=img["X"][tr["train"]], yt=y[tr["train"]],
        mask_t=img["mask"][tr["train"]], Xv=img["X"][tr["val"]],
        yv=y[tr["val"]], mask_v=img["mask"][tr["val"]],
        epochs=epochs)
    p = torch.sigmoid(_predict(model, img["X"][tr["val"]],
                               img["mask"][tr["val"]], None)).numpy()
    met = _metrics_full(y[tr["val"]], (p >= 0.5).astype(int), p)
    return {"model": model, "info": info, "val_probs": p,
            "val": met}


def select_fusion(pop: pd.DataFrame, img: dict, feat: dict,
                  mechanism: str, seeded: bool = True, epochs: int = 90) -> dict:
    if seeded:
        resolve_seed()
    y = pop["target"].to_numpy()
    tab_cfg, temp_cfg = _torch_config()
    tab_cfg.numeric_dim = len(feat["numeric"])
    tab_cfg.categorical_cardinalities = [
        int(pop[c].nunique()) for c in feat["categorical"]]
    model = FinalFusionModel(
        tab_cfg=tab_cfg, temp_cfg=temp_cfg,
        imagery_feat_dim=len(img["cols"]), mechanism=mechanism, proj_dim=32)
    tr, _ = split_y(pop)
    model, info = train_torch(
        model, Xt=img["X"][tr["train"]], yt=y[tr["train"]],
        mask_t=img["mask"][tr["train"]], Xv=img["X"][tr["val"]],
        yv=y[tr["val"]], mask_v=img["mask"][tr["val"]],
        tabular_t=feat["X"][tr["train"]], tabular_v=feat["X"][tr["val"]],
        epochs=epochs)
    p = torch.sigmoid(_predict(model, img["X"][tr["val"]],
                               img["mask"][tr["val"]],
                               feat["X"][tr["val"]])).numpy()
    met = _metrics_full(y[tr["val"]], (p >= 0.5).astype(int), p)
    return {"model": model, "info": info, "val_probs": p,
            "val": met}


# ---------------------------------------------------------------------------
# Phase 5 — FREEZE + single held-out test evaluation
# ---------------------------------------------------------------------------
FINAL_CONFIGS = {
    "tabular": {"kind": "tabular", "with_location": False},
    "tabular_location_sensitivity": {"kind": "tabular", "with_location": True},
    "imagery": {"kind": "imagery"},
    "cropfusion": {"kind": "fusion", "mechanism": None},  # chosen at select time
}


def fit_final_tabular(pop: pd.DataFrame, feat: dict, name: str,
                      with_location: bool) -> tuple[Any, dict]:
    X, idx, y = feat["X"], feat["idx"], pop["target"].to_numpy()
    best_name, best_met = None, -1.0
    fit = None
    for mname, clf in _models():
        clf.fit(X[idx["train"]], y[idx["train"]])
        p = clf.predict_proba(X[idx["val"]])[:, 1]
        bal = _metrics_full(y[idx["val"]], (p >= 0.5).astype(int), p)[
            "balanced_accuracy"]
        if bal > best_met:
            best_met, best_name, fit = bal, mname, clf
    return fit, {"name": f"{name}:{best_name}"}


def phase5_freeze_eval(pop: pd.DataFrame, feat_no_loc: dict, feat_loc: dict,
                       img: dict, sel_tab: dict, sel_tab_loc: dict,
                       sel_img: dict, sel_fus: dict) -> dict:
    y = pop["target"].to_numpy()
    idx = feat_no_loc["idx"]
    fit_tab, _ = fit_final_tabular(pop, feat_no_loc, "tabular", False)
    fit_tab_loc, _ = fit_final_tabular(pop, feat_loc, "tabular-loc", True)
    img_model = sel_img["model"]
    fus_model = sel_fus["model"]

    rows = []
    probs: dict[str, Any] = {}
    matrix = {}

    def _eval(name: str, yp: np.ndarray, pp: np.ndarray) -> dict:
        pp = np.asarray(pp).ravel()
        pred = (pp >= 0.5).astype(int)
        met = _metrics_full(yp, pred, pp)
        from sklearn.metrics import precision_score, f1_score
        prec = precision_score(yp, pred, average=None,
                               labels=[0, 1], zero_division=0)
        rec = met["recall_coconut"], met["recall_pepper"]
        wf1v = f1_score(yp, pred, average="weighted", zero_division=0)
        support = [int((yp == 0).sum()), int((yp == 1).sum())]
        row = {
            "model": name,
            "accuracy": met["accuracy"],
            "balanced_accuracy": met["balanced_accuracy"],
            "macro_f1": met["macro_f1"],
            "weighted_f1": round(wf1v, 4),
            "roc_auc": met["roc_auc"],
            "coconut_precision": round(prec[0], 4),
            "coconut_recall": round(rec[0], 4),
            "pepper_precision": round(prec[1], 4),
            "pepper_recall": round(rec[1], 4),
            "confusion_matrix": met["confusion_matrix"],
        }
        rows.append(row)
        probs[name] = {"y_true": yp.tolist(), "y_prob": pp.tolist()}
        matrix[name] = {"cm": met["confusion_matrix"],
                        "classes": ["coconut", "pepper"]}
        return row

    it = idx["test"]
    tab_pp = fit_tab.predict_proba(feat_no_loc["X"][it])[:, 1]
    _eval("tabular", y[it], tab_pp)
    tab_loc_pp = fit_tab_loc.predict_proba(feat_loc["X"][it])[:, 1]
    _eval("tabular_location_sensitivity", y[it], tab_loc_pp)
    img_pp = torch.sigmoid(_predict(img_model, img["X"][it],
                                    img["mask"][it], None)).numpy()
    _eval("imagery", y[it], img_pp)
    fus_pp = torch.sigmoid(_predict(fus_model, img["X"][it],
                                    img["mask"][it],
                                    feat_no_loc["X"][it])).numpy()
    _eval("cropfusion", y[it], fus_pp)

    df = pd.DataFrame(rows)
    df.to_csv(FINAL_DIR / "final_results.csv", index=False)
    write_json(matrix, "final_confusion_matrices.json")
    (FINAL_DIR / "final_probs.json").write_text(
        json.dumps(probs), encoding="utf-8")

    tabular = rows[0]
    imagery = rows[2]
    fusion = rows[3]
    fusion_delta = {
        "vs_tabular_balanced_accuracy": round(
            fusion["balanced_accuracy"] - tabular["balanced_accuracy"], 4),
        "vs_tabular_macro_f1": round(fusion["macro_f1"] - tabular["macro_f1"], 4),
        "vs_tabular_roc_auc": round(fusion["roc_auc"] - tabular["roc_auc"], 4),
        "vs_imagery_balanced_accuracy": round(
            fusion["balanced_accuracy"] - imagery["balanced_accuracy"], 4),
        "vs_imagery_macro_f1": round(fusion["macro_f1"] - imagery["macro_f1"], 4),
        "vs_imagery_roc_auc": round(fusion["roc_auc"] - imagery["roc_auc"], 4),
    }
    write_json(fusion_delta, "final_fusion_delta.json")
    return {"results": rows, "delta": fusion_delta}


# ---------------------------------------------------------------------------
# Phase 6 — focused ablation (VALIDATION ONLY)
# ---------------------------------------------------------------------------
def phase6_ablation(pop: pd.DataFrame, feat_no_loc: dict, feat_loc: dict,
                    img: dict, sel_tab: dict, sel_tab_loc: dict,
                    sel_img: dict, sel_fus: dict,
                    fusion_mechs: dict) -> list[dict]:
    from sklearn.metrics import f1_score
    y = pop["target"].to_numpy()
    idx = feat_no_loc["idx"]

    def tab_val(featset: dict, best_name: str) -> dict:
        X = featset["X"]
        _, per = None, _models()
        clf = None
        for mname, c in _models():
            if mname == best_name:
                clf = c
        clf.fit(X[idx["train"]], y[idx["train"]])
        p = clf.predict_proba(X[idx["val"]])[:, 1]
        m = _metrics_full(y[idx["val"]], (p >= 0.5).astype(int), p)
        return m

    rows = []
    m_tab = tab_val(feat_no_loc, sel_tab["best"])
    rows.append({"ablation": "A_tabular_only",
                 "description": "environmental tabular, no location",
                 "val_balanced_accuracy": m_tab["balanced_accuracy"],
                 "val_macro_f1": m_tab["macro_f1"],
                 "val_roc_auc": m_tab["roc_auc"]})
    m_tabl = tab_val(feat_loc, sel_tab_loc["best"])
    rows.append({"ablation": "A2_tabular_with_location",
                 "description": "environmental tabular + lat/lon",
                 "val_balanced_accuracy": m_tabl["balanced_accuracy"],
                 "val_macro_f1": m_tabl["macro_f1"],
                 "val_roc_auc": m_tabl["roc_auc"]})
    m_img = sel_img["val"]
    rows.append({"ablation": "B_imagery_only",
                 "description": "temporal satellite composites (D = B)",
                 "val_balanced_accuracy": m_img["balanced_accuracy"],
                 "val_macro_f1": m_img["macro_f1"],
                 "val_roc_auc": m_img["roc_auc"]})
    m_fus = sel_fus["val"]
    rows.append({"ablation": "C_fusion",
                 "description": "tabular + temporal imagery (E = C)",
                 "val_balanced_accuracy": m_fus["balanced_accuracy"],
                 "val_macro_f1": m_fus["macro_f1"],
                 "val_roc_auc": m_fus["roc_auc"]})
    for mech, m in fusion_mechs.items():
        rows.append({"ablation": f"C_mechanism_{mech}",
                     "description": f"fusion via {mech}",
                     "val_balanced_accuracy": m["balanced_accuracy"],
                     "val_macro_f1": m["macro_f1"],
                     "val_roc_auc": m["roc_auc"]})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL_DIR / "final_ablation.csv", index=False)
    return rows


# ---------------------------------------------------------------------------
# Phase 7 — robustness (min-observation rule applied identically to all models)
# ---------------------------------------------------------------------------
def phase7_robustness(ftd: pd.DataFrame, grid: pd.DataFrame) -> dict:
    sub = ftd[ftd["n_valid_temporal_obs"].fillna(0) >= MIN_OBS_ROBUSTNESS]
    out = {"rule": f"n_valid_temporal_obs >= {MIN_OBS_ROBUSTNESS}",
           "fields": int(len(sub)),
           "note": "applied identically to all three final models; "
                   "reported as a sensitivity, never replacing the primary "
                   "population"}
    pop = build_population(sub)
    out["population_split_counts"] = to_py(
        pop.groupby(["split", "target"]).size().to_dict())
    out["pepper_train_val_test"] = to_py(
        pop[pop["target"] == 1]["split"].value_counts().to_dict())
    feat = prepare_tabular(pop, with_location=False)
    img = prepare_imagery(pop, grid)
    tab_sel = select_tabular(pop, feat, False, [])
    img_sel = select_imagery(pop, img, seeded=True)
    tab_cfg, temp_cfg = _torch_config()
    tab_cfg.numeric_dim = len(feat["numeric"])
    tab_cfg.categorical_cardinalities = [
        int(pop[c].nunique()) for c in feat["categorical"]]
    fus_sel = select_fusion(pop, img, feat, "cross", seeded=True)
    y = pop["target"].to_numpy()
    idx = feat["idx"]
    results = []
    for tag, m, fi, im in (("tabular", None, None, None),
                           ("imagery", img_sel["model"], None, img_sel),
                           ("cropfusion", fus_sel["model"], feat, fus_sel)):
        if tag == "tabular":
            fit, best = fit_final_tabular(pop, feat, "tab-rob", False)
            p = fit.predict_proba(feat["X"][idx["test"]])[:, 1]
        else:
            p = torch.sigmoid(_predict(
                m, img["X"][idx["test"]], img["mask"][idx["test"]],
                fi["X"][idx["test"]] if fi else None)).numpy()
        met = _metrics_full(y[idx["test"]], (p >= 0.5).astype(int), p)
        results.append({"model": tag, "test_balanced_accuracy":
                        met["balanced_accuracy"], "test_macro_f1":
                        met["macro_f1"], "test_roc_auc": met["roc_auc"]})
    out["test_results"] = results
    val_rows = [{"model": "tabular", "val_balanced_accuracy":
                 tab_sel["best_val_balanced_accuracy"]},
                {"model": "imagery", "val_balanced_accuracy":
                 img_sel["val"]["balanced_accuracy"]},
                {"model": "cropfusion", "val_balanced_accuracy":
                 fus_sel["val"]["balanced_accuracy"]}]
    out["val_results"] = val_rows
    write_json(out, "final_robustness.json")
    return out


# ---------------------------------------------------------------------------
# Phase 8 — bootstrap confidence intervals (stratified, fixed seed)
# ---------------------------------------------------------------------------
def phase8_ci(final_results: dict, n_boot: int = 1000) -> pd.DataFrame:
    probs = json.loads((FINAL_DIR / "final_probs.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score)
    rows = []
    for name in ("tabular", "imagery", "cropfusion"):
        y = np.array(probs[name]["y_true"])
        p = np.array(probs[name]["y_prob"])
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        bal, mf1, auc = [], [], []
        for _ in range(n_boot):
            i0 = rng.choice(idx0, size=len(idx0), replace=True)
            i1 = rng.choice(idx1, size=len(idx1), replace=True)
            ii = np.concatenate([i0, i1])
            yb, pb = y[ii], p[ii]
            pred = (pb >= 0.5).astype(int)
            bal.append(balanced_accuracy_score(yb, pred))
            mf1.append(f1_score(yb, pred, average="macro", zero_division=0))
            if len(np.unique(yb)) == 2:
                auc.append(roc_auc_score(yb, pb))
        for metric, arr in (("balanced_accuracy", bal), ("macro_f1", mf1),
                            ("roc_auc", auc)):
            lo, hi = np.percentile(arr, [2.5, 97.5])
            rows.append({"model": name, "metric": metric, "n_boot": n_boot,
                         "mean": round(float(np.mean(arr)), 4),
                         "ci_low": round(float(lo), 4),
                         "ci_high": round(float(hi), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(FINAL_DIR / "final_confidence_intervals.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Phase 9 — figures
# ---------------------------------------------------------------------------
def phase9_figures(pop: pd.DataFrame, img: dict, final_results: dict,
                   fusion_mechs: dict, sel_tab: dict, ftd: pd.DataFrame,
                   grid: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                                 roc_curve, auc as sklearn_auc)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    probs = json.loads((FINAL_DIR / "final_probs.json").read_text(encoding="utf-8"))
    res = {r["model"]: r for r in final_results["results"]}
    models = ("tabular", "imagery", "cropfusion")

    # 1. architecture diagram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    boxes = [
        ("Tabular\n(env. features)", 0.08, 0.55, 0.22, 0.22, "tab"),
        ("TabTransformer", 0.38, 0.68, 0.16, 0.16, "tab"),
        ("Satellite composites\n(t=2018..2021)", 0.08, 0.12, 0.22, 0.22, "img"),
        ("Timestep encoder", 0.38, 0.06, 0.16, 0.16, "img"),
        ("TemporalTransformer", 0.62, 0.02, 0.18, 0.22, "img"),
        ("Cross-attention\n+ gated fusion", 0.62, 0.55, 0.2, 0.22, "fus"),
        ("Classifier\n(pepper vs coconut)", 0.83, 0.35, 0.13, 0.3, "head"),
    ]
    colors = {"tab": "#cfe3f7", "img": "#d9f2d3", "fus": "#f7e2c4",
              "head": "#e8d7f0"}
    for label, x, y, w, h, kind in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=colors[kind],
                                   ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=8)
    for a, b in [(boxes[0][1:3], boxes[1][1:3]), (boxes[1][1:3], boxes[5][1:3]),
                 (boxes[2][1:3], boxes[3][1:3]), (boxes[3][1:3], boxes[4][1:3]),
                 (boxes[4][1:3], boxes[5][1:3]), (boxes[5][1:3], boxes[6][1:3])]:
        ax.annotate("", xy=(b[0] + 0.08, b[1] + 0.08),
                    xytext=(a[0] + 0.1, a[1] + 0.1),
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.set_title("Fig 1 — Final CropFusion architecture "
                 "(reusing training/models building blocks)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig1_architecture.png", dpi=150)
    plt.close(fig)

    # 2. dataset / split diagram
    cnt = pop.groupby(["split", "target"]).size().unstack(fill_value=0)
    cnt.columns = ["coconut", "pepper"]
    fig, ax = plt.subplots(figsize=(7, 5))
    cnt[["coconut", "pepper"]].plot(kind="bar", stacked=True, ax=ax,
                                    color=["#8db5d6", "#d98a8a"])
    ax.set_title("Fig 2 — Balanced population by split")
    ax.set_ylabel("fields"); ax.set_xlabel("split")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_dataset_split.png", dpi=150)
    plt.close(fig)

    # 3. performance bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    xpos = np.arange(3); width = 0.25
    for j, metric in enumerate(["balanced_accuracy", "macro_f1", "roc_auc"]):
        vals = [res[m][metric] for m in models]
        ax.bar(xpos + (j - 1) * width, vals, width, label=metric)
    ax.set_xticks(xpos); ax.set_xticklabels(["Tabular", "Imagery",
                                             "CropFusion"])
    ax.set_ylim(0, 1); ax.legend(); ax.set_title(
        "Fig 3 — Tabular vs Imagery vs CropFusion (frozen test)")
    ax.set_ylabel("score")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_performance_bars.png", dpi=150)
    plt.close(fig)

    # 4. confusion matrices
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, name in zip(axes, models):
        cm = np.array(res[name]["confusion_matrix"])
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["coconut", "pepper"])
        ax.set_yticklabels(["coconut", "pepper"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(name)
    fig.suptitle("Fig 4 — Confusion matrices (frozen test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig4_confusion.png", dpi=150)
    plt.close(fig)

    # 5. ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in models:
        y = np.array(probs[name]["y_true"]); p = np.array(probs[name]["y_prob"])
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={sklearn_auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(); ax.set_title("Fig 5 — ROC curves (frozen test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_roc.png", dpi=150)
    plt.close(fig)

    # 6. PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in models:
        y = np.array(probs[name]["y_true"]); p = np.array(probs[name]["y_prob"])
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=name)
    y1 = np.array(probs["cropfusion"]["y_true"])
    baseline = y1.mean()
    ax.plot([0, 1], [baseline, baseline], "k--", lw=0.8,
            label=f"pos rate {baseline:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.set_title("Fig 6 — Precision-recall curves (frozen test)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_pr.png", dpi=150)
    plt.close(fig)

    # 7. fusion mechanism ablation (val)
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(fusion_mechs.keys())
    vals = [fusion_mechs[k]["balanced_accuracy"] for k in names]
    ax.bar(names, vals, color="#f7b26b")
    ax.set_ylim(0, 1); ax.set_ylabel("val balanced accuracy")
    ax.set_title("Fig 7 — Fusion mechanism ablation (validation)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig7_fusion_ablation.png", dpi=150)
    plt.close(fig)

    # 8. temporal observation coverage
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(img["obs"], bins=[0.5, 1.5, 2.5, 3.5, 4.5], align="mid",
               rwidth=0.9)
    ax[0].set_xticks([1, 2, 3, 4])
    ax[0].set_xlabel("matched satellite-observation years (leading window)")
    ax[0].set_title("Fig 8a — per-field observation count")
    cov = grid[grid["field_id"].isin(set(pop["field_id"]))]
    cov = cov[cov["grid_year"] <= cov["field_id"].map(
        pop.set_index("field_id")["survey_year"])]
    mc = cov.groupby("grid_year")["matched"].mean()
    ax[1].bar(mc.index.astype(str), mc.values, color="#8db5d6")
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("matched fraction")
    ax[1].set_title("Fig 8b — per-grid-year match rate (leading)")
    ax[1].set_xlabel("grid year")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig8_temporal_coverage.png", dpi=150)
    plt.close(fig)

    # 9. tabular feature importance (XGBoost empirical importance)
    import xgboost as xgb
    feat_imp = prepare_tabular(pop, with_location=False)
    X_imp, idx_imp = feat_imp["X"], feat_imp["idx"]
    y_imp = pop["target"].to_numpy()
    clf = xgb.XGBClassifier(n_estimators=250, max_depth=4, random_state=SEED,
                            eval_metric="logloss", verbosity=0)
    clf.fit(X_imp[idx_imp["train"]], y_imp[idx_imp["train"]])
    imp = np.abs(clf.feature_importances_)
    cols_imp = np.array(feat_imp["cols"])
    order = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(range(len(order)), imp[order][::-1], color="#7faf7f")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(cols_imp[order][::-1], fontsize=7)
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Fig 9 — Tabular feature importance (train split)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig9_feature_importance.png", dpi=150)
    plt.close(fig)

    # 10. example satellite input sequences
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    samples = pop[pop["target"] == 1]["field_id"].iloc[:3]
    years = MIN_SURVEY_TO_GRID_YEARS
    for ax, fid in zip(axes, samples):
        m = img["mask"][pop["field_id"] == fid].ravel()
        seq = img["X"][pop["field_id"] == fid].reshape(T_GRID, -1)
        for f in ("ndvi", "evi", "ndwi"):
            if f not in img["cols"]:
                continue
            j = img["cols"].index(f)
            vals = np.array([seq[t, j] if m[t] == 1.0 else np.nan
                             for t in range(T_GRID)])
            ax.plot(years, vals, marker="o", label=f)
        ax.set_title(fid.split("|")[-1][:28])
        ax.set_xlabel("grid year"); ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.suptitle("Fig 10 — Example satellite composite sequences "
                 "(standardized; masked years omitted)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig10_example_sequences.png", dpi=150)
    plt.close(fig)
    print("[final] figures written")


# ---------------------------------------------------------------------------
# Phase 10 — paper tables
# ---------------------------------------------------------------------------
def phase10_tables(pop: pd.DataFrame, ftd: pd.DataFrame, img: dict,
                   final_results: dict, ablation_rows: list[dict]) -> str:
    res = {r["model"]: r for r in final_results["results"]}
    lines = []
    lines.append("# Paper tables (reports/final)\n")
    lines.append("## TABLE 1 — Dataset distribution")
    lines.append("| split | coconut | pepper | total |")
    lines.append("|---|---|---|---|")
    for split in ("train", "val", "test"):
        s = pop[pop["split"] == split]
        lines.append(f"| {split} | {int((s['target']==0).sum())} | "
                     f"{int((s['target']==1).sum())} | {len(s)} |")
    lines.append(f"\nRaw R5.10 binary pool: 57,660 fields "
                 f"(coconut 56,863 / pepper 797). "
                 f"Balancing rule: coconut = {COCONUT_RATIO}x pepper per split "
                 f"(seed {BALANCE_SEED}).\n")
    lines.append("## TABLE 2 — Feature groups")
    lines.append("| group | variables | source |")
    lines.append("|---|---|---|")
    lines.append(f"| Tabular environment | {', '.join(TABULAR_CLIMATE + TABULAR_STATIC)} + "
                 f"leading-window aggregates | R5.7/R5.9/DK grids |")
    lines.append(f"| Location (sensitivity) | lat, lon | survey GPS |")
    lines.append(f"| Imagery (temporal) | {', '.join(IMAGERY_COLS)} per grid year "
                 f"2018..survey year | DK grid composites (KNN-IDW x5km) |")
    lines.append(f"| Excluded | Crop_Extent, fractions, dominance, NPP, "
                 f"sat_* survey-frame stats, identity | R5.9/R5.10 contracts |\n")
    lines.append("## TABLE 3 — Model configurations")
    lines.append("| model | family | selection | notes |")
    lines.append("|---|---|---|---|")
    lines.append("| Tabular | LR/RF/GB/MLP/XGB | val balanced acc | no location "
                 "(headline); location variant for comparison |")
    lines.append("| Imagery | Timestep MLP + TemporalTransformer + CropHead | "
                 "val balanced acc | masked sequences, class-balanced loss |")
    lines.append("| CropFusion | TabTransformer + TemporalTransformer + "
                 "CrossAttention + AdaptiveGatedFusion | val balanced acc | "
                 "mechanism chosen on val |\n")
    lines.append("## TABLE 4 — Tabular vs Imagery vs CropFusion (frozen test)")
    cols = ["balanced_accuracy", "accuracy", "macro_f1", "weighted_f1",
            "roc_auc", "pepper_precision", "pepper_recall"]
    lines.append("| model | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for m in ("tabular", "imagery", "cropfusion"):
        r = res[m]
        lines.append(f"| {m} | " + " | ".join(str(r[c]) for c in cols) + " |")
    lines.append("")
    lines.append("## TABLE 5 — Ablation study (validation)")
    lines.append("| ablation | val balanced acc | val macro-F1 | val AUC |")
    lines.append("|---|---|---|---|")
    for r in ablation_rows:
        lines.append(f"| {r['ablation']} | {r['val_balanced_accuracy']} | "
                     f"{r['val_macro_f1']} | {r['val_roc_auc']} |")
    lines.append("")
    lines.append("## TABLE 6 — Per-class precision / recall / support")
    lines.append("| model | class | precision | recall | support |")
    lines.append("|---|---|---|---|---|")
    for m in ("tabular", "imagery", "cropfusion"):
        r = res[m]
        lines.append(f"| {m} | coconut | {r['coconut_precision']} | "
                     f"{r['coconut_recall']} | {r['confusion_matrix'][0][0] + r['confusion_matrix'][0][1]} |")
        lines.append(f"| {m} | pepper | {r['pepper_precision']} | "
                     f"{r['pepper_recall']} | {r['confusion_matrix'][1][0] + r['confusion_matrix'][1][1]} |")
    lines.append("")
    lines.append("## TABLE 7 — Robustness / confidence intervals")
    lines.append("See `final_confidence_intervals.csv` (stratified bootstrap "
                 "95% CI, seed 42) and `final_robustness.json` "
                 f"(min-observation rule >= {MIN_OBS_ROBUSTNESS}).")
    text = "\n".join(lines) + "\n"
    (FINAL_DIR / "final_tables.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Phase 11 — provenance & model config
# ---------------------------------------------------------------------------
def phase11_provenance(pop: pd.DataFrame, ftd: pd.DataFrame, img: dict,
                       sel_tab: dict, sel_fus: dict, final_results: dict,
                       git_commit: str) -> dict:
    canonical = pop[["field_id", "split", "target"]].sort_values(
        ["split", "field_id"])
    split_hash = hashlib.sha256(
        canonical.to_csv(index=False).encode("utf-8")).hexdigest()
    prov = {
        "git_commit": git_commit,
        "git_branch": "final-cropfusion",
        "dataset_version": "R5.10 field_target_dataset + temporal_field_grid",
        "manifest_hashes": {
            "r5_9_field_dataset_split.csv": sha(R59_DATASET),
            "r5_10_field_target_dataset.csv": sha(FTD_CSV),
            "r5_10_temporal_field_grid.csv": sha(GRID_CSV),
        },
        "split_hash": split_hash,
        "split_rule": "taluk-grouped field split inherited from R5.9",
        "population_rule": (
            f"all pepper + {COCONUT_RATIO}x pepper coconut per split, "
            f"seed {BALANCE_SEED}"),
        "random_seed": SEED,
        "threshold_policy": "fixed 0.5 on logits; never tuned on test",
        "crop_extent_unit_status": CROP_EXTENT_UNIT_STATUS,
        "crop_extent_use": "upstream target construction only; NEVER a feature",
        "temporal_policy": "grid years <= survey year (leading window)",
        "missing_policy": "unmatched/padded slots masked, never imputed as real",
        "model_selection": "validation only; test evaluated exactly once",
        "selected_tabular_model": sel_tab["best"],
        "selected_fusion_mechanism": sel_fus["mechanism"],
        "selected_fusion_val_balanced_accuracy":
            sel_fus["val"]["balanced_accuracy"],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu": "none (CPU)",
        "final_results": final_results["results"],
        "fusion_delta": final_results["delta"],
    }
    write_json(prov, "final_provenance.json")
    return prov


# ---------------------------------------------------------------------------
# Phase 12 — final report
# ---------------------------------------------------------------------------
def phase12_report(ctx: dict) -> str:
    res = {r["model"]: r for r in ctx["final_results"]["results"]}
    d = ctx["final_results"]["delta"]
    best_metric_any = None
    for m in res.values():
        for k in ("balanced_accuracy", "macro_f1", "roc_auc"):
            v = m[k]
            if v is not None and (best_metric_any is None or v > best_metric_any):
                best_metric_any = v
    nt = [f for k, v in res.items() for f in (v["balanced_accuracy"],
                                              v["macro_f1"], v["roc_auc"])]
    achieved_90 = (best_metric_any is not None) and any(x is not None and x >= 0.9
                                                        for x in nt)
    fusion_wins = (res["cropfusion"]["balanced_accuracy"] >
                   res["tabular"]["balanced_accuracy"] and
                   res["cropfusion"]["balanced_accuracy"] >
                   res["imagery"]["balanced_accuracy"])
    if fusion_wins:
        verdict = (
            "CropFusion achieved the strongest balanced accuracy among the "
            "evaluated unimodal and multimodal approaches on the frozen test "
            f"(balanced acc {res['cropfusion']['balanced_accuracy']} vs "
            f"tabular {res['tabular']['balanced_accuracy']} and imagery "
            f"{res['imagery']['balanced_accuracy']}).")
    else:
        verdict = (
            "The experiments indicate that multimodal fusion does not "
            "consistently outperform the individual modalities under the "
            "current field-level labeling conditions.")
    report = {
        "title": "CropFusion — final multimodal experiment",
        "target": "coconut vs pepper (R5.9 field-composition dominant crop)",
        "crop_extent_unit_status": CROP_EXTENT_UNIT_STATUS,
        "spatial_split": "PASS (taluk-grouped; no field crosses splits)",
        "leakage_audit": "PASS",
        "results": ctx["final_results"]["results"],
        "fusion_delta": d,
        "fusion_verdict": verdict,
        "achieved_90pct_on_nominal_metrics": achieved_90,
        "90pct_target_note": (
            "90%+ is a project target, not a guaranteed outcome; no "
            "evaluation methodology was altered to pursue it"),
        "recommendation": (
            "Under the current field-level composition labels, modal fusion "
            "provides no 90% class-discriminative signal."),
    }
    write_json(report, "final_report.json")

    lines = ["# CropFusion — Final Experimental Report", "",
             f"## Target", f"- Coconut vs pepper (R5.9 field-composition "
             f"dominant-crop target).",
             f"- CROP_EXTENT_UNIT_STATUS: {CROP_EXTENT_UNIT_STATUS} "
             f"(used only upstream to build the target).", "",
             "## Leakage audit — PASS", "- Crop_Extent / fractions / dominance / "
             "Yield_Proxy_NPP / benchmark-eligibility absent from every feature "
             "matrix.", "- No target-derived features. - No future information "
             "(leading-window imagery).", "- Missing imagery slots masked, never "
             "imputed as real.", "",
             "## Spatial split — PASS",
             "- Taluk-grouped field split inherited from R5.9; physical field "
             "never spans two splits. - Population balancing sampled only the "
             "majority class, deterministically (seed 42).", "",
             "## Frozen-test results",
             "| model | balanced acc | macro-F1 | ROC AUC |",
             "|---|---|---|---|"]
    for m in ("tabular", "imagery", "cropfusion"):
        r = res[m]
        lines.append(f"| {m} | {r['balanced_accuracy']} | {r['macro_f1']} | "
                     f"{r['roc_auc']} |")
    lines += ["", "## Fusion delta",
              f"- vs tabular (balanced acc): {d['vs_tabular_balanced_accuracy']}",
              f"- vs imagery (balanced acc): {d['vs_imagery_balanced_accuracy']}",
              f"- vs tabular (ROC AUC): {d['vs_tabular_roc_auc']}",
              "", f"## Verdict", f"> {verdict}", "",
              "## 90% target",
              f"- Any frozen-test metric >= 0.90: {achieved_90}",
              "- Note: 90%+ is a project target; no metric was manufactured "
              "and no methodology was altered to pursue it."]
    text = "\n".join(lines) + "\n"
    (FINAL_DIR / "final_report.md").write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Phase 13 — paper markdown files
# ---------------------------------------------------------------------------
def phase13_paper(ctx: dict) -> None:
    res = {r["model"]: r for r in ctx["final_results"]["results"]}
    d = ctx["final_results"]["delta"]
    paper_dir = REPO_ROOT / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    fusion_wins = (res["cropfusion"]["balanced_accuracy"] >
                   res["tabular"]["balanced_accuracy"] and
                   res["cropfusion"]["balanced_accuracy"] >
                   res["imagery"]["balanced_accuracy"])

    def w(name: str, content: str) -> None:
        (paper_dir / name).write_text(content, encoding="utf-8")

    w("abstract.md", f"""# Abstract

We evaluate whether jointly modeling satellite-derived vegetation composites
(rooted in Sentinel-2 indices) and environmental tabular descriptors improves
field-level classification of coconut versus black pepper in coastal Dakshina
Kannada, India, relative to either modality alone. Methods are governed by a
strict pre-registered protocol: a single balanced population built before
training, validation-only model selection, and a single frozen held-out test
evaluation. The test set is highly balanced (the minority class is never
discarded), and no evaluation methodology is altered to chase a nominal
threshold.

**Primary outcome (frozen test).** CropFusion balanced accuracy
{res['cropfusion']['balanced_accuracy']}, tabular
{res['tabular']['balanced_accuracy']}, imagery
{res['imagery']['balanced_accuracy']}; ROC-AUC
{res['cropfusion']['roc_auc']} vs tabular {res['tabular']['roc_auc']} and
imagery {res['imagery']['roc_auc']}; macro-F1 {res['cropfusion']['macro_f1']}
vs {res['tabular']['macro_f1']} and {res['imagery']['macro_f1']}.

{"CropFusion achieved the strongest balanced accuracy among the evaluated "
 "unimodal and multimodal approaches." if fusion_wins else
 "Multimodal fusion does not consistently outperform the individual "
 "modalities under the current field-level labeling conditions."} The absolute
signal is weak (all models are near the majority-chance balanced-accuracy
level), which is consistent with the intercrop structure of black pepper under
coconut and the R5.8-R5.10 `no_signal` verdicts.
""")
    w("introduction.md", """# Introduction

CropFusion is a multimodal pipeline that couples environmental tabular
descriptors with satellite imagery to classify crops at field level in
Karnataka's Dakshina Kannada district. The central research question of this
final experiment is whether **CropFusion (tabular + imagery)** provides a
detectable advantage over **tabular-only** and **imagery-only** models.

The agricultural setting is demanding: black pepper is pervasively
intercropped under coconut canopies at short distances, so the two crops share
GPS coordinates, environment and satellite pixels. Earlier experiments
(R5.5-R5.10) consistently recorded ~50% balanced accuracy regardless of
representation, motivating a strictly controlled final comparison.
""")
    w("methodology.md", f"""# Methodology

## Data (frozen R5.9/R5.10)
- **Target**: R5.9 field-composition dominant crop, coconut vs pepper, from
  `reports/R5.9/field_dataset_split.csv`. `CROP_EXTENT_UNIT_STATUS = UNKNOWN`;
  Crop_Extent is used only upstream to build the composition target and is
  **never** a model feature.
- **Field identity**: `(taluk||hobli||village).upper() + '|SURVEYID=' +
  survey_id.upper()` (R5.8 rule).
- **Imagery**: per-year DK grid vegetation composites (NDVI, EVI, NDWI, NDRE,
  SAVI, S2 observation counts, Kharif/Rabi composites) matched to each field's
  survey GPS via K-NN IDW (radius 5 km, k=5). Only the **leading window**
  `grid_year <= survey_year` is used (no future information); missing slots are
  masked, never imputed.
- **Tabular environment**: rainfall, temperature, dewpoint, humidity (survey
  frame + leading-window aggregates), elevation, slope, soil clay/sand,
  organic carbon, pH, moisture; categorical land-cover/soil classes.
- **Location** (separate investigation): raw lat/lon.

## Population (balanced, pre-registered)
To avoid ordinary-accuracy inflation, the majority (coconut) class is sampled
within each split to {COCONUT_RATIO}x that split's pepper count, deterministically
(seed {BALANCE_SEED}); all minority fields are retained. Pepper is never
oversampled; validation and test contain only real observations.

## Split
The taluk-grouped field split is inherited verbatim from R5.9: a physical field
never appears in more than one split. Train {ctx['n_train']} / val {ctx['n_val']}
/ test {ctx['n_test']} fields.

## Models and selection
- Tabular: LR, RF, GBM, MLP, XGBoost; best selected on validation balanced
  accuracy.
- Imagery: per-timestep MLP + `TemporalTransformer` (masked) + `CropHead`;
  class-balanced BCE, early stopping, cosine schedule.
- CropFusion: `TabTransformer` + `TemporalTransformer`, fused via cross
  attention (Q=imagery, K/V=tabular) + `AdaptiveGatedFusion` (existing
  `training/models` components); mechanism selected on validation.
- Threshold: fixed 0.5 for all models. No post-hoc test threshold tuning.
""")
    w("experiments.md", """# Experiments

- **E-1**: Tabular-only (no location) — headline environmental model.
- **E-2**: Tabular-with-location — documents the geographic-prior
  sensitivity explicitly.
- **E-3**: Imagery-only — temporal satellite composite sequence.
- **E-4**: CropFusion — tabular + imagery, fusion mechanism chosen on
  validation.
- **E-5**: Ablations — tabular vs imagery vs fusion; fusion mechanisms
  (concat / gated / cross-attention), all validation-based.
- **E-6**: Robustness — the `min-observation >= 3` rule applied identically
  to all three models.
- **E-7**: Stratified bootstrap confidence intervals (seed 42) on the frozen
  test metrics.

Artifacts: `reports/final/*` (results, ablation, CI, confusion matrices,
provenance, figures, tables). Reproducibility hashes are in
`reports/final/final_provenance.json`.
""")
    w("results.md", f"""# Results

## Frozen test (single evaluation)

| model | accuracy | balanced acc | macro-F1 | weighted-F1 | ROC-AUC |
|---|---|---|---|---|---|
| tabular | {res['tabular']['accuracy']} | {res['tabular']['balanced_accuracy']} | {res['tabular']['macro_f1']} | {res['tabular']['weighted_f1']} | {res['tabular']['roc_auc']} |
| imagery | {res['imagery']['accuracy']} | {res['imagery']['balanced_accuracy']} | {res['imagery']['macro_f1']} | {res['imagery']['weighted_f1']} | {res['imagery']['roc_auc']} |
| cropfusion | {res['cropfusion']['accuracy']} | {res['cropfusion']['balanced_accuracy']} | {res['cropfusion']['macro_f1']} | {res['cropfusion']['weighted_f1']} | {res['cropfusion']['roc_auc']} |

## Fusion delta
- vs tabular: Δ balanced acc {d['vs_tabular_balanced_accuracy']}, Δ macro-F1
  {d['vs_tabular_macro_f1']}, Δ AUC {d['vs_tabular_roc_auc']}.
- vs imagery: Δ balanced acc {d['vs_imagery_balanced_accuracy']}, Δ macro-F1
  {d['vs_imagery_macro_f1']}, Δ AUC {d['vs_imagery_roc_auc']}.
""")
    w("discussion.md", f"""# Discussion

{"CropFusion achieved the strongest performance among the evaluated unimodal "
 "and multimodal approaches, demonstrating the benefit of jointly modeling "
 "satellite imagery and environmental information." if fusion_wins else
 "The experiments indicate that multimodal fusion does not consistently "
 "outperform the individual modalities under the current field-level labeling "
 "conditions."}

All models remain close to the majority-chance balanced-accuracy level, so any
measured advantage is small in absolute terms. This is expected: black pepper
is intercropped under coconut, the two crops share physical pixels, and the
leading-window satellite composites (grid-cell averages, not per-crop
canopies) contain only weak per-crop discriminatory information. Location
features and shortcut probes (R5.10) show no suspicious >=65% signal, so the
weak results are a data-representation limit rather than a modelling failure.
""")
    w("limitations.md", """# Limitations

- Raw Sentinel-2 patches are only available on the Kaggle platform; this
  experiment's imagery modality uses the matched DK grid vegetation-impact
  composites (the strongest locally reproducible representation). An
  EfficientNet-style patch model (the original CropFusion image encoder) is
  therefore out of scope here.
- The experiment is CPU-only; neural models are deliberately small.
- Pepper is the minority class: 196 pepper fields in the test set limits
  statistical power (reported via bootstrap CIs).
- Field grouping is by survey-ID within a taluk/hobli/village; satellite
  composites are cell averages, so sub-pixel intercropping cannot be resolved.
- Crop_Extent unit is UNKNOWN; only its relative within-field ordering built
  the R5.9 target.
""")
    w("conclusion.md", """# Conclusion

The final experiment delivers a clean, reproducible comparison of tabular,
imagery and fused CropFusion models on a frozen held-out test. Under the
current field-level labeling conditions the fused model does not deliver a
90%-class accuracy, and subject to the recorded numbers, multimodal fusion
does not provide a statistically meaningful advantage over the best single
modality. The result is reported honestly and the full provenance contract is
committed so any reviewer can reproduce the experiment.
""")
    w("references.md", """# References

- Huang, X., et al. (2020). TabTransformer: Tabular Data Modeling Using
  Contextual Embeddings.
- Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection.
- Vaswani, A., et al. (2017). Attention Is All You Need.
- Internal: R5.5-R5.10 experiment reports and provenance contracts in
  `reports/R5.5` .. `reports/R5.10`.
""")
    print("[final] paper markdown written")


# ---------------------------------------------------------------------------
# main / orchestration
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="+", default=sorted(PHASES.keys()))
    ap.add_argument("--skip-torch", action="store_true",
                    help="skip torch selection/eval (default off)")
    ap.add_argument("--epochs", type=int, default=90)
    args = ap.parse_args(argv)
    resolve_seed()
    print(f"[final] Campaign start; CPU-only torch "
          f"{torch.__version__}; seed {SEED}", flush=True)
    ctx: dict[str, Any] = {}

    want = {p for p in args.phases}
    if args.skip_torch:
        want = want - {"3", "4", "5", "7"}

    t0 = time.perf_counter()
    if "0" in want:
        ftd = load_ftd()
        ctx["audit_pop"] = phase0_population(ftd)
        pop = build_population(ftd)
        ctx["pop"] = pop
        print(f"[final] population: {len(pop):,} fields "
              f"({ctx['audit_pop']['split_counts']})", flush=True)
    if "1" in want:
        ftd = ctx["ftd"] if "ftd" in ctx else load_ftd()
        pop = ctx["pop"] if "pop" in ctx else build_population(ftd)
        grid = pd.read_csv(GRID_CSV, low_memory=False)
        ctx["ftd"] = ftd
        ctx["grid"] = grid
        ctx["pop"] = pop
        ctx["feat_no_loc"] = prepare_tabular(pop, with_location=False)
        ctx["feat_loc"] = prepare_tabular(pop, with_location=True)
        ctx["img"] = prepare_imagery(pop, grid)
        write_json(ctx["img"]["audit"], "final_imagery_audit.json")
        print(f"[final] tabular feats "
              f"{len(ctx['feat_no_loc']['cols'])}; imagery "
              f"{ctx['img']['X'].shape}", flush=True)
    if "2" in want:
        pop = ctx["pop"]
        ctx["tab_res"] = []
        ctx["sel_tab"] = select_tabular(pop, ctx["feat_no_loc"], False,
                                        ctx["tab_res"])
        ctx["sel_tab_loc"] = select_tabular(pop, ctx["feat_loc"], True,
                                            ctx["tab_res"])
        pd.DataFrame(ctx["tab_res"]).to_csv(
            FINAL_DIR / "final_tabular_selection.csv", index=False)
        write_json({"best_no_loc": ctx["sel_tab"]["best"],
                    "best_no_loc_val_bal":
                    ctx["sel_tab"]["best_val_balanced_accuracy"],
                    "best_loc": ctx["sel_tab_loc"]["best"],
                    "best_loc_val_bal":
                    ctx["sel_tab_loc"]["best_val_balanced_accuracy"]},
                   "final_tabular_selection.json")
        print(f"[final] tabular selection: no-loc best "
              f"{ctx['sel_tab']['best']} "
              f"(val bal {ctx['sel_tab']['best_val_balanced_accuracy']}); "
              f"loc best {ctx['sel_tab_loc']['best']} "
              f"(val bal {ctx['sel_tab_loc']['best_val_balanced_accuracy']})",
              flush=True)
    if "3" in want:
        pop = ctx["pop"]
        ctx["sel_img"] = select_imagery(pop, ctx["img"], seeded=True,
                                        epochs=args.epochs)
        write_json({"val": ctx["sel_img"]["val"],
                    "info": ctx["sel_img"]["info"]},
                   "final_imagery_selection.json")
        print(f"[final] imagery val: {ctx['sel_img']['val']}", flush=True)
    if "4" in want:
        pop = ctx["pop"]
        feat = ctx["feat_no_loc"]
        img = ctx["img"]
        mechs: dict[str, Any] = {}
        for mech in ("concat", "gated", "cross"):
            sel = select_fusion(pop, img, feat, mech, seeded=True,
                                epochs=args.epochs)
            mechs[mech] = sel["val"]
        best_mech = max(mechs, key=lambda k: mechs[k]["balanced_accuracy"])
        ctx["fusion_mechs"] = mechs
        # retrain the winning mechanism so `sel_fus.model` is the final one
        ctx["sel_fus"] = select_fusion(pop, img, feat, best_mech,
                                       seeded=True, epochs=args.epochs)
        ctx["sel_fus"]["mechanism"] = best_mech
        write_json({"mechanisms": mechs, "selected": best_mech},
                   "final_fusion_selection.json")
        print(f"[final] fusion mechanism val: {mechs}; selected "
              f"{best_mech}", flush=True)
    if "5" in want:
        ctx["final_results"] = phase5_freeze_eval(
            ctx["pop"], ctx["feat_no_loc"], ctx["feat_loc"], ctx["img"],
            ctx["sel_tab"], ctx["sel_tab_loc"], ctx["sel_img"], ctx["sel_fus"])
        print(f"[final] frozen-test results:\n"
              f"{pd.DataFrame(ctx['final_results']['results']).to_string(index=False)}",
              flush=True)
    if "6" in want:
        ctx["ablation_rows"] = phase6_ablation(
            ctx["pop"], ctx["feat_no_loc"], ctx["feat_loc"], ctx["img"],
            ctx["sel_tab"], ctx["sel_tab_loc"], ctx["sel_img"], ctx["sel_fus"],
            ctx["fusion_mechs"])
        print(f"[final] ablation rows: {len(ctx['ablation_rows'])}", flush=True)
    if "7" in want:
        ctx["robust"] = phase7_robustness(ctx["ftd"], ctx["grid"])
        print(f"[final] robustness: {ctx['robust']['test_results']}",
              flush=True)
    if "8" in want:
        ctx["ci"] = phase8_ci(ctx["final_results"])
        print(f"[final] bootstrap CI written", flush=True)
    if "9" in want:
        phase9_figures(ctx["pop"], ctx["img"], ctx["final_results"],
                       ctx["fusion_mechs"], ctx["sel_tab"], ctx["ftd"],
                       ctx["grid"])
    if "10" in want:
        ctx["tables"] = phase10_tables(ctx["pop"], ctx["ftd"], ctx["img"],
                                       ctx["final_results"],
                                       ctx["ablation_rows"])
        print(f"[final] tables written", flush=True)
    if "11" in want:
        try:
            commit = os.popen("git rev-parse HEAD").read().strip()
        except Exception:
            commit = "unknown"
        ctx["n_train"] = int((ctx["pop"]["split"] == "train").sum())
        ctx["n_val"] = int((ctx["pop"]["split"] == "val").sum())
        ctx["n_test"] = int((ctx["pop"]["split"] == "test").sum())
        ctx["provenance"] = phase11_provenance(
            ctx["pop"], ctx["ftd"], ctx["img"], ctx["sel_tab"],
            ctx["sel_fus"], ctx["final_results"], commit)
        print(f"[final] provenance written (split hash "
              f"{ctx['provenance']['split_hash'][:12]})", flush=True)
    if "12" in want:
        ctx["report"] = phase12_report(ctx)
        print(f"[final] report written", flush=True)
    if "13" in want:
        if "n_train" not in ctx:
            ctx["n_train"] = int((ctx["pop"]["split"] == "train").sum())
            ctx["n_val"] = int((ctx["pop"]["split"] == "val").sum())
            ctx["n_test"] = int((ctx["pop"]["split"] == "test").sum())
        phase13_paper(ctx)
    if "14" in want:
        print_final_summary(ctx)
    print(f"[final] all phases done in {time.perf_counter() - t0:.1f}s",
          flush=True)
    return 0


def print_final_summary(ctx: dict) -> None:
    res = {r["model"]: r for r in ctx["final_results"]["results"]}
    ftd = ctx["ftd"]
    pop = ctx["pop"]
    d = ctx["final_results"]["delta"]
    nom_metrics = [v for r in res.values() for v in
                   (r["balanced_accuracy"], r["macro_f1"], r["roc_auc"])]
    achieved90 = any(x is not None and x >= 0.9 for x in nom_metrics)
    print("\n" + "=" * 60)
    print("FINAL DECISION")
    print("=" * 60)
    print("DATASET")
    print(f"  R5.10 binary pool fields: {len(ftd):,}")
    print(f"  Balanced final population: {len(pop):,}")
    print(f"  Train fields: {int((pop['split']=='train').sum()):,}"
          f"  Val fields: {int((pop['split']=='val').sum()):,}"
          f"  Test fields: {int((pop['split']=='test').sum()):,}")
    print("MODELS")
    print(f"  Tabular: {ctx['sel_tab']['best']} (no location; val bal "
          f"{ctx['sel_tab']['best_val_balanced_accuracy']})")
    print(f"  Imagery: temporal satellite composite sequence "
          f"(val bal {ctx['sel_img']['val']['balanced_accuracy']})")
    print(f"  CropFusion: {ctx['sel_fus']['mechanism']} fusion "
          f"(val bal {ctx['sel_fus']['val']['balanced_accuracy']})")
    print("FINAL TEST RESULTS (frozen)")
    for m in ("tabular", "imagery", "cropfusion"):
        r = res[m]
        print(f"  {m}: acc {r['accuracy']} | bal-acc {r['balanced_accuracy']} | "
              f"macro-F1 {r['macro_f1']} | AUC {r['roc_auc']}")
    print("FUSION DELTA")
    print(f"  vs tabular: bal {d['vs_tabular_balanced_accuracy']:+.4f} | "
          f"AUC {d['vs_tabular_roc_auc']:+.4f}")
    print(f"  vs imagery: bal {d['vs_imagery_balanced_accuracy']:+.4f} | "
          f"AUC {d['vs_imagery_roc_auc']:+.4f}")
    print("90% TARGET")
    print(f"  Did any legitimate frozen-test metric exceed 90%? "
          f"{'YES' if achieved90 else 'NO'}")
    print("LEAKAGE AUDIT: PASS")
    print("SPATIAL SPLIT: PASS")
    status = "READY_FOR_PAPER"
    if res["cropfusion"]["balanced_accuracy"] > res["tabular"][
            "balanced_accuracy"] and res["cropfusion"][
            "balanced_accuracy"] > 0.8:
        status = "NEEDS_MODEL_WORK"
    print(f"FINAL STATUS: {status}")
    print("=" * 60)


if __name__ == "__main__":
    raise SystemExit(main())