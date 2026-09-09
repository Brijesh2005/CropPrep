# Yield Target Analysis

> **Generated:** 2026-09-09 (Phase 2)
> **Status:** COMPLETE — recommendation = **UNRESOLVED** (no defensible numeric yield target exists in approved sources).

---

## 1. Objective

Decide the **supervised yield regression target** for the fusion model, using **approved sources only** (Phase-2 rule). The decision must be based on actual data inspection, not aspiration.

## 2. Strict rules (Phase-2 constraints, carried from Phase 1)

1. **NOT `Yield_Proxy_NPP`** as target (NPP-derived proxy; excluded).
2. No invented/unit-converted yield field.
3. No selecting a candidate *because it makes a better-looking result*.
4. If no defensible actual or derived yield quantity exists in approved sources → mark **UNRESOLVED**, do NOT fabricate a `yield` column in `data/master.csv`.

## 3. Candidate examination (all candidates found in actual files)

| # | Candidate | Source | What it actually is | Usable as yield target? |
|---|---|---|---|---|
| A | `Crop_Extent` | OGD Crop Survey (`ogd_unified_all_hoblis.csv`, `ogd_putturu_kharif_2020_21.csv`) | Parcel **area** string in acres-ares-sq.m format, e.g. `0-34-0.00` = 0 ac 34 ares 0 m². It is a survey-time extent/area classification, NOT a harvested quantity. | **NO.** It is dimensionally area, not yield (mass/area or volume/area). Treating it as yield is a fabricated unit change (rule 2/3). |
| B | `Yield_Proxy_NPP` | `DK_Features_*.csv` | District-level NPP composite (`Annual + Kharif/Rabi composites`), value ~0.5976 for DK 2020. District-wide, no per-observation meaning; NPP is biomass, not farm yield. | **NO.** Explicitly excluded as a supervised target (rule 1). At most a coarse feature candidate. |
| C | No numeric yield column | — | Neither OGD nor DK_Features contains `production`, `yield_amount`, `yield_kg`, or any per-parcel harvest quantity. The frozen corpus (`crop_supervised_v2.csv`) has NO yield column either. | This is the actual state. |

**Verified absence:** searching every approved CSV column list found **no** production/yield column. OGD carries only `Crop_Extent` (area). DK_Features carries only `Yield_Proxy_NPP` (excluded). Sentinel-2 supplies imagery only.

## 4. Final recommendation

**UNRESOLVED — do not include a numeric yield target in the master dataset.**

- `data/master.csv` keeps a `yield_target` column that is **empty** for every row; it exists only to make the absence explicit and to receive a value if a defensible target is later established.
- The master dataset is and remains a **crop-classification** dataset (coconut/pepper/coffee/cardamom). Fusion evaluation in the next phase should treat **crop classification** as the supervised task; yield regression must be explicitly re-scoped as future work pending a real target source.
- Re-opening this decision requires ONE of:
  1. An approved per-observation yield/production dataset for Dakshina Kannada (survey, FPO, or government series), or
  2. An explicitly-approved, documented model-based yield inference whose target derivation is agreed and reproducible — not silently invented.

## 5. Leakage note

Had we used `Crop_Extent` (area) as a "yield," it would double-count the survey-time field-level signal that is already present as a covariate and would not measure crop productivity. NPP-based proxies at district granularity would be trivially leaky across the spatial split (whole DK district on one side of the split). Both are avoided by keeping the target unresolved.