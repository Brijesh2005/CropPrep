"""
R5.2.5 Phase 2: Steps 4-8 — Label validation, distributions, spatial/temporal quality, multimodal.
"""
import sys
sys.path.insert(0, "D:/CropPrep")

import json, os, csv, math, time
from collections import Counter, defaultdict
from datetime import datetime
from shared.enums import CropType

OUTPUT_DIR = "D:/CropPrep/govt_crop_matched_v1"

# Load full match
full_match = []
with open(os.path.join(OUTPUT_DIR, "government_crop_stam_match.csv"), "r", encoding="utf-8") as f:
    full_match = list(csv.DictReader(f))

# Load valid samples
valid = [r for r in full_match if r["valid_cropfusion_sample"] == "True"]

# Load data_season for existing corpus
existing = list(csv.DictReader(open("D:/CropPrep/Tabular_Datasets/data_season.csv", "r", encoding="utf-8")))

# ============================================================
# STEP 4: VALIDATE CROP LABELS
# ============================================================
print("=" * 70)
print("STEP 4: VALIDATE CROP LABELS")
print("=" * 70)

valid_crops = {ct.value for ct in CropType}
# Our 5 target classes
target_classes = {"coconut", "pepper", "coffee", "cardamom", "blackgram"}

label_issues = []
for i, r in enumerate(valid):
    ct = r["crop_type"]
    if ct not in valid_crops:
        label_issues.append({"idx": i, "crop_type": ct, "issue": "NOT_IN_ENUM"})
    elif ct not in target_classes:
        label_issues.append({"idx": i, "crop_type": ct, "issue": "NOT_IN_TARGET"})
    
    # Check for -1 or unknown
    if ct in ("-1", "unknown", "none", ""):
        label_issues.append({"idx": i, "crop_type": ct, "issue": "INVALID_LABEL"})
    
    # Check source crop consistency
    src = r["source_crop"].lower()
    expected_map = {
        "coconut": ["coconut"],
        "pepper": ["pepper (black)", "pepper"],
        "coffee": ["coffee robusta", "coffee arabica"],
        "cardamom": ["cardamom"],
        "blackgram": ["urad"],
    }
    expected = expected_map.get(ct, [])
    if expected and src not in expected:
        label_issues.append({"idx": i, "source_crop": r["source_crop"], "crop_type": ct,
                            "issue": "SOURCE_CROP_MISMATCH"})

print(f"  Valid samples: {len(valid):,}")
print(f"  Label issues: {len(label_issues)}")
if label_issues:
    issue_types = Counter(i["issue"] for i in label_issues)
    print(f"  Issue types: {dict(issue_types)}")
    for issue in label_issues[:10]:
        print(f"    {issue}")

# Verify class ID stability
print("\n  CropType enum check:")
for ct in CropType:
    print(f"    {ct.name} = '{ct.value}' (member index {list(CropType).index(ct)})")

print("\n  All 5 target classes present:")
for ct in target_classes:
    count = sum(1 for r in valid if r["crop_type"] == ct)
    print(f"    {ct}: {count}")

# ============================================================
# STEP 5: FINAL CLASS DISTRIBUTION
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: FINAL CLASS DISTRIBUTION")
print("=" * 70)

# Existing corpus distribution (from data_season.csv, 2018-2019 only for crop labels)
CROP_ALIASES = {
    "coconut": ["coconut"],
    "pepper": ["pepper"],
    "coffee": ["coffee"],
    "cardamom": ["cardamum"],
    "blackgram": ["blackgram"],
    "arecanut": ["arecanut"],
    "paddy": ["paddy"],
    "ragi": ["ragi"],
    "maize": ["maize"],
    "ginger": ["ginger"],
    "cocoa": ["cocoa"],
    "cashew": ["cashew"],
    "tea": ["tea"],
    "cotton": ["cotton"],
    "groundnut": ["groundnut"],
}

def normalize_crop(name):
    n = name.strip().lower()
    for canonical, aliases in CROP_ALIASES.items():
        if n in aliases:
            return canonical
    return n

existing_2018_2019 = [r for r in existing if r.get("Year", "").strip() in ("2018", "2019")]
existing_crop_dist = Counter(normalize_crop(r.get("Crops", "")) for r in existing_2018_2019)
existing_total = sum(existing_crop_dist.values())

# Government matched distribution
gov_dist = Counter(r["crop_type"] for r in valid)
gov_total = sum(gov_dist.values())

# Combined
combined = Counter()
combined.update(existing_crop_dist)
combined.update(gov_dist)
combined_total = sum(combined.values())

print(f"\n  EXISTING CORPUS (data_season 2018-2019):")
print(f"  {'Class':<15} {'Count':>6} {'%':>7} {'Years':>6} {'Seasons':>8}")
for crop, count in existing_crop_dist.most_common():
    pct = 100 * count / existing_total
    years = set(r.get("Year", "") for r in existing_2018_2019 if normalize_crop(r.get("Crops", "")) == crop)
    seasons = set(r.get("Season", "") for r in existing_2018_2019 if normalize_crop(r.get("Crops", "")) == crop)
    print(f"  {crop:<15} {count:>6} {pct:>6.1f}% {','.join(sorted(years)):>6} {','.join(sorted(seasons)):>8}")
print(f"  {'TOTAL':<15} {existing_total:>6}")

print(f"\n  GOVERNMENT MATCHED:")
print(f"  {'Class':<15} {'Count':>6} {'%':>7} {'Villages':>8} {'Years':>6}")
for crop in ["coconut", "pepper", "coffee", "cardamom", "blackgram"]:
    count = gov_dist.get(crop, 0)
    pct = 100 * count / gov_total if gov_total else 0
    villages = len(set(r["village"] for r in valid if r["crop_type"] == crop))
    years = len(set(r["year"] for r in valid if r["crop_type"] == crop))
    print(f"  {crop:<15} {count:>6} {pct:>6.1f}% {villages:>8} {years:>6}")
print(f"  {'TOTAL':<15} {gov_total:>6}")

print(f"\n  COMBINED MERGED:")
print(f"  {'Class':<15} {'Count':>6} {'%':>7} {'Imbalance':>10}")
max_count = max(combined.values()) if combined else 1
for crop in sorted(combined.keys()):
    count = combined[crop]
    pct = 100 * count / combined_total
    imbalance = max_count / count if count > 0 else float("inf")
    print(f"  {crop:<15} {count:>6} {pct:>6.1f}% {imbalance:>9.1f}x")
print(f"  {'TOTAL':<15} {combined_total:>6}")

# Imbalance ratio
if combined:
    counts = list(combined.values())
    print(f"\n  Imbalance ratio (max/min among target classes):")
    target_counts = [combined.get(c, 0) for c in ["coconut", "pepper", "coffee", "cardamom", "blackgram"]]
    target_nonzero = [c for c in target_counts if c > 0]
    if target_nonzero:
        print(f"    Max class: {max(target_counts)} (coconut)")
        print(f"    Min class: {min(target_nonzero)} (blackgram)")
        print(f"    Ratio: {max(target_counts)/min(target_nonzero):.0f}:1")

# Write distributions
for name, dist, total in [
    ("existing_corpus_distribution", existing_crop_dist, existing_total),
    ("government_distribution", gov_dist, gov_total),
    ("merged_corpus_distribution", combined, combined_total),
]:
    path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "count", "percentage"])
        for crop, count in sorted(dist.items(), key=lambda x: -x[1]):
            w.writerow([crop, count, round(100*count/total, 2) if total else 0])
    print(f"  Written: {path}")

# ============================================================
# STEP 6: SPATIAL QUALITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: SPATIAL QUALITY")
print("=" * 70)

dists = []
for r in valid:
    d = r["distance_km"]
    if d and d != "None":
        try:
            dists.append(float(d))
        except:
            pass

dists.sort()
n = len(dists)
if n > 0:
    print(f"  Samples with distances: {n:,}")
    print(f"  Min:     {dists[0]:.4f} km")
    print(f"  Median:  {dists[n//2]:.4f} km")
    print(f"  Mean:    {sum(dists)/n:.4f} km")
    print(f"  P90:     {dists[int(n*0.9)]:.4f} km")
    print(f"  P95:     {dists[int(n*0.95)]:.4f} km")
    print(f"  Max:     {dists[-1]:.4f} km")
    print(f"  Std:     {math.sqrt(sum((d - sum(dists)/n)**2 for d in dists)/n):.4f} km")

# Distance histogram (text)
print(f"\n  Distance histogram:")
bins = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
for i in range(len(bins) - 1):
    count = sum(1 for d in dists if bins[i] <= d < bins[i+1])
    bar = "#" * min(count // 50, 60)
    print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}km: {count:>5} {bar}")

# ============================================================
# STEP 7: TEMPORAL QUALITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: TEMPORAL QUALITY")
print("=" * 70)

# Year distribution
year_dist = Counter(r["year"] for r in valid)
print(f"\n  Year distribution:")
for year in sorted(year_dist.keys()):
    print(f"    {year}: {year_dist[year]:,}")

# Season distribution
season_dist = Counter(r["season"] for r in valid)
print(f"\n  Season distribution:")
for season, count in season_dist.most_common():
    print(f"    {season}: {count:,}")

# Survey date range
survey_dates = []
for r in valid:
    sd = r.get("survey_date", "")
    if sd and sd != "None":
        try:
            survey_dates.append(datetime.strptime(sd, "%Y-%m-%d").date())
        except:
            pass

if survey_dates:
    print(f"\n  Survey date range: {min(survey_dates)} to {max(survey_dates)}")
    date_years = Counter(d.year for d in survey_dates)
    print(f"  Survey date years: {dict(sorted(date_years.items()))}")

# Temporal match status
temp_dist = Counter(r["temporal_status"] for r in valid)
print(f"\n  Temporal match status: {dict(temp_dist)}")

# ============================================================
# STEP 8: MULTIMODAL AVAILABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: MULTIMODAL AVAILABILITY")
print("=" * 70)

tab_avail = sum(1 for r in valid if r.get("tabular_matched") == "True")
img_avail = sum(1 for r in valid if r.get("satellite_status") in ("FULL", "PARTIAL"))
ndvi_avail = sum(1 for r in valid if r.get("ndvi_available") == "True")
evi_avail = sum(1 for r in valid if r.get("evi_available") == "True")
sentinel = sum(1 for r in valid if r.get("satellite_status") != "NOT_AVAILABLE")

all_avail = sum(1 for r in valid
                if r.get("tabular_matched") == "True"
                and r.get("ndvi_available") == "True"
                and r.get("evi_available") == "True"
                and r.get("temporal_status") in ("EXACT_SEASON", "WITHIN_TOLERANCE"))

print(f"  Total valid samples: {len(valid):,}")
print(f"  Tabular available:   {tab_avail:,} ({100*tab_avail/len(valid):.1f}%)")
print(f"  Image available:     {img_avail:,} ({100*img_avail/len(valid):.1f}%)")
print(f"  NDVI available:      {ndvi_avail:,} ({100*ndvi_avail/len(valid):.1f}%)")
print(f"  EVI available:       {evi_avail:,} ({100*evi_avail/len(valid):.1f}%)")
print(f"  Sentinel-2 compatible: {sentinel:,} ({100*sentinel/len(valid):.1f}%)")
print(f"  All modalities:      {all_avail:,} ({100*all_avail/len(valid):.1f}%)")

# Missing modalities
missing_tab = len(valid) - tab_avail
missing_img = len(valid) - img_avail
missing_ndvi = len(valid) - ndvi_avail
missing_evi = len(valid) - evi_avail
print(f"\n  Missing modalities:")
print(f"    Tabular: {missing_tab:,}")
print(f"    NDVI: {missing_ndvi:,}")
print(f"    EVI: {missing_evi:,}")

# Satellite status breakdown
sat_dist = Counter(r.get("satellite_status", "?") for r in valid)
print(f"\n  Satellite status: {dict(sat_dist)}")

print("\nPhase 2 complete.")
