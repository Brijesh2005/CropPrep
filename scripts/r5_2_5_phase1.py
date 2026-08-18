"""
R5.2.5 — FINAL CROP CORPUS MERGE AND DUPLICATE AUDIT

Phase 1: Steps 1-3 — Duplicate audit, internal dupes, overlap check.
"""
import sys
sys.path.insert(0, "D:/CropPrep")

import json, os, csv, math, time
from collections import Counter, defaultdict
from datetime import datetime
from dataclasses import dataclass, field

OUTPUT_DIR = "D:/CropPrep/govt_crop_matched_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load full R5.2.4 match results (all 60,752 model-head records)
# ============================================================
print("=" * 70)
print("R5.2.5 — FINAL CROP CORPUS MERGE AND DUPLICATE AUDIT")
print("=" * 70)

print("\nLoading R5.2.4 full match results...")
full_match = []
with open(os.path.join(OUTPUT_DIR, "government_crop_stam_match.csv"), "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        full_match.append(row)

print(f"  Total model-head records: {len(full_match):,}")

# ============================================================
# STEP 1: AUDIT THE 57,147 DUPLICATES
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: AUDIT THE 57,147 DUPLICATES")
print("=" * 70)

dupes = [r for r in full_match if r["is_duplicate"] == "True"]
non_dupes = [r for r in full_match if r["is_duplicate"] != "True"]

print(f"  Duplicates: {len(dupes):,}")
print(f"  Non-duplicates: {len(non_dupes):,}")

# Duplicate key was: (village_taluk, year, season, crop_type, survey_date)
# Check what fields differ between duplicates
dupe_groups = defaultdict(list)
for r in full_match:
    key = (r["village"].strip().lower(), r["taluk"].strip().lower(),
           r["year"], r["season"], r["crop_type"], r["survey_date"])
    dupe_groups[key].append(r)

print(f"\n  Unique duplicate keys: {len(dupe_groups):,}")
multi_record_groups = {k: v for k, v in dupe_groups.items() if len(v) > 1}
print(f"  Groups with >1 record: {len(multi_record_groups):,}")

# Analyze what differs within groups
print("\n  Analyzing within-group differences...")
diff_field_counts = Counter()
sample_diffs = []
for key, records in list(multi_record_groups.items())[:1000]:
    if len(records) < 2:
        continue
    fields_that_differ = set()
    for field_name in ["lat", "lon", "source_crop", "hobli", "distance_km"]:
        vals = set(r[field_name] for r in records)
        if len(vals) > 1:
            fields_that_differ.add(field_name)
            diff_field_counts[field_name] += 1
    
    if fields_that_differ and len(sample_diffs) < 20:
        sample_diffs.append({
            "group_key": key,
            "n_records": len(records),
            "differing_fields": list(fields_that_differ),
            "lat_range": (min(float(r["lat"]) for r in records if r["lat"]),
                          max(float(r["lat"]) for r in records if r["lat"])) if "lat" in fields_that_differ else None,
            "lon_range": (min(float(r["lon"]) for r in records if r["lon"]),
                          max(float(r["lon"]) for r in records if r["lon"])) if "lon" in fields_that_differ else None,
            "source_crops": list(set(r["source_crop"] for r in records)),
            "hoblis": list(set(r["hobli"] for r in records)),
            "distance_range": (min(float(r["distance_km"]) for r in records if r["distance_km"] and r["distance_km"] != "None"),
                               max(float(r["distance_km"]) for r in records if r["distance_km"] and r["distance_km"] != "None")) if "distance_km" in fields_that_differ else None,
        })

print("\n  Fields that differ within duplicate groups:")
for field_name, count in diff_field_counts.most_common():
    print(f"    {field_name}: {count:,} groups")

print("\n  Sample duplicate groups (first 10):")
for i, sd in enumerate(sample_diffs[:10]):
    print(f"\n    Group {i+1}: key={sd['group_key']}")
    print(f"      Records: {sd['n_records']}")
    print(f"      Differing: {sd['differing_fields']}")
    if sd["lat_range"]:
        print(f"      Lat range: ({sd['lat_range'][0]:.6f}, {sd['lat_range'][1]:.6f})")
    if sd["lon_range"]:
        print(f"      Lon range: ({sd['lon_range'][0]:.6f}, {sd['lon_range'][1]:.6f})")
    if sd["distance_range"]:
        print(f"      Distance range: ({sd['distance_range'][0]:.4f}, {sd['distance_range'][1]:.4f}) km")
    print(f"      Source crops: {sd['source_crops']}")
    print(f"      Hoblis: {sd['hoblis']}")

# Determine if duplicates could represent different fields/plots
print("\n  --- Duplicate legitimacy assessment ---")
# Check: do duplicate records with different coordinates represent different fields?
coord_dupes = 0
for key, records in multi_record_groups.items():
    lats = set(round(float(r["lat"]), 4) for r in records if r["lat"])
    lons = set(round(float(r["lon"]), 4) for r in records if r["lon"])
    if len(lats) > 1 or len(lons) > 1:
        coord_dupes += 1

print(f"  Groups with different coords: {coord_dupes:,} / {len(multi_record_groups):,}")

# Check group size distribution
group_sizes = Counter(len(v) for v in multi_record_groups.values())
print(f"\n  Group size distribution:")
for size in sorted(group_sizes.keys()):
    print(f"    Size {size}: {group_sizes[size]:,} groups ({group_sizes[size] * size:,} records)")

# Write duplicate analysis
dup_analysis = {
    "total_records": len(full_match),
    "duplicate_count": len(dupes),
    "non_duplicate_count": len(non_dupes),
    "unique_duplicate_keys": len(dupe_groups),
    "multi_record_groups": len(multi_record_groups),
    "fields_differing": dict(diff_field_counts),
    "coord_different_groups": coord_dupes,
    "group_size_distribution": {str(k): v for k, v in sorted(group_sizes.items())},
    "sample_diffs": sample_diffs[:20],
    "assessment": "",
}

if coord_dupes > len(multi_record_groups) * 0.5:
    dup_analysis["assessment"] = "MANY_GROUPS_HAVE_DIFFERENT_COORDS"
elif coord_dupes > 0:
    dup_analysis["assessment"] = "SOME_GROUPS_HAVE_DIFFERENT_COORDS"
else:
    dup_analysis["assessment"] = "ALL_GROUPS_SHARE_COORDS"

with open(os.path.join(OUTPUT_DIR, "duplicate_analysis.json"), "w", encoding="utf-8") as f:
    json.dump(dup_analysis, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Written: {OUTPUT_DIR}/duplicate_analysis.json")

# Write duplicate examples CSV
with open(os.path.join(OUTPUT_DIR, "duplicate_examples.csv"), "w", newline="", encoding="utf-8") as f:
    fields = ["group_key_village", "group_key_year", "group_key_season", "group_key_crop",
              "group_key_survey_date", "n_records", "differing_fields", "sample_record_hobli",
              "sample_record_lat", "sample_record_lon", "sample_record_source_crop",
              "sample_record_distance_km", "sample_record_crop_status"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for sd in sample_diffs[:50]:
        key = sd["group_key"]
        records = multi_record_groups[key]
        r0 = records[0]
        w.writerow({
            "group_key_village": key[0], "group_key_year": key[1],
            "group_key_season": key[2], "group_key_crop": key[3],
            "group_key_survey_date": key[4], "n_records": sd["n_records"],
            "differing_fields": "|".join(sd["differing_fields"]),
            "sample_record_hobli": r0["hobli"],
            "sample_record_lat": r0["lat"],
            "sample_record_lon": r0["lon"],
            "sample_record_source_crop": r0["source_crop"],
            "sample_record_distance_km": r0["distance_km"],
            "sample_record_crop_status": r0["crop_status"],
        })
print(f"  Written: {OUTPUT_DIR}/duplicate_examples.csv")

# ============================================================
# STEP 2: GOVERNMENT-INTERNAL DUPLICATE CHECK
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: GOVERNMENT-INTERNAL DUPLICATE CHECK")
print("=" * 70)

# Use all 60,752 records for internal analysis
print("  Using all 60,752 model-head records...")

# Exact duplicate rows (all fields identical)
row_tuples = [tuple(sorted(r.items())) for r in full_match]
exact_dupes = len(row_tuples) - len(set(row_tuples))
print(f"  Exact duplicate rows: {exact_dupes:,}")

# Semantic duplicates: same village/year/season/crop but different coords
sem_key = lambda r: (r["village"].strip().lower(), r["year"], r["season"], r["crop_type"])
sem_groups = defaultdict(list)
for r in full_match:
    sem_groups[sem_key(r)].append(r)

sem_dupes = sum(1 for k, v in sem_groups.items() if len(v) > 1)
sem_total = sum(len(v) for k, v in sem_groups.items() if len(v) > 1)
print(f"  Semantic duplicate groups (village/year/season/crop): {sem_dupes:,} ({sem_total:,} records)")

# Same village/year/season/crop but different coordinates
coord_diff_groups = 0
for k, records in sem_groups.items():
    if len(records) > 1:
        lats = set(round(float(r["lat"]), 4) for r in records if r["lat"])
        lons = set(round(float(r["lon"]), 4) for r in records if r["lon"])
        if len(lats) > 1 or len(lons) > 1:
            coord_diff_groups += 1
print(f"  Same v/y/s/c but DIFFERENT coords: {coord_diff_groups:,}")

# Same village/year/season/crop but different survey dates
date_diff_groups = 0
for k, records in sem_groups.items():
    if len(records) > 1:
        dates = set(r["survey_date"] for r in records if r["survey_date"])
        if len(dates) > 1:
            date_diff_groups += 1
print(f"  Same v/y/s/c but DIFFERENT survey dates: {date_diff_groups:,}")

# ============================================================
# STEP 3: GOVERNMENT VS EXISTING CORPUS OVERLAP
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: GOVERNMENT VS EXISTING CORPUS OVERLAP")
print("=" * 70)

# Load data_season.csv as the source of existing 74 crop-labeled observations
existing = []
with open("D:/CropPrep/Tabular_Datasets/data_season.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        existing.append(row)

# Filter to 2018-2019 (where STAM matching produces crop labels)
existing_crop_labeled = [r for r in existing
                         if r.get("Year", "").strip() in ("2018", "2019")]
print(f"  Existing crop-labeled candidates (2018-2019): {len(existing_crop_labeled):,}")

# Canonical crop name mapping for comparison
CROP_ALIASES = {
    "coconut": ["coconut", "coconut"],
    "pepper": ["pepper", "pepper (black)"],
    "coffee": ["coffee", "coffee robusta", "coffee arabica"],
    "cardamom": ["cardamom", "cardamum"],
    "blackgram": ["blackgram", "urad"],
    "arecanut": ["arecanut", "betel nuts (areca nuts)"],
    "paddy": ["paddy", "paddy-h", "paddy-l"],
    "ragi": ["ragi", "ragi -h", "ragi-l"],
    "maize": ["maize", "jowar-h", "jowar-l"],
    "ginger": ["ginger"],
    "cocoa": ["cocoa"],
    "cashew": ["cashew", "cashewnuts"],
    "tea": ["tea"],
    "cotton": ["cotton"],
    "groundnut": ["groundnut"],
}

# Normalize existing crop names to canonical
def normalize_crop(name):
    n = name.strip().lower()
    for canonical, aliases in CROP_ALIASES.items():
        if n in aliases:
            return canonical
    return n

# Build existing observation identity
existing_obs = []
for r in existing_crop_labeled:
    crop = normalize_crop(r.get("Crops", ""))
    year = r.get("Year", "").strip()
    season = r.get("Season", "").strip()
    location = r.get("Location", "").strip()
    existing_obs.append({
        "crop": crop,
        "year": year,
        "season": season,
        "location": location,
        "yield_kg_ha": r.get("yeilds", ""),
        "area": r.get("Area", ""),
    })

# Government valid samples
gov_valid = [r for r in full_match if r["valid_cropfusion_sample"] == "True"]
print(f"  Government valid samples: {len(gov_valid):,}")

# Check overlap: government village/taluk vs existing location
# Government taluks: BELTANGADI, KOKKADA, MANGALURU A, MULKI, PANEMANGALURU, PANJA
# Existing locations: Bangalore, Chikmangaluru, Davangere, Gulbarga, Hassan, Kasaragodu, Kodagu, Madikeri, Mangalore, Mysuru, Raichur

# Map government taluks to existing location names
TALUK_TO_LOCATION = {
    "BELTANGADI": "Chikmangaluru",
    "KOKKADA": "Chikmangaluru",
    "MANGALURU A": "Mangalore",
    "MULKI": "Mangalore",
    "PANEMANGALURU": "Mangalore",
    "PANJA": "Madikeri",
}

# Check overlap
exact_overlaps = 0
spatial_temporal_overlaps = 0
same_vyc_overlaps = 0
genuinely_new = 0

overlap_details = []
new_observations = []

for g in gov_valid:
    g_crop = g["crop_type"]
    g_year = g["year"]
    g_season = g["season"]
    g_taluk = g["taluk"].strip().upper()
    g_location = TALUK_TO_LOCATION.get(g_taluk, g_taluk)
    
    matched_existing = None
    for e in existing_obs:
        if (e["crop"] == g_crop and
            e["year"] == str(g_year) and
            e["season"] == g_season):
            matched_existing = e
            break
    
    if matched_existing:
        exact_overlaps += 1
        overlap_details.append({
            "govt_village": g["village"],
            "govt_taluk": g["taluk"],
            "govt_crop": g_crop,
            "govt_year": g_year,
            "govt_season": g_season,
            "existing_location": matched_existing["location"],
            "existing_crop": matched_existing["crop"],
            "existing_year": matched_existing["year"],
            "existing_season": matched_existing["season"],
            "match_type": "exact_crop_year_season",
        })
    else:
        genuinely_new += 1
        new_observations.append(g)

print(f"  Exact overlaps (crop+year+season): {exact_overlaps:,}")
print(f"  Genuinely new observations: {genuinely_new:,}")

# Show overlap details
print("\n  Overlap examples (first 10):")
for od in overlap_details[:10]:
    print(f"    Govt: {od['govt_crop']}/{od['govt_year']}/{od['govt_season']}/{od['govt_village']}/{od['govt_taluk']}")
    print(f"    Existing: {od['existing_crop']}/{od['existing_year']}/{od['existing_season']}/{od['existing_location']}")

# Write overlap report
overlap_report = {
    "existing_crop_labeled_candidates": len(existing_crop_labeled),
    "government_valid_samples": len(gov_valid),
    "exact_overlaps": exact_overlaps,
    "genuinely_new": genuinely_new,
    "overlap_details": overlap_details[:50],
}

with open(os.path.join(OUTPUT_DIR, "government_corpus_overlap.json"), "w", encoding="utf-8") as f:
    json.dump(overlap_report, f, indent=2, ensure_ascii=False, default=str)
print(f"\n  Written: {OUTPUT_DIR}/government_corpus_overlap.json")

print("\nPhase 1 complete.")
