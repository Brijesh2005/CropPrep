"""
R5.2.4 — Government Crop Survey Integration and STAM Matching (v5)

Uses dict-based spatial grid index with 3-decimal precision (~100m cells).
Each OGD coord checks only ~9 neighboring grid cells.
"""
import sys
sys.path.insert(0, "D:/CropPrep")

import json, os, csv, math, time
from collections import Counter, defaultdict
from datetime import date, datetime
from shared.enums import CropType, Season
from shared.enums.crop_taxonomy import resolve_crop_label, LabelMatchStatus

# ============================================================
CFG = {"spatial_tolerance_km": 5.0, "temporal_tolerance_days": 15}
SPATIAL_TOLS = [1.0, 2.0, 5.0, 10.0]
TEMPORAL_TOLS = [7, 15, 30]
OGD_DIR = "D:/CropPrep/govt_crop_survey_data"
OUTPUT_DIR = "D:/CropPrep/govt_crop_matched_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)
SEASON_MONTHS = {"Kharif": (6, 10), "Rabi": (11, 3), "Zaid": (4, 5)}
OGD_SEASON_MAP = {"Kharif": "Kharif", "Rabi": "Rabi"}
GRID_PREC = 3  # decimal places for grid cells (~111m)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

t0 = time.time()
print("=" * 70)
print("R5.2.4 — GOVERNMENT CROP SURVEY INTEGRATION AND STAM MATCHING")
print("=" * 70)

# ============================================================
# STEP 1: Load OGD + resolve labels
# ============================================================
print("\n[1/10] Loading and resolving OGD labels...")
all_ogd = []
for f in sorted(os.listdir(OGD_DIR)):
    if f.startswith("ogd_") and f.endswith(".json") and not f.startswith("_"):
        with open(os.path.join(OGD_DIR, f), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and len(data) > 0:
            all_ogd.extend(data)

model_head_classes = {"coconut", "pepper", "coffee", "cardamom", "blackgram"}
model_head = []
for row in all_ogd:
    res = resolve_crop_label(row.get("Cropname", ""))
    entry = dict(row)
    entry["_crop_type"] = res.crop_type.value
    entry["_crop_status"] = res.status.value
    if entry["_crop_type"] in model_head_classes:
        model_head.append(entry)
print(f"  Total OGD: {len(all_ogd):,}  Model-head: {len(model_head):,}")

# ============================================================
# STEP 2: Load DK grid + build spatial grid index
# ============================================================
print("\n[2/10] Loading DK_Features grid + building grid index...")
dk_grid = defaultdict(list)  # (int(lat*1000), int(lon*1000)) -> [cell_dict, ...]

for year in range(2018, 2024):
    path = f"D:/CropPrep/Tabular_Datasets/DK_Features_{year}.csv"
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                if lat <= 0 or lon <= 0:
                    continue
                cell = {
                    "year": year, "lat": lat, "lon": lon,
                    "ndvi": row.get("NDVI", ""),
                    "evi": row.get("EVI", ""),
                    "kharif_ndvi": row.get("Kharif_NDVI", ""),
                    "rabi_ndvi": row.get("Rabi_NDVI", ""),
                    "kharif_evi": row.get("Kharif_EVI", ""),
                    "rabi_evi": row.get("Rabi_EVI", ""),
                }
                gk = (int(round(lat, GRID_PREC) * 10**GRID_PREC),
                      int(round(lon, GRID_PREC) * 10**GRID_PREC))
                dk_grid[gk].append(cell)
            except:
                pass

total_dk = sum(len(v) for v in dk_grid.values())
print(f"  DK grid cells: {total_dk:,}  Grid buckets: {len(dk_grid):,}")

# ============================================================
# STEP 3: Spatial matching using grid index
# ============================================================
print("\n[3/10] Spatial matching...")

def grid_key(lat, lon):
    return (int(round(lat, GRID_PREC) * 10**GRID_PREC),
            int(round(lon, GRID_PREC) * 10**GRID_PREC))

def query_nearest(lat, lon, grid, cell_size_deg=0.001, radius_cells=5):
    """Find nearest DK cell using grid index."""
    gk = grid_key(lat, lon)
    step = 1  # step in grid key units (each unit = 0.001 degrees)
    
    best_dist = float("inf")
    best_cell = None
    
    for di in range(-radius_cells, radius_cells + 1):
        for dj in range(-radius_cells, radius_cells + 1):
            key = (gk[0] + di, gk[1] + dj)
            for cell in grid.get(key, []):
                d = haversine_km(lat, lon, cell["lat"], cell["lon"])
                if d < best_dist:
                    best_dist = d
                    best_cell = cell
                    if d < 0.001:  # essentially exact match
                        return best_dist, best_cell
    
    return best_dist, best_cell

# Deduplicate by coordinate
coord_to_indices = defaultdict(list)
for i, entry in enumerate(model_head):
    try:
        lat = float(entry.get("Latitude", "0"))
        lon = float(entry.get("Longtitude", "0"))
        if lat > 0 and lon > 0:
            key = (round(lat, 6), round(lon, 6))
        else:
            key = None
    except:
        key = None
    coord_to_indices[key].append(i)

valid_coords = {k: v for k, v in coord_to_indices.items() if k is not None}
print(f"  Unique coords to match: {len(valid_coords):,}")

# Pre-compute for each unique coord
coord_results = {}
for ci, (coord, indices) in enumerate(valid_coords.items()):
    olat, olon = coord
    dist, cell = query_nearest(olat, olon, dk_grid, radius_cells=10)
    coord_results[coord] = (dist, cell)
    if (ci + 1) % 5000 == 0:
        elapsed = time.time() - t0
        print(f"  {ci+1}/{len(valid_coords)} coords processed ({elapsed:.1f}s)...")

print(f"  All {len(coord_results):,} coords processed ({time.time()-t0:.1f}s)")

# Assign back
spatial_matches = [None] * len(model_head)
spatial_distances = []

for coord, indices in coord_to_indices.items():
    if coord is None:
        for idx in indices:
            spatial_matches[idx] = {"match_status": "NO_COORDINATES", "distance_km": None,
                                    "has_ndvi": False, "has_evi": False,
                                    "has_kharif_ndvi": False, "has_rabi_ndvi": False, "matched_year": None}
        continue
    
    dist, cell = coord_results[coord]
    spatial_distances.append(dist)
    
    has_ndvi = bool(cell and cell.get("ndvi"))
    has_evi = bool(cell and cell.get("evi"))
    has_kharif = bool(cell and cell.get("kharif_ndvi"))
    has_rabi = bool(cell and cell.get("rabi_ndvi"))
    
    status = "MATCHED" if dist <= CFG["spatial_tolerance_km"] else ("NEAR_MISS" if dist <= 10.0 else "NO_MATCH")
    
    result = {"match_status": status, "distance_km": round(dist, 4),
              "has_ndvi": has_ndvi, "has_evi": has_evi,
              "has_kharif_ndvi": has_kharif, "has_rabi_ndvi": has_rabi,
              "matched_year": cell["year"] if cell else None}
    
    for idx in indices:
        spatial_matches[idx] = result

spatial_status = Counter(m["match_status"] for m in spatial_matches)
vd_finite = [d for d in spatial_distances if d is not None and math.isfinite(d)]
nv = len(vd_finite)
if nv > 0:
    vd_finite.sort()
    print(f"  Distances (finite): min={vd_finite[0]:.4f} median={vd_finite[nv//2]:.4f} mean={sum(vd_finite)/nv:.4f} P90={vd_finite[int(nv*0.9)]:.4f} max={vd_finite[-1]:.4f}")
print(f"  Inf distances: {sum(1 for d in spatial_distances if not math.isfinite(d))}")
print(f"  Status: {dict(spatial_status)}")

# ============================================================
# STEP 4: Temporal matching
# ============================================================
print("\n[4/10] Temporal matching...")
temporal_matches = []
temporal_diffs = []

for entry in model_head:
    ogd_season = entry.get("Season", "")
    ogd_years = entry.get("Years", "")
    survey_str = entry.get("CropSurveyDate", "")
    
    survey_date = None
    if survey_str and survey_str != "NULL":
        try:
            survey_date = datetime.strptime(survey_str, "%Y-%m-%d").date()
        except:
            pass
    
    try:
        year = int(ogd_years.split("-")[0])
    except:
        year = 2020
    
    mapped_season = OGD_SEASON_MAP.get(ogd_season, ogd_season)
    sm = SEASON_MONTHS.get(mapped_season)
    
    if sm:
        s_m, e_m = sm
        if s_m <= e_m:
            ss, se = date(year, s_m, 1), date(year, e_m, 28)
        else:
            ss, se = date(year, s_m, 1), date(year + 1, e_m, 28)
    else:
        ss = se = None
    
    tstatus = "NO_MATCH"
    ddiff = None
    if survey_date and ss and se:
        if ss <= survey_date <= se:
            tstatus, ddiff = "EXACT_SEASON", 0
        else:
            ddiff = min(abs((survey_date - ss).days), abs((survey_date - se).days))
            tstatus = "WITHIN_TOLERANCE" if ddiff <= CFG["temporal_tolerance_days"] else "OUTSIDE_TOLERANCE"
    elif survey_date:
        if survey_date.year == year:
            tstatus, ddiff = "YEAR_MATCH", 0
    else:
        tstatus = "YEAR_ONLY"
    
    temporal_matches.append({"status": tstatus, "survey_date": str(survey_date) if survey_date else None,
                             "ogd_season": ogd_season, "mapped_season": mapped_season, "year": year,
                             "date_difference_days": ddiff})
    if ddiff is not None:
        temporal_diffs.append(ddiff)

ts = Counter(m["status"] for m in temporal_matches)
print(f"  Status: {dict(ts)}")
if temporal_diffs:
    tds = sorted(temporal_diffs)
    print(f"  Date diffs: min={tds[0]} median={tds[len(tds)//2]} mean={sum(tds)/len(tds):.1f} max={tds[-1]}")

# ============================================================
# STEP 5: Tabular matching
# ============================================================
print("\n[5/10] Tabular matching...")
tab_recs = list(csv.DictReader(open("D:/CropPrep/Tabular_Datasets/data_season.csv", "r", encoding="utf-8")))
tab_index = defaultdict(list)
for rec in tab_recs:
    loc = rec.get("Location", "").strip().lower()
    try:
        yr = int(rec.get("Year", "0"))
    except:
        continue
    season = rec.get("Season", "").strip()
    tab_index[(loc, yr, season)].append(rec)

tab_matches = []
for entry in model_head:
    hobli = entry.get("Hobli_Name", "").strip().lower()
    taluk = entry.get("Taluk_Name", "").strip().lower()
    village = entry.get("Village_Name", "").strip().lower()
    try:
        year = int(entry.get("Years", "2020").split("-")[0])
    except:
        year = 2020
    
    matched = False
    matched_level = "none"
    for ln, lk in [("village", village), ("taluk", taluk), ("hobli", hobli)]:
        if matched:
            break
        for sn in ["Kharif", "Rabi", "Zaid"]:
            if (lk, year, sn) in tab_index:
                matched, matched_level = True, ln
                break
    
    if not matched:
        dist_name = entry.get("District_Name", "").lower().strip()
        if "dakshina kannada" in dist_name and year in range(2018, 2024):
            matched, matched_level = True, "district_grid"
    
    tab_matches.append({"matched": matched, "matched_level": matched_level})
print(f"  Status: {dict(Counter(m['matched_level'] for m in tab_matches))}")

# ============================================================
# STEP 6: Satellite matching
# ============================================================
print("\n[6/10] Satellite matching...")
sat_matches = []
for i, entry in enumerate(model_head):
    sp = spatial_matches[i]
    tm = temporal_matches[i]
    year = tm.get("year", 2020)
    sentinel = year >= 2017
    img = sentinel and (sp.get("has_ndvi") or sp.get("has_evi"))
    
    if img and sp.get("has_ndvi") and sp.get("has_evi"):
        ss = "FULL"
    elif img:
        ss = "PARTIAL"
    else:
        ss = "NOT_AVAILABLE"
    
    sat_matches.append({"image_available": img, "NDVI_available": sentinel and sp.get("has_ndvi"),
                        "EVI_available": sentinel and sp.get("has_evi"),
                        "sentinel2_compatible": sentinel, "match_status": ss})
print(f"  Status: {dict(Counter(m['match_status'] for m in sat_matches))}")

# ============================================================
# STEP 7: Complete multimodal match + dedup
# ============================================================
print("\n[7/10] Complete multimodal match + dedup...")
complete = []
seen = set()
dupes = 0

for i, entry in enumerate(model_head):
    ct = entry["_crop_type"]
    sp, tm, tb, sa = spatial_matches[i], temporal_matches[i], tab_matches[i], sat_matches[i]
    
    village = entry.get("Village_Name", "").strip().lower()
    taluk = entry.get("Taluk_Name", "").strip().lower()
    loc_key = f"{village}_{taluk}"
    year, season = tm.get("year", 2020), tm.get("mapped_season", "")
    survey_date = tm.get("survey_date", "")
    
    dk_key = (loc_key, year, season, ct, survey_date)
    is_dup = dk_key in seen
    if is_dup:
        dupes += 1
    else:
        seen.add(dk_key)
    
    vs = sp.get("match_status") == "MATCHED"
    vt = tm["status"] in ("EXACT_SEASON", "WITHIN_TOLERANCE", "YEAR_MATCH", "YEAR_ONLY")
    vtb = tb["matched"]
    vsa = sa["image_available"]
    all_valid = vs and vt and vtb and vsa and not is_dup
    
    rej = []
    if not vs: rej.append(f"spatial_{sp.get('match_status','?')}")
    if not vt: rej.append(f"temporal_{tm['status']}")
    if not vtb: rej.append("no_tabular")
    if not vsa: rej.append("no_satellite")
    if is_dup: rej.append("duplicate")
    
    complete.append({
        "source_crop": entry.get("Cropname", ""), "crop_type": ct,
        "crop_status": entry["_crop_status"], "hobli": entry.get("Hobli_Name", ""),
        "taluk": entry.get("Taluk_Name", ""), "village": entry.get("Village_Name", ""),
        "lat": entry.get("Latitude", ""), "lon": entry.get("Longtitude", ""),
        "year": year, "season": season, "survey_date": survey_date,
        "distance_km": sp.get("distance_km"), "spatial_status": sp.get("match_status"),
        "temporal_status": tm["status"], "tabular_matched": tb["matched"],
        "tabular_level": tb["matched_level"], "satellite_status": sa["match_status"],
        "ndvi_available": sa["NDVI_available"], "evi_available": sa["EVI_available"],
        "is_duplicate": is_dup, "valid_cropfusion_sample": all_valid,
        "rejection_reasons": rej,
    })

valid_samples = [m for m in complete if m["valid_cropfusion_sample"]]
rej_counter = Counter()
for m in complete:
    for r in m["rejection_reasons"]:
        rej_counter[r] += 1

print(f"  Total: {len(complete):,}  Valid: {len(valid_samples):,}  Dupes: {dupes:,}")
print(f"  Rejections: {dict(rej_counter)}")

# ============================================================
# STEP 8: Tolerance sensitivity
# ============================================================
print("\n[8/10] Tolerance sensitivity...")
print("  Spatial:")
for tol in SPATIAL_TOLS:
    print(f"    {tol:.1f}km: {sum(1 for d in spatial_distances if d <= tol):,}")
print("  Temporal:")
for tol in TEMPORAL_TOLS:
    print(f"    {tol}d: {sum(1 for d in temporal_diffs if d <= tol):,}")

# ============================================================
# STEP 9: Class distribution
# ============================================================
print("\n[9/10] Class distribution...")
current_crops = Counter()
with open("D:/CropPrep/Tabular_Datasets/data_season.csv", "r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        c = row.get("Crops", "").strip().lower()
        if c:
            current_crops[c] += 1

govt_raw = Counter(e["_crop_type"] for e in model_head)
govt_matched = Counter(m["crop_type"] for m in valid_samples)
combined = Counter()
combined.update(current_crops)
combined.update(govt_matched)

print(f"  Current: {sum(current_crops.values()):,}  Govt raw: {sum(govt_raw.values()):,}  Govt matched: {sum(govt_matched.values()):,}")
for c in ["coconut","pepper","coffee","cardamom","blackgram"]:
    print(f"    {c}: raw={govt_raw.get(c,0):,} matched={govt_matched.get(c,0):,}")

# ============================================================
# STEP 10: Write output files
# ============================================================
print("\n[10/10] Writing output files...")

path1 = os.path.join(OUTPUT_DIR, "government_crop_stam_match.csv")
with open(path1, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=complete[0].keys())
    w.writeheader()
    w.writerows(complete)
print(f"  {path1}")

path2 = os.path.join(OUTPUT_DIR, "government_crop_matched_v1.csv")
with open(path2, "w", newline="", encoding="utf-8") as f:
    if valid_samples:
        w = csv.DictWriter(f, fieldnames=valid_samples[0].keys())
        w.writeheader()
        w.writerows(valid_samples)
print(f"  {path2}")

if len(valid_samples) >= 1000:
    rec = "A. SUFFICIENT — proceed to integration"
elif len(valid_samples) >= 100:
    rec = "A. SUFFICIENT — small but usable"
elif len(valid_samples) > 0:
    rec = "B. INSUFFICIENT — download additional hoblis"
else:
    rec = "C. MATCHING QUALITY TOO WEAK"

report = {
    "audit_version": "R5.2.4", "generated_at": datetime.now().isoformat(),
    "config": CFG,
    "data_summary": {"total_ogd": len(all_ogd), "model_head": len(model_head),
                     "valid_samples": len(valid_samples), "duplicates": dupes},
    "spatial": {"tolerance_km": CFG["spatial_tolerance_km"],
                "min_km": vd_finite[0] if vd_finite else None, "median_km": vd_finite[nv//2] if vd_finite else None,
                "mean_km": round(sum(vd_finite)/nv, 4) if nv else None,
                "p90_km": vd_finite[int(nv*0.9)] if nv else None, "p95_km": vd_finite[int(nv*0.95)] if nv else None,
                "max_km": vd_finite[-1] if vd_finite else None,
                "inf_count": sum(1 for d in spatial_distances if not math.isfinite(d)),
                "status": dict(spatial_status)},
    "temporal": {"tolerance_days": CFG["temporal_tolerance_days"], "status": dict(ts)},
    "tabular": dict(Counter(m["matched_level"] for m in tab_matches)),
    "satellite": dict(Counter(m["match_status"] for m in sat_matches)),
    "rejections": dict(rej_counter),
    "class_dist": {"current": dict(current_crops), "govt_raw": dict(govt_raw),
                   "govt_matched": dict(govt_matched),
                   "combined": {k: v for k, v in combined.items() if v > 0}},
    "tolerance_sensitivity": {
        "spatial": {str(t): sum(1 for d in spatial_distances if d <= t) for t in SPATIAL_TOLS},
        "temporal": {str(t): sum(1 for d in temporal_diffs if d <= t) for t in TEMPORAL_TOLS},
    },
    "recommendation": rec,
}

path3 = os.path.join(OUTPUT_DIR, "government_crop_stam_match_report.json")
with open(path3, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(f"  {path3}")

elapsed = time.time() - t0
print(f"\n{'='*70}\nCOMPLETE — {elapsed:.1f}s\n{'='*70}")
print(f"\nRECOMMENDATION: {rec}")
