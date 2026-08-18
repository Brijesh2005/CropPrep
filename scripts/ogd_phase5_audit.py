"""
Phase 5: Clean duplicates + full audit with real OGD data.
"""
import json
import os
import csv
from collections import Counter

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

# Remove duplicates (beltangadi and mulki appear twice)
duplicates = [
    "ogd_discovered_MjQxNDExOT.json",  # same as ogd_mulki_kharif_2021_22.json
    "ogd_discovered_MjQzMTExOD.json",  # same as ogd_beltangadi_kharif_2020_21.json
    "ogd_discovered_MjQ1MjExOD.json",  # truncated re-download of panja
    "ogd_discovered_MjQzMjExOD.json",  # truncated re-download of kokkada
]

print("=== Cleaning duplicate files ===")
for f in duplicates:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Removed: {f}")

# Now load ALL unique data
print("\n=== Loading all OGD data ===")
all_data = []
hobli_summary = {}

for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("ogd_") and f.endswith(".json"):
        path = os.path.join(OUTPUT_DIR, f)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list) and len(data) > 0:
                h = data[0].get("Hobli_Name", "?")
                t = data[0].get("Taluk_Name", "?")
                s = data[0].get("Season", "?")
                y = data[0].get("Years", "?")
                n = len(data)
                crops = Counter(row.get("Cropname", "?") for row in data)
                hobli_summary[f] = {
                    "hobli": h, "taluk": t, "season": s, "year": y,
                    "records": n, "unique_crops": len(crops),
                    "top_crops": dict(crops.most_common(5))
                }
                all_data.extend(data)
                print(f"  {f}: {n:,} records ({h}, {t}, {s} {y})")
        except Exception as e:
            print(f"  ERROR {f}: {e}")

print(f"\n  TOTAL: {len(all_data):,} records across {len(hobli_summary)} files")

# Analyze crop distribution
print("\n=== Crop Distribution ===")
all_crops = Counter(row.get("Cropname", "?") for row in all_data)
print(f"  Total unique crop names: {len(all_crops)}")
print(f"\n  All crops:")
for crop, count in all_crops.most_common():
    print(f"    {crop}: {count:,}")

# Geographic analysis
print("\n=== Geographic Coverage ===")
lats = []
lons = []
for row in all_data:
    try:
        lat = float(row.get("Latitude", "0"))
        lon = float(row.get("Longtitude", "0"))
        if lat > 0 and lon > 0:
            lats.append(lat)
            lons.append(lon)
    except:
        pass

if lats:
    print(f"  Valid coordinates: {len(lats):,} / {len(all_data):,}")
    print(f"  Latitude range: {min(lats):.6f} to {max(lats):.6f}")
    print(f"  Longitude range: {min(lons):.6f} to {max(lons):.6f}")
    print(f"  Mean lat: {sum(lats)/len(lats):.6f}, mean lon: {sum(lons)/len(lons):.6f}")

# Save unified dataset
unified_csv = os.path.join(OUTPUT_DIR, "ogd_unified_all_hoblis.csv")
if all_data:
    with open(unified_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
        writer.writeheader()
        writer.writerows(all_data)
    print(f"\n  Unified CSV saved: {unified_csv}")

# Save summary
summary_path = os.path.join(OUTPUT_DIR, "ogd_download_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({
        "total_records": len(all_data),
        "total_unique_crops": len(all_crops),
        "crop_distribution": dict(all_crops),
        "hoblis": hobli_summary,
        "coordinate_stats": {
            "valid_count": len(lats),
            "lat_min": min(lats) if lats else 0,
            "lat_max": max(lats) if lats else 0,
            "lon_min": min(lons) if lons else 0,
            "lon_max": max(lons) if lons else 0,
        }
    }, f, indent=2, ensure_ascii=False)
print(f"  Summary saved: {summary_path}")
