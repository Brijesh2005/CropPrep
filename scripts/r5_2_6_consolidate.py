"""
Consolidate all downloaded hobli data and compute full crop distribution.
Also check for rare classes (blackgram, cardamom, coffee) across all hoblis.
"""
import csv, os, json
from collections import defaultdict, Counter

DATA_DIR = "D:/CropPrep/govt_crop_survey_data"

# All downloaded files
files = [
    # Original 6
    ("ogd_beltangadi_kharif_2020_21.csv", "Beltangadi", "Belthangady", "Kharif", "2020-21"),
    ("ogd_kokkada_kharif_2020_21.csv", "Kokkada", "Belthangady", "Kharif", "2020-21"),
    ("ogd_mangaluru_a_kharif_2020_2021.csv", "Mangaluru A", "Mangalore", "Kharif", "2020-21"),
    ("ogd_mulki_kharif_2021_22.csv", "Mulki", "Mangalore", "Kharif", "2021-22"),
    ("ogd_panemangaluru_rabi_2021_2022.csv", "Panemangaluru", "Bantwal", "Rabi", "2021-22"),
    ("ogd_panja_kharif_2020_21.csv", "Panja", "Sullia", "Kharif", "2020-21"),
    # New downloads v1
    ("ogd_venuru_kharif_2020_21.csv", "Venuru", "Belthangady", "Kharif", "2020-21"),
    ("ogd_sulya_kharif_2020_21.csv", "Sulya", "Sullia", "Kharif", "2020-21"),
    ("ogd_uppinangadi_kharif_2021_22.csv", "Uppinangadi", "Puttur", "Kharif", "2021-22"),
    ("ogd_vitla_kharif_2020_21.csv", "Vitla", "Bantwal", "Kharif", "2020-21"),
    ("ogd_putturu_kharif_2020_21.csv", "Putturu", "Puttur", "Kharif", "2020-21"),
    # New downloads v2
    ("ogd_mangaluru_b_kharif_2020_21.csv", "Mangaluru B", "Mangalore", "Kharif", "2020-21"),
    ("ogd_suratkal_kharif_2020_21.csv", "Suratkal", "Mangalore", "Kharif", "2020-21"),
    ("ogd_bantvala_kharif_2020_21.csv", "Bantvala", "Bantwal", "Kharif", "2020-21"),
    ("ogd_venuru_rabi_2021_22.csv", "Venuru", "Belthangady", "Rabi", "2021-22"),
]

all_crop_counts = Counter()
hobli_crop_counts = defaultdict(Counter)
taluk_crop_counts = defaultdict(Counter)
total_records = 0
hobli_totals = {}

# Rare class tracking
rare_class_records = defaultdict(list)  # crop -> list of records

for filename, hobli, taluk, season, years in files:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print("MISSING: {}".format(filename))
        continue
    
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop = row.get("Cropname", "?").strip()
            count += 1
            all_crop_counts[crop] += 1
            hobli_crop_counts[hobli][crop] += 1
            taluk_crop_counts[taluk][crop] += 1
            
            # Track rare class records
            crop_lower = crop.lower()
            if any(x in crop_lower for x in ["blackgram", "black gram", "blackgram"]):
                rare_class_records["blackgram"].append({
                    "hobli": hobli, "taluk": taluk, "season": season, "years": years,
                    "cropname": crop, "lat": row.get("Latitude", ""), "lon": row.get("Longtitude", ""),
                })
            elif "cardamom" in crop_lower:
                rare_class_records["cardamom"].append({
                    "hobli": hobli, "taluk": taluk, "season": season, "years": years,
                    "cropname": crop, "lat": row.get("Latitude", ""), "lon": row.get("Longtitude", ""),
                })
            elif "coffee" in crop_lower:
                rare_class_records["coffee"].append({
                    "hobli": hobli, "taluk": taluk, "season": season, "years": years,
                    "cropname": crop, "lat": row.get("Latitude", ""), "lon": row.get("Longtitude", ""),
                })
    
    hobli_totals[hobli] = count
    total_records += count
    print("  {}: {} records ({} {})".format(hobli, count, season, years))

print("\n" + "=" * 70)
print("TOTAL RECORDS: {}".format(total_records))
print("UNIQUE HOBLES: {}".format(len(hobli_totals)))
print("=" * 70)

print("\nFULL CROP DISTRIBUTION:")
for crop, count in sorted(all_crop_counts.items(), key=lambda x: -x[1]):
    print("  {}: {}".format(crop, count))

print("\nRARE CLASS RECORDS:")
for crop, records in rare_class_records.items():
    print("  {}: {} records".format(crop, len(records)))
    for r in records[:5]:
        print("    {}/{} ({}/{}) - {}".format(r["hobli"], r["taluk"], r["season"], r["years"], r["cropname"]))
    if len(records) > 5:
        print("    ... and {} more".format(len(records) - 5))

print("\nPER-HOBLE TOTALS:")
for hobli, total in sorted(hobli_totals.items(), key=lambda x: -x[1]):
    print("  {}: {}".format(hobli, total))

# Save full summary
summary = {
    "total_records": total_records,
    "unique_hoblis": len(hobli_totals),
    "hobli_totals": hobli_totals,
    "crop_distribution": dict(all_crop_counts),
    "rare_classes": {k: len(v) for k, v in rare_class_records.items()},
    "rare_class_details": {k: v for k, v in rare_class_records.items()},
}
with open(os.path.join(DATA_DIR, "r5_2_6_full_consolidation.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nSaved: r5_2_6_full_consolidation.json")
