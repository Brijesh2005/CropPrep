"""
Phase 4: Broader brute-force + final audit with real data.
"""
import subprocess
import json
import os
import csv
import base64
import time
from collections import Counter

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

def curl_download(url, output_path, timeout=30):
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout), "-o", output_path, url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    return os.path.getsize(output_path) if os.path.exists(output_path) else 0

# Known hoblis in Dakshina Kannada (from web knowledge):
# Mangalore, Bantwal, Belthangady, Puttur, Sullia, Karkala, Moodbidri, 
# Kadaba, Panja, Kokkada, Beltangadi, Mulki, Mangaluru

# More brute-force with wider patterns
print("=== Extended brute-force key discovery ===")

# Patterns we've seen:
# 24311181 -> Beltangadi (Belthangady taluk)
# 24321181 -> Kokkada (Belthangady taluk) 
# 24141191 -> Mulki (Mangalore taluk)
# 24521181 -> Panja (Sullia taluk)
# 24221192 -> Panemangaluru (Bantwal taluk)
# 24111181 -> Mangaluru A (Mangalore taluk)

# Generate more candidates
candidates = set()

# Pattern 1: 24XXYYZZ where XX=taluk, YY=sequence, ZZ=season/year
for prefix in range(2400, 2460):
    for suffix_base in [1181, 1182, 1191, 1192, 1183, 1193, 1201, 1202]:
        candidates.add(f"{prefix}{suffix_base}")

# Pattern 2: 24XYYYZZZZ
for x in range(0, 10):
    for yyy in range(100, 600):
        for zz in [1181, 1191]:
            candidates.add(f"24{x}{yyy}{zz}")

# Remove known keys
known = {"24311181", "24321181", "24141191", "24521181", "24221192", "24111181"}
candidates -= known

# Test in batches
new_discoveries = {}
tested = 0
for num in sorted(candidates):
    if tested >= 100:  # Limit to avoid taking too long
        break
    encoded = base64.b64encode(num.encode()).decode()
    url = f"https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key={encoded}"
    tmp = os.path.join(OUTPUT_DIR, "_tmp_test.json")
    size = curl_download(url, tmp, timeout=10)
    tested += 1
    if size > 1000:
        try:
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 10:
                h = data[0].get("Hobli_Name", "?")
                t = data[0].get("Taluk_Name", "?")
                s = data[0].get("Season", "?")
                y = data[0].get("Years", "?")
                key_info = f"{h} ({t}) {s} {y}"
                # Check if we already have this
                existing = [v["hobli"] for v in new_discoveries.values()]
                if h not in existing:
                    print(f"  NEW: {num} -> {key_info} ({len(data)} records)")
                    new_discoveries[num] = {"key": encoded, "hobli": h, "taluk": t, "data": data}
                    safe_name = f"ogd_{h.lower().replace(' ', '_')}_{s.lower()}_{y.replace('-','_')}.json"
                    out_path = os.path.join(OUTPUT_DIR, safe_name)
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False)
                    csv_path = out_path.replace(".json", ".csv")
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
        except:
            pass
    os.remove(tmp) if os.path.exists(tmp) else None

print(f"\nTested {tested} keys, found {len(new_discoveries)} new hoblis")

# Final summary
print("\n" + "=" * 60)
print("FINAL DATA SUMMARY")
print("=" * 60)

total_records = 0
all_crops = Counter()
hobli_data = {}

for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("ogd_") and f.endswith(".json") and not f.startswith("_"):
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
                total_records += n
                crops = Counter(row.get("Cropname", "?") for row in data)
                all_crops.update(crops)
                hobli_data[f] = {"hobli": h, "taluk": t, "season": s, "year": y, "records": n, "crops": len(crops)}
                print(f"  {f}: {n:,} records, {len(crops)} crops, {h} ({t}), {s} {y}")
        except:
            pass

print(f"\n  TOTAL RECORDS: {total_records:,}")
print(f"  TOTAL UNIQUE CROPS: {len(all_crops)}")
print(f"\n  TOP 20 CROPS:")
for crop, count in all_crops.most_common(20):
    print(f"    {crop}: {count:,}")
