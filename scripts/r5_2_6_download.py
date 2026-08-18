"""
Download the highest-value new hobli resources for rare class expansion.
Focus on taluks likely to have coffee/cardamom/blackgram.
"""
import subprocess, base64, json, time, os

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

# Priority resources for rare class expansion:
# 1. VENURU/Belthangady (Kharif 2020-21) — same taluk as Beltangadi/Kokkada which have cardamom
# 2. SULYA/Sullia (Kharif 2020-21) — same taluk as Panja which has coffee/blackgram
# 3. UPPINANGADI/Puttur (Kharif 2021-22) — shows Pepper (Black)
# 4. VITLA/Bantwal (Kharif 2020-21) — different taluk for diversity
# 5. PUTTURU/Puttur (Kharif 2020-21) — different year from UPPINANGADI

RESOURCES = [
    {"key": "24331181", "name": "ogd_venuru_kharif_2020_21", "label": "VENURU/Belthangady/Kharif/2020-21"},
    {"key": "24511181", "name": "ogd_sulya_kharif_2020_21", "label": "SULYA/Sullia/Kharif/2020-21"},
    {"key": "24431191", "name": "ogd_uppinangadi_kharif_2021_22", "label": "UPPINANGADI/Puttur/Kharif/2021-22"},
    {"key": "24231181", "name": "ogd_vitla_kharif_2020_21", "label": "VITLA/Bantwal/Kharif/2020-21"},
    {"key": "24421181", "name": "ogd_putturu_kharif_2020_21", "label": "PUTTURU/Puttur/Kharif/2020-21"},
]

def download_resource(key, name, label):
    b64 = base64.b64encode(key.encode()).decode()
    url = "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=" + b64
    json_path = os.path.join(OUTPUT_DIR, name + ".json")
    csv_path = os.path.join(OUTPUT_DIR, name + ".csv")
    
    print("  Downloading: {} -> {}".format(label, name))
    try:
        # Download raw text
        r = subprocess.run(
            ["curl", "-s", "-m", "120", "-o", json_path, url],
            capture_output=True, text=True, timeout=150
        )
        
        # Check file size
        size = os.path.getsize(json_path) if os.path.exists(json_path) else 0
        print("    Downloaded: {} bytes".format(size))
        
        if size < 100:
            print("    TOO SMALL - skipping")
            return None
        
        # Try to parse
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            print("    JSON parse error (truncated response) - saving raw")
            # Try to fix truncated JSON
            # Count approximate records
            import re
            count_estimate = raw.count('"Survey_id"')
            print("    Estimated records: ~{}".format(count_estimate))
            return {"name": name, "label": label, "size": size, "truncated": True, "est_count": count_estimate}
        
        if isinstance(data, list) and len(data) > 0:
            print("    Parsed: {} records".format(len(data)))
            crops = {}
            for row in data:
                c = row.get("Cropname", "?")
                crops[c] = crops.get(c, 0) + 1
            print("    Crop distribution (top 15):")
            for crop, count in sorted(crops.items(), key=lambda x: -x[1])[:15]:
                print("      {}: {}".format(crop, count))
            
            # Write CSV
            if data:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    import csv
                    w = csv.DictWriter(f, fieldnames=data[0].keys())
                    w.writeheader()
                    w.writerows(data)
                print("    CSV: {}".format(csv_path))
            
            return {"name": name, "label": label, "count": len(data), "crops": crops, "truncated": False}
        
    except Exception as e:
        print("    Error: {}".format(str(e)[:80]))
        return None

print("=" * 70)
print("DOWNLOADING HIGH-VALUE HOBLI RESOURCES")
print("=" * 70)

results = []
for res in RESOURCES:
    result = download_resource(res["key"], res["name"], res["label"])
    if result:
        results.append(result)
    time.sleep(2)

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)
for r in results:
    print("  {}: {}".format(r["label"], r.get("count", r.get("est_count", "?"))))

# Save summary
with open(os.path.join(OUTPUT_DIR, "r5_2_6_download_summary.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved summary.")
