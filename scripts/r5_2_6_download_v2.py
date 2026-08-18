"""
Download remaining resources: truncated ones (longer timeout) + Rabi variants.
Focus on resources that might have coffee/cardamom/blackgram.
"""
import subprocess, base64, json, time, os, csv

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

# Truncated resources with longer timeout
RESOURCES = [
    # Truncated - try with longer timeout
    {"key": "24121181", "name": "ogd_mangaluru_b_kharif_2020_21", "label": "MANGALURU B/Mangalore/Kharif/2020-21"},
    {"key": "24131181", "name": "ogd_suratkal_kharif_2020_21", "label": "SURATKAL/Mangalore/Kharif/2020-21"},
    {"key": "24211181", "name": "ogd_bantvala_kharif_2020_21", "label": "BANTVALA/Bantwal/Kharif/2020-21"},
    # Rabi variants for diversity
    {"key": "24331192", "name": "ogd_venuru_rabi_2021_22", "label": "VENURU/Belthangady/Rabi/2021-22"},
    {"key": "24511192", "name": "ogd_sulya_rabi_2021_22", "label": "SULYA/Sullia/Rabi/2021-22"},
    {"key": "24421192", "name": "ogd_putturu_rabi_2021_22", "label": "PUTTURU/Puttur/Rabi/2021-22"},
]

def download_resource(key, name, label):
    b64 = base64.b64encode(key.encode()).decode()
    url = "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=" + b64
    json_path = os.path.join(OUTPUT_DIR, name + ".json")
    csv_path = os.path.join(OUTPUT_DIR, name + ".csv")
    
    print("  Downloading: {}".format(label))
    try:
        r = subprocess.run(
            ["curl", "-s", "-m", "180", "-o", json_path, url],
            capture_output=True, text=True, timeout=210
        )
        
        size = os.path.getsize(json_path) if os.path.exists(json_path) else 0
        print("    Size: {} bytes".format(size))
        
        if size < 100:
            print("    EMPTY - skipping")
            return None
        
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read()
        
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            import re
            count_est = raw.count('"Survey_id"')
            print("    TRUNCATED - ~{} records".format(count_est))
            return {"name": name, "label": label, "size": size, "truncated": True, "est_count": count_est}
        
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
                    w = csv.DictWriter(f, fieldnames=data[0].keys())
                    w.writeheader()
                    w.writerows(data)
            
            return {"name": name, "label": label, "count": len(data), "crops": crops, "truncated": False}
        
    except Exception as e:
        print("    Error: {}".format(str(e)[:80]))
        return None

print("=" * 70)
print("DOWNLOADING REMAINING RESOURCES")
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
    status = "TRUNCATED" if r.get("truncated") else r.get("count", "?")
    print("  {}: {}".format(r["label"], status))

with open(os.path.join(OUTPUT_DIR, "r5_2_6_download_summary_v2.json"), "w") as f:
    json.dump(results, f, indent=2, default=str)
