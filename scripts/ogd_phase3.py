"""
Phase 3: Re-download truncated files + find more hoblis via web search.
"""
import subprocess
import json
import os
import csv
import re
import time

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

def curl_download(url, output_path, timeout=180):
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout), "-o", output_path, url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    return os.path.getsize(output_path) if os.path.exists(output_path) else 0

# Re-download the truncated Kokkada and Panja files
print("=== Re-downloading truncated files ===")
resources_to_download = {
    "kokkada_kharif_2020_21": "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=MjQzMjExODE%3D",
    "panja_kharif_2020_21": "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=MjQ1MjExODE%3D",
}

for name, url in resources_to_download.items():
    json_path = os.path.join(OUTPUT_DIR, f"ogd_{name}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
    print(f"\nDownloading: {name}")
    size = curl_download(url, json_path, timeout=180)
    print(f"  Size: {size:,} bytes")
    if size > 0:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"  Records: {len(data)}")
                if data:
                    print(f"  Keys: {list(data[0].keys())}")
                    csv_path = json_path.replace(".json", ".csv")
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    crops = {}
                    for row in data:
                        c = row.get("Cropname", "?")
                        crops[c] = crops.get(c, 0) + 1
                    print(f"  Unique crops: {len(crops)}")
                    for c, n in sorted(crops.items(), key=lambda x: -x[1])[:10]:
                        print(f"    {c}: {n}")
        except Exception as e:
            print(f"  Parse error: {e}")
    time.sleep(1)

# Now try to enumerate keys by brute-force pattern
# Known: MjQzMTExODE= (24311181), MjQzMjExODE= (24321181), MjQxNDExOTE= (24141191), MjQ1MjExODE= (24521181)
# Pattern seems to be: MjQ[X]NjExODE= where X varies
# Let's try more combinations
import base64

print("\n" + "=" * 60)
print("Brute-force key discovery")
print("=" * 60)

# Generate candidate keys based on patterns
# The keys seem to be: "24" + digit(s) + "1181" or "1191"
# Let's try a wider range
candidate_numbers = []
for prefix in range(2410, 2460):
    candidate_numbers.append(str(prefix) + "1181")
    candidate_numbers.append(str(prefix) + "1191")
    candidate_numbers.append(str(prefix) + "1182")
    candidate_numbers.append(str(prefix) + "1192")

# Also try 24X11181 pattern
for x in range(0, 10):
    for suffix in ["1181", "1191", "1182"]:
        candidate_numbers.append(f"24{x}1{suffix}")

# Deduplicate
candidate_numbers = list(set(candidate_numbers))

# Already known good keys (decoded):
known_keys = {"24311181", "24321181", "24141191", "24521181"}
new_keys = [k for k in candidate_numbers if k not in known_keys]

print(f"Testing {len(new_keys)} candidate keys (excluding {len(known_keys)} known)...")

discovered = {}
for num in new_keys[:50]:  # Test first 50
    encoded = base64.b64encode(num.encode()).decode()
    url = f"https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key={encoded}"
    tmp = os.path.join(OUTPUT_DIR, "_tmp_test.json")
    size = curl_download(url, tmp, timeout=15)
    if size > 1000:
        try:
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                h = data[0].get("Hobli_Name", "?")
                t = data[0].get("Taluk_Name", "?")
                s = data[0].get("Season", "?")
                y = data[0].get("Years", "?")
                print(f"  HIT: {num} -> {h}, {t}, {s} {y} ({len(data)} records)")
                discovered[num] = {"key": encoded, "data": data, "hobli": h, "taluk": t}
                # Save immediately
                safe_name = f"ogd_{h.lower().replace(' ', '_')}_{s.lower()}_{y.replace('-','_')}.json"
                out_path = os.path.join(OUTPUT_DIR, safe_name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                csv_path = out_path.replace(".json", ".csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
                print(f"    Saved: {safe_name}")
        except:
            pass
    os.remove(tmp) if os.path.exists(tmp) else None

print(f"\nDiscovered {len(discovered)} new keys")
