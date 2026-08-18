"""
Download all Dakshina Kannada crop survey OGD data using curl.
Curl works when Python requests times out.
"""
import subprocess
import json
import os
import csv
import time

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Known API URLs from extracted NUXT data
# Each hobli resource has a unique base64-encoded key
RESOURCES = {
    "beltangadi_kharif_2020_21": {
        "url": "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=MjQzMTExODE=",
        "hobli": "Beltangadi", "taluk": "Belthangady", "season": "Kharif", "year": "2020-2021",
    },
    "mulki_kharif_2021_22": {
        "url": "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=MjQxNDExOTE=",
        "hobli": "Mulki", "taluk": "Mangalore", "season": "Kharif", "year": "2021-2022",
    },
}

def download_with_curl(url, output_path, timeout=120):
    """Download file using curl."""
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout), "-o", output_path, url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        return size
    return 0

print("=" * 60)
print("Downloading Dakshina Kannada Crop Survey Data")
print("=" * 60)

# Download each known resource
for name, info in RESOURCES.items():
    json_path = os.path.join(OUTPUT_DIR, f"ogd_{name}.json")
    print(f"\nDownloading: {name}")
    print(f"  URL: {info['url']}")
    
    size = download_with_curl(info["url"], json_path, timeout=120)
    print(f"  Downloaded: {size:,} bytes")
    
    if size > 0:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"  Records: {len(data)}")
                if len(data) > 0:
                    print(f"  Keys: {list(data[0].keys())}")
                    # Save as CSV too
                    csv_path = os.path.join(OUTPUT_DIR, f"ogd_{name}.csv")
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    print(f"  CSV saved: {csv_path}")
                    
                    # Print crop distribution
                    crops = {}
                    for row in data:
                        crop = row.get("Cropname", "UNKNOWN")
                        crops[crop] = crops.get(crop, 0) + 1
                    print(f"  Unique crops: {len(crops)}")
                    for crop, count in sorted(crops.items(), key=lambda x: -x[1])[:10]:
                        print(f"    {crop}: {count}")
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
    
    time.sleep(1)

# Now try to discover more resources by scraping catalog pages
# We know the catalog UUIDs for different seasons
CATALOGS = {
    "kharif_2020_21": "5cb603d3-fa1d-488b-be41-894a310e0a0b",
    "rabi_2019_20": None,  # Need to find
    "kharif_2018_19": None,  # Need to find
    "rabi_2021_22": None,  # Need to find
    "kharif_2021_22": "8dabb741-c498-4d34-b284-f6c5aebda7d3",
    "kharif_2020_21_alt": None,
}

# Try to scrape resource pages from the catalog to find more API keys
# Use curl to fetch resource pages and extract field_datafile_url
print("\n" + "=" * 60)
print("Discovering more resources from catalog pages")
print("=" * 60)

# Known resource page URLs from web search
RESOURCE_PAGES = [
    "https://karnataka.data.gov.in/resource/crop-survey-beltangadi-hobli-belthangady-taluk-dakshina-kannada-district-karnataka-kharif",
    "https://karnataka.data.gov.in/resource/crop-survey-mulki-hobli-mangalore-taluk-dakshina-kannada-district-karnataka-kharif-2",
    "https://www.data.gov.in/resource/crop-survey-kokkada-hobli-belthangady-taluk-dakshina-kannada-district-karnataka-kharif",
    "https://www.data.gov.in/resource/crop-survey-panja-hobli-sullia-taluk-dakshina-kannada-district-karnataka-kharif-season",
]

import re

discovered_urls = {}
for page_url in RESOURCE_PAGES:
    print(f"\nScraping: {page_url.split('/')[-1][:60]}...")
    tmp_path = os.path.join(OUTPUT_DIR, "_tmp_page.html")
    size = download_with_curl(page_url, tmp_path, timeout=30)
    if size > 0:
        with open(tmp_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        # Extract field_datafile_url from NUXT data
        matches = re.findall(r'field_datafile_url:"([^"]+)"', html)
        if matches:
            url = matches[0].replace("\\u002F", "/")
            print(f"  Found API URL: {url}")
            # Extract key
            key_match = re.search(r'key=([^"&]+)', url)
            if key_match:
                key = key_match.group(1)
                discovered_urls[key] = url
        else:
            print(f"  No API URL found (size: {size:,})")
    os.remove(tmp_path) if os.path.exists(tmp_path) else None
    time.sleep(1)

# Download any newly discovered URLs
print("\n" + "=" * 60)
print("Downloading newly discovered resources")
print("=" * 60)

for key, url in discovered_urls.items():
    if key not in [r["url"].split("key=")[-1] for r in RESOURCES.values()]:
        name = f"ogd_discovered_{key[:10]}"
        json_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        print(f"\nDownloading: {key}")
        size = download_with_curl(url, json_path, timeout=120)
        print(f"  Downloaded: {size:,} bytes")
        if size > 0:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    print(f"  Records: {len(data)}")
                    if len(data) > 0:
                        csv_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                        print(f"  CSV saved")
            except:
                pass

# Summary
print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("ogd_") and not f.startswith("_"):
        path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")
