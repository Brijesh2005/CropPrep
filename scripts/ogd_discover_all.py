"""
Phase 2: Parse remaining downloads + discover ALL hobli resources.
"""
import subprocess
import json
import os
import csv
import re
import time

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

def download_with_curl(url, output_path, timeout=120):
    cmd = ["curl", "-s", "-L", "--max-time", str(timeout), "-o", output_path, url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 10)
    if os.path.exists(output_path):
        return os.path.getsize(output_path)
    return 0

# Parse the 2 newly downloaded JSON files
print("=== Parsing newly downloaded JSON files ===")
for f in os.listdir(OUTPUT_DIR):
    if f.startswith("ogd_discovered_") and f.endswith(".json"):
        path = os.path.join(OUTPUT_DIR, f)
        csv_path = path.replace(".json", ".csv")
        if not os.path.exists(csv_path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list) and len(data) > 0:
                    print(f"\n{f}: {len(data)} records")
                    print(f"  First record: {json.dumps(data[0], indent=2)[:500]}")
                    crops = {}
                    for row in data:
                        crop = row.get("Cropname", "UNKNOWN")
                        crops[crop] = crops.get(crop, 0) + 1
                    print(f"  Unique crops: {len(crops)}")
                    for crop, count in sorted(crops.items(), key=lambda x: -x[1])[:10]:
                        print(f"    {crop}: {count}")
                    # Save CSV
                    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
                    print(f"  CSV: {csv_path}")
            except Exception as e:
                print(f"  Error: {e}")

# Now discover more resources
# The key pattern: MjQzMTExODE= (base64 of 24311181), MjQzMjExODE= (24321181), etc.
# Let's try to enumerate more keys
# Pattern seems to be: 24X11181 where X varies
# MjQzMTExODE= -> 24311181 (Beltangadi)
# MjQzMjExODE= -> 24321181 (Kokkada)
# MjQxNDExOTE= -> 24141191 (Mulki)
# MjQ1MjExODE= -> 24521181 (Panja)

# Let's also try to scrape more resource pages from data.gov.in
# to find all hoblis in each catalog
print("\n" + "=" * 60)
print("Discovering more hobli resources")
print("=" * 60)

# Try scraping catalog pages for resource lists
# The catalog pages use client-side rendering, so we need to find
# the API that loads the resource list
# Try the backend API for catalog resources
CATALOG_APIS = [
    "https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/catalog/5cb603d3-fa1d-488b-be41-894a310e0a0b/field_resources?api-key=579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63",
    "https://www.data.gov.in/backend/dms/v1/ogdpv2/catalog/5cb603d3-fa1d-488b-be41-894a310e0a0b/field_resources?api-key=579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63",
    "https://karnataka.data.gov.in/api/datastore/resource.json?resource_id=5cb603d3-fa1d-488b-be41-894a310e0a0b&api-key=579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63",
]

for api_url in CATALOG_APIS:
    print(f"\nTrying: {api_url[:100]}...")
    tmp = os.path.join(OUTPUT_DIR, "_tmp_api.json")
    size = download_with_curl(api_url, tmp, timeout=30)
    if size > 0:
        try:
            with open(tmp, "r") as f:
                data = json.load(f)
            print(f"  Response: {json.dumps(data)[:500]}")
        except:
            with open(tmp, "r", errors="replace") as f:
                print(f"  Raw: {f.read()[:500]}")
    else:
        print(f"  Empty response")
    os.remove(tmp) if os.path.exists(tmp) else None

# Try to enumerate hoblis by guessing resource page URLs
# Known hoblis in Dakshina Kannada:
HOBLIS = [
    "beltangadi", "belthangady", "mulki", "mangalore", "sullia", "sulia",
    "puttur", "bantval", "bantwa", "belthangady", "beltangadi",
    "kokkada", "panja", "madikeri", "kushalnagar", "somwarpet",
    "virajpet", "fraser-pet", "ponnampet", "kundapura", "karkala",
    " Moodbidri", "manjeshwar", "kanhangad", "puttur", "belthangady",
    "bantwala", "surathkal", "panambur", "tagore-garden",
]

# Let's try a different approach: search for all resource pages
# in the Kharif 2020-21 catalog by fetching the catalog page
# and looking for resource links in the HTML
print("\nSearching for resource links in catalog pages...")

CATALOG_PAGES = [
    "https://karnataka.data.gov.in/catalog/crop-survey-dakshina-kannada-district-karnataka-kharif-season-2020-21",
    "https://www.data.gov.in/catalog/crop-survey-dakshina-kannada-district-karnataka-kharif-season-2020-21",
    "https://karnataka.data.gov.in/catalog/crop-survey-dakshina-kannada-district-karnataka-kharif-season-2021-2022",
]

all_resource_urls = set()
for catalog_url in CATALOG_PAGES:
    print(f"\nFetching catalog: {catalog_url.split('/')[-1][:60]}...")
    tmp = os.path.join(OUTPUT_DIR, "_tmp_catalog.html")
    size = download_with_curl(catalog_url, tmp, timeout=30)
    if size > 0:
        with open(tmp, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        # Find resource links
        links = re.findall(r'href="(/resource/[^"]+)"', html)
        all_resource_urls.update(links)
        # Also look for resource URLs in NUXT data
        nuxt_urls = re.findall(r'"(/resource/[^"]+)"', html)
        all_resource_urls.update(nuxt_urls)
        print(f"  Found {len(links)} resource links")
    os.remove(tmp) if os.path.exists(tmp) else None
    time.sleep(1)

print(f"\nTotal unique resource URLs found: {len(all_resource_urls)}")
for url in sorted(all_resource_urls):
    print(f"  {url}")

# Now fetch each resource page to extract API keys
print("\n" + "=" * 60)
print("Extracting API keys from resource pages")
print("=" * 60)

discovered_keys = {}
for res_url in sorted(all_resource_urls):
    full_url = f"https://karnataka.data.gov.in{res_url}"
    print(f"\nFetching: {res_url[:60]}...")
    tmp = os.path.join(OUTPUT_DIR, "_tmp_res.html")
    size = download_with_curl(full_url, tmp, timeout=30)
    if size > 0:
        with open(tmp, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        urls = re.findall(r'field_datafile_url:"([^"]+)"', html)
        if urls:
            api_url = urls[0].replace("\\u002F", "/")
            key_match = re.search(r'key=([^"&]+)', api_url)
            if key_match:
                key = key_match.group(1)
                discovered_keys[key] = api_url
                print(f"  Key: {key}")
    os.remove(tmp) if os.path.exists(tmp) else None
    time.sleep(1)

print(f"\nTotal API keys discovered: {len(discovered_keys)}")
for key, url in discovered_keys.items():
    print(f"  {key}: {url[:80]}...")
