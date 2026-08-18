"""
Direct OGD API access using the datastore endpoint.
We need resource IDs. Let's find them from the web pages.
"""
import json
import os
import requests
import ssl
import time

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"
OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "CropPrep-Audit/1.0"})
session.verify = False

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Known resource page URLs from web search
RESOURCE_PAGES = {
    "Kharif 2020-21 - Mulki": "https://karnataka.data.gov.in/resource/crop-survey-mulki-hobli-mangalore-taluk-dakshina-kannada-district-karnataka-kharif-2",
    "Kharif 2020-21 - Beltangadi": "https://www.data.gov.in/resource/crop-survey-beltangadi-hobli-belthangady-taluk-dakshina-kannada-district-karnataka-kharif",
    "Kharif 2020-21 - Kokkada": "https://www.data.gov.in/resource/crop-survey-kokkada-hobli-belthangady-taluk-dakshina-kannada-district-karnataka-kharif",
    "Kharif 2020-21 - Panja": "https://www.data.gov.in/resource/crop-survey-panja-hobli-sullia-taluk-dakshina-kannada-district-karnataka-kharif-season",
    "Kharif 2019-20 - Mulki": "https://www.data.gov.in/resource/crop-survey-mulki-hobli-mangalore-taluk-dakshina-kannada-district-karnataka-kharif-1",
    "Kharif 2018-19 - Kadaba": "https://www.data.gov.in/resource/crop-survey-kadaba-hobli-kadaba-taluk-dakshina-kannada-district-karnataka-kharif-season-0",
    "Rabi 2021-22 - Venuru": "https://www.data.gov.in/resource/crop-survey-venuru-hobli-belthangady-taluk-dakshina-kannada-district-karnataka-rabi-0",
}

# Step 1: Try to extract resource IDs from the resource pages
print("=" * 60)
print("Step 1: Extracting resource IDs from OGD portal pages")
print("=" * 60)

resource_ids = {}
for name, url in RESOURCE_PAGES.items():
    print(f"\n  Fetching: {name}")
    print(f"  URL: {url}")
    try:
        resp = session.get(url, timeout=15)
        print(f"  Status: {resp.status_code}")
        
        # Look for resource_id in the page content
        content = resp.text
        
        # Try to find datastore API URL pattern
        import re
        # Pattern: resource_id=UUID
        matches = re.findall(r'resource_id[=:]["\']?([0-9a-f-]{36})', content)
        if matches:
            print(f"  Found resource IDs: {matches}")
            resource_ids[name] = matches[0]
        
        # Also look for download links
        download_matches = re.findall(r'href=["\']([^"\']*download[^"\']*)["\']', content, re.I)
        if download_matches:
            print(f"  Found download links: {download_matches[:3]}")
        
        # Look for API URLs
        api_matches = re.findall(r'(datastore/resource\.json[^"\']*)', content)
        if api_matches:
            print(f"  Found API URLs: {api_matches[:3]}")
        
        # Look for JSON data in script tags
        json_matches = re.findall(r'__NUXT_DATA__\s*=\s*(\[.*?\])\s*<', content[:50000], re.S)
        if json_matches:
            try:
                nuxt_data = json.loads(json_matches[0])
                # Search for UUID patterns in the NUXT data
                uuids = [item for item in nuxt_data if isinstance(item, str) and len(item) == 36 and '-' in item]
                if uuids:
                    print(f"  Found UUIDs in NUXT data: {uuids[:5]}")
                    for uid in uuids:
                        if uid not in [v for v in resource_ids.values()]:
                            resource_ids[name] = uid
                            break
            except:
                pass
        
    except Exception as e:
        print(f"  Error: {e}")
    
    time.sleep(1)

print(f"\n\nCollected resource IDs: {resource_ids}")

# Step 2: Try to download data using discovered resource IDs
print("\n" + "=" * 60)
print("Step 2: Downloading data using discovered resource IDs")
print("=" * 60)

for name, res_id in resource_ids.items():
    print(f"\n  Downloading: {name} (ID: {res_id})")
    
    # Try the datastore API
    api_url = f"https://data.gov.in/api/datastore/resource.json?resource_id={res_id}&api-key={API_KEY}&limit=5"
    try:
        resp = session.get(api_url, timeout=15)
        print(f"    Datastore API status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Keys: {list(data.keys())}")
            if 'records' in data:
                print(f"    Records count: {len(data['records'])}")
                print(f"    First record: {json.dumps(data['records'][0], indent=2)[:500]}")
            elif 'data' in data:
                print(f"    Data: {json.dumps(data['data'], indent=2)[:500]}")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Also try the Karnataka portal API
    karnataka_url = f"https://karnataka.data.gov.in/api/datastore/resource.json?resource_id={res_id}&api-key={API_KEY}&limit=5"
    try:
        resp = session.get(karnataka_url, timeout=15)
        print(f"    Karnataka API status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"    Keys: {list(data.keys())}")
    except Exception as e:
        print(f"    Karnataka error: {e}")
    
    time.sleep(1)

# Step 3: Try the sync-metadata approach
print("\n" + "=" * 60)
print("Step 3: Trying datagovindia sync")
print("=" * 60)

try:
    from datagovindia import DataGovIndia
    datagovin = DataGovIndia(api_key=API_KEY)
    print("  Syncing metadata (this may take a while)...")
    datagovin.sync_metadata()
    print("  Sync complete!")
    
    # Now search
    results = datagovin.search("crop survey dakshina kannada")
    print(f"  Search results: {len(results)}")
    if len(results) > 0:
        print(results.to_string())
except Exception as e:
    print(f"  Sync error: {e}")
