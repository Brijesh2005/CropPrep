"""
Download crop survey data from the Karnataka government API.
"""
import json
import os
import csv
import time
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

session = requests.Session()
session.verify = False
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

# The actual data API endpoint discovered from the OGD page
# Base URL pattern: https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata
# Key is base64 encoded - different keys for different resources

# Known keys from the NUXT data
API_KEYS = {
    "kharif_2021_22_mulki": "MjQxNDExOTE=",
}

# Try to access the API
print("=" * 60)
print("Downloading crop survey data from Karnataka API")
print("=" * 60)

for name, key in API_KEYS.items():
    url = f"https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key={key}"
    print(f"\nFetching: {name}")
    print(f"URL: {url}")
    
    try:
        resp = session.get(url, timeout=30)
        print(f"Status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
        print(f"Size: {len(resp.content)} bytes")
        
        if resp.status_code == 200:
            # Try to parse as JSON
            try:
                data = resp.json()
                if isinstance(data, list):
                    print(f"Records: {len(data)}")
                    if len(data) > 0:
                        print(f"First record keys: {list(data[0].keys())}")
                        print(f"First record: {json.dumps(data[0], indent=2)}")
                        
                        # Save to CSV
                        csv_path = os.path.join(OUTPUT_DIR, f"ogd_{name}.csv")
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=data[0].keys())
                            writer.writeheader()
                            writer.writerows(data)
                        print(f"Saved to: {csv_path}")
                        
                        # Save raw JSON too
                        json_path = os.path.join(OUTPUT_DIR, f"ogd_{name}.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Saved to: {json_path}")
                elif isinstance(data, dict):
                    print(f"Dict keys: {list(data.keys())}")
                    # Check if data is nested
                    for k, v in data.items():
                        if isinstance(v, list):
                            print(f"  {k}: {len(v)} items")
                            if len(v) > 0:
                                print(f"  First item: {json.dumps(v[0], indent=2)[:500]}")
                        else:
                            print(f"  {k}: {str(v)[:200]}")
            except json.JSONDecodeError:
                print(f"Not JSON. First 500 chars: {resp.text[:500]}")
        else:
            print(f"Error body: {resp.text[:500]}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)

# Also try the catalog-level API to get all hoblis
print("\n" + "=" * 60)
print("Trying catalog-level API for all hoblis")
print("=" * 60)

# The catalog UUID: 8dabb741-c498-4d34-b284-f6c5aebda7d3
# Try fetching the catalog to get all resource keys
catalog_url = "https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/catalog/8dabb741-c498-4d34-b284-f6c5aebda7d3"
print(f"Fetching catalog: {catalog_url}")
try:
    resp = session.get(catalog_url, timeout=15)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Response: {resp.text[:2000]}")
except Exception as e:
    print(f"Error: {e}")

# Try the backend API for resources of this catalog
resources_url = "https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/resource"
params = {"catalog": "8dabb741-c498-4d34-b284-f6c5aebda7d3", "api-key": "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"}
print(f"\nFetching resources: {resources_url}")
try:
    resp = session.get(resources_url, params=params, timeout=15)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Response: {resp.text[:2000]}")
except Exception as e:
    print(f"Error: {e}")
