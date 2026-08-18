"""
Try discovered UUIDs as resource IDs with the OGD datastore API.
"""
import json
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"

# Discovered UUIDs from karnataka.data.gov.in page
uuids = [
    "1f292b23-98d1-4a19-a3e4-2409d4688656",
    "8dabb741-c498-4d34-b284-f6c5aebda7d3",
]

# Try each with different API patterns
api_patterns = [
    "https://karnataka.data.gov.in/api/datastore/resource.json?resource_id={id}&api-key={key}&limit=2",
    "https://data.gov.in/api/datastore/resource.json?resource_id={id}&api-key={key}&limit=2",
    "https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/resource/{id}?api-key={key}",
]

for uuid in uuids:
    print(f"\n=== Testing UUID: {uuid} ===")
    for pattern in api_patterns:
        url = pattern.format(id=uuid, key=API_KEY)
        print(f"  URL: {url[:100]}...")
        try:
            resp = requests.get(url, timeout=8, verify=False, headers={"User-Agent": "Mozilla/5.0"})
            print(f"  Status: {resp.status_code}, Size: {len(resp.content)}")
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    print(f"  Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if isinstance(v, list) and len(v) > 0:
                                print(f"  {k}: {len(v)} items, first: {json.dumps(v[0])[:300]}")
                            elif isinstance(v, dict):
                                print(f"  {k}: {list(v.keys())[:5]}")
                            else:
                                print(f"  {k}: {str(v)[:200]}")
                except:
                    preview = resp.text[:500]
                    print(f"  Response: {preview}")
            elif resp.status_code != 0:
                print(f"  Body: {resp.text[:200]}")
        except Exception as e:
            print(f"  Error: {e}")
