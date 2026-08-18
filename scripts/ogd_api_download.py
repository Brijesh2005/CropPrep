"""
OGD API — targeted download attempt for DK Crop Survey.
Try the known catalog resource IDs from the web portal.
"""
import json
import os
import urllib.request
import urllib.parse
import ssl
import time
import sys

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"
OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def try_url(url, timeout=15):
    """Try a URL, return (status, data_or_error)."""
    print(f"  {url[:100]}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, raw
    except urllib.error.HTTPError as e:
        return e.code, str(e.reason).encode()
    except Exception as e:
        return 0, str(e).encode()


# ---- Step 1: Try to list all resources for a known catalog slug ----
print("=" * 60)
print("Step 1: Try resource listing endpoints")
print("=" * 60)

endpoints = [
    # data.gov.in API v1 patterns
    f"https://data.gov.in/backend/dms/v1/ogdpv2/resource?api-key={API_KEY}&limit=5",
    # Direct catalog resources
    f"https://data.gov.in/backend/dms/v1/ogdpv2/catalog/crop-survey-dakshina-kannada-district-karnataka-kharif-season-2020-21?api-key={API_KEY}",
    # Karnataka portal
    f"https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/resource?api-key={API_KEY}&limit=5",
    # Try the catalog API
    f"https://data.gov.in/backend/dms/v1/ogdpv2/catalog?api-key={API_KEY}&limit=5",
    # Try resource download API pattern (v3)
    f"https://data.gov.in/backend/dms/v3/ogdpv2/resource?api-key={API_KEY}&limit=5",
]

for url in endpoints:
    status, data = try_url(url)
    print(f"    Status: {status}, Size: {len(data)} bytes")
    if status == 200 and len(data) > 10:
        text = data.decode("utf-8", errors="replace")[:1000]
        print(f"    Preview: {text}")
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                print(f"    Keys: {list(parsed.keys())[:10]}")
        except:
            pass
    print()
    time.sleep(0.5)

# ---- Step 2: Try to get resources by catalog title search ----
print("=" * 60)
print("Step 2: Try search endpoints")
print("=" * 60)

search_endpoints = [
    f"https://data.gov.in/backend/dms/v1/ogdpv2/search?api-key={API_KEY}&keyword=crop+survey+dakshina+kannada",
    f"https://data.gov.in/backend/dms/v1/ogdpv2/resource?api-key={API_KEY}&filters%5Btitle%5D=crop+survey+dakshina+kannada",
    f"https://data.gov.in/backend/dms/v1/ogdpv2/resource?api-key={API_KEY}&query=crop+survey+dakshina+kannada",
]

for url in search_endpoints:
    status, data = try_url(url)
    print(f"    Status: {status}, Size: {len(data)} bytes")
    if status == 200 and len(data) > 10:
        text = data.decode("utf-8", errors="replace")[:2000]
        print(f"    Preview: {text}")
    print()
    time.sleep(0.5)

# ---- Step 3: Try the Karnataka state portal resource API ----
print("=" * 60)
print("Step 3: Try Karnataka portal")
print("=" * 60)

karnataka_endpoints = [
    f"https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/resource?api-key={API_KEY}&limit=10",
    f"https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/catalog?api-key={API_KEY}&limit=10",
    f"https://karnataka.data.gov.in/backend/dms/v1/ogdpv2/resource/crop-survey-mulki-hobli-mangalore-taluk-dakshina-kannada-district-karnataka-kharif-2?api-key={API_KEY}",
]

for url in karnataka_endpoints:
    status, data = try_url(url)
    print(f"    Status: {status}, Size: {len(data)} bytes")
    if status == 200 and len(data) > 10:
        text = data.decode("utf-8", errors="replace")[:2000]
        print(f"    Preview: {text}")
    print()
    time.sleep(0.5)

print("=" * 60)
print("Done")
print("=" * 60)
