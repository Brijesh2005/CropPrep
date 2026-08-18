"""
OGD API downloader for DK Crop Survey datasets.
Uses the data.gov.in API v3 to search and download resources.
"""
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import ssl
import csv
import io
from collections import Counter

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"
BASE_URL = "https://data.gov.in/backend/dms/v1/ogdpv2"
OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Disable SSL verification for Windows
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def api_get(endpoint, params=None):
    """Make an API GET request."""
    if params is None:
        params = {}
    params["api-key"] = API_KEY
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    print(f"  GET {endpoint} -> ", end="", flush=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"OK ({len(json.dumps(data))} bytes)")
            return data
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}: {e.reason}")
        try:
            body = e.read().decode("utf-8")[:500]
            print(f"    Body: {body}")
        except:
            pass
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def search_catalogs():
    """Search for DK crop survey catalogs."""
    print("\n=== Searching for DK Crop Survey catalogs ===")
    
    # Try the resource search endpoint
    params = {
        "keyword": "crop survey dakshina kannada",
        "limit": 50,
    }
    data = api_get("resource", params)
    if data:
        print(f"  Found resource result: {json.dumps(data, indent=2)[:1000]}")
    
    # Try catalog search
    params = {
        "keyword": "crop survey dakshina kannada karnataka",
        "limit": 50,
    }
    data = api_get("catalog", params)
    if data:
        print(f"  Found catalog result keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        if isinstance(data, dict):
            for key in data:
                val = data[key]
                if isinstance(val, list):
                    print(f"    {key}: {len(val)} items")
                elif isinstance(val, dict):
                    print(f"    {key}: {list(val.keys())[:10]}")
                else:
                    print(f"    {key}: {str(val)[:200]}")
    
    return data


def try_resource_endpoints():
    """Try different resource endpoint patterns."""
    print("\n=== Trying resource endpoints ===")
    
    # Pattern 1: resource/list with filters
    data = api_get("resource/list", {
        "filters[title]": "crop survey dakshina kannada",
        "limit": 50,
    })
    if data:
        print(f"  resource/list result: {json.dumps(data, indent=2)[:2000]}")
    
    # Pattern 2: search
    data = api_get("search", {
        "keyword": "crop survey dakshina kannada",
        "limit": 50,
    })
    if data:
        print(f"  search result keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    
    # Pattern 3: datasets
    data = api_get("datasets", {
        "keyword": "crop survey dakshina kannada",
        "limit": 50,
    })
    if data:
        print(f"  datasets result keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")


def try_known_catalog_ids():
    """Try to access the known catalog IDs from the web search."""
    print("\n=== Trying known catalog IDs ===")
    
    # From web search, we found these catalog slugs
    known_catalogs = [
        "crop-survey-dakshina-kannada-district-karnataka-kharif-season-2020-21",
        "crop-survey-dakshina-kannada-district-karnataka-rabi-season-2019-2020",
        "crop-survey-dakshina-kannada-district-karnataka-kharif-season-2019-2020",
        "crop-survey-dakshina-kannada-district-karnataka-kharif-season-2018-19",
    ]
    
    for slug in known_catalogs:
        # Try to get catalog details
        data = api_get(f"catalog/{slug}", {"api-key": API_KEY})
        if data:
            print(f"  {slug}: {json.dumps(data, indent=2)[:500]}")
        
        # Try resource/list with this catalog
        data = api_get("resource/list", {
            "catalog": slug,
            "limit": 50,
        })
        if data:
            print(f"  {slug} resources: {json.dumps(data, indent=2)[:500]}")


def try_v3_api():
    """Try the v3 API pattern."""
    print("\n=== Trying v3 API pattern ===")
    
    v3_base = "https://data.gov.in/backend/dms/v3/ogdpv2"
    
    # Search for resources
    url = f"{v3_base}/resource?api-key={API_KEY}&keyword=crop+survey+dakshina+kannada&limit=50"
    print(f"  GET {url[:120]}... -> ", end="", flush=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"OK")
            print(f"  Result: {json.dumps(data, indent=2)[:2000]}")
            return data
    except Exception as e:
        print(f"ERROR: {e}")
    
    # Try catalog
    url = f"{v3_base}/catalog?api-key={API_KEY}&keyword=crop+survey+dakshina+kannada&limit=50"
    print(f"  GET catalog -> ", end="", flush=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"OK")
            print(f"  Result: {json.dumps(data, indent=2)[:2000]}")
            return data
    except Exception as e:
        print(f"ERROR: {e}")
    
    return None


def try_backend_search():
    """Try the backend search endpoint."""
    print("\n=== Trying backend search ===")
    
    # The main data.gov.in portal uses a search API
    url = f"https://data.gov.in/backend/dms/v1/ogdpv2/search?api-key={API_KEY}&keyword=crop+survey+dakshina+kannada&filters%5Bsector%5D%5B%5D=Agriculture&page=1&size=20"
    print(f"  GET search -> ", end="", flush=True)
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"OK")
            print(f"  Result keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            print(f"  Result: {json.dumps(data, indent=2)[:3000]}")
            return data
    except urllib.error.HTTPError as e:
        print(f"HTTP {e.code}")
        try:
            body = e.read().decode("utf-8")[:1000]
            print(f"  Body: {body}")
        except:
            pass
    except Exception as e:
        print(f"ERROR: {e}")
    
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("OGD API Discovery for DK Crop Survey Datasets")
    print("=" * 60)
    
    # Test API key validity
    print("\n=== Testing API key ===")
    data = api_get("resource", {"limit": 1})
    
    # Try all endpoint patterns
    search_catalogs()
    try_resource_endpoints()
    try_known_catalog_ids()
    try_v3_api()
    try_backend_search()
    
    print("\n=== Done ===")
