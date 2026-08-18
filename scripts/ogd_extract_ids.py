"""
Minimal OGD resource ID extraction using requests with very short timeouts.
"""
import re
import json
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"

# Try the Karnataka portal resource page
url = "https://karnataka.data.gov.in/resource/crop-survey-mulki-hobli-mangalore-taluk-dakshina-kannada-district-karnataka-kharif-2"
print(f"Fetching: {url}")

try:
    resp = requests.get(url, timeout=10, verify=False, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Status: {resp.status_code}")
    content = resp.text
    print(f"Length: {len(content)}")
    
    # Find all UUIDs
    uuids = re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', content)
    print(f"UUIDs found: {uuids}")
    
    # Find resource_id patterns
    res_ids = re.findall(r'resource_id[=:]["\']?([0-9a-f-]{36})', content)
    print(f"resource_id matches: {res_ids}")
    
    # Find API URLs
    api_urls = re.findall(r'(api/datastore[^"\'<>\s]+)', content)
    print(f"API URLs: {api_urls}")
    
    # Find catalog references
    catalogs = re.findall(r'catalog[=:]["\']?([0-9a-f-]{36})', content)
    print(f"Catalog UUIDs: {catalogs}")
    
    # Try to find JSON data
    nuxt = re.findall(r'window\.__NUXT__\s*=\s*(\{.*?\})\s*;?\s*<', content[:100000], re.S)
    if nuxt:
        print(f"Found NUXT data: {nuxt[0][:500]}")
    
    # Find any download URLs
    downloads = re.findall(r'href=["\']([^"\']*\.csv[^"\']*)', content, re.I)
    print(f"CSV download links: {downloads}")
    
    # Find JSON endpoints
    json_eps = re.findall(r'["\']([^"\']*json[^"\']*)["\']', content[:50000])
    print(f"JSON endpoints: {[x for x in json_eps if 'api' in x.lower() or 'resource' in x.lower()][:10]}")
    
except Exception as e:
    print(f"Error: {e}")

# Also try the main data.gov.in API with the correct base URL pattern
print("\n=== Trying direct datastore API ===")
# The correct pattern from the documentation
test_url = f"https://data.gov.in/api/datastore/resource.json?resource_id=579b464d-66ec-23bd-d000-001a7bbca880&api-key={API_KEY}&limit=1"
try:
    resp = requests.get(test_url, timeout=10, verify=False)
    print(f"Direct API test: {resp.status_code}")
    if resp.status_code == 200:
        print(resp.text[:500])
except Exception as e:
    print(f"Error: {e}")
