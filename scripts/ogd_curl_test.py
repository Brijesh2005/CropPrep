"""
Try OGD datastore API with very long timeout and retry.
Use curl as a fallback since requests times out.
"""
import subprocess
import json
import os

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The standard OGD API endpoint
# We need resource_id - try the node UUIDs we discovered
API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"

# Try curl which might handle connection differently
urls = [
    f"https://data.gov.in/api/datastore/resource.json?resource_id=22f72e55-b37f-4075-bb3d-5c8cbdd4e7c9&api-key={API_KEY}&limit=5",
    f"https://data.gov.in/api/datastore/resource.json?resource_id=1f292b23-98d1-4a19-a3e4-2409d4688656&api-key={API_KEY}&limit=5",
    f"https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=MjQzMTExODE%3D",
]

for i, url in enumerate(urls):
    print(f"\n=== URL {i+1} ===")
    print(f"URL: {url[:120]}...")
    try:
        result = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "--max-time", "30", "-L", url],
            capture_output=True, text=True, timeout=45
        )
        lines = result.stdout.split('\n')
        status_code = lines[-1] if lines else 'unknown'
        body = '\n'.join(lines[:-1])
        print(f"Status: {status_code}")
        print(f"Body size: {len(body)} bytes")
        if len(body) > 0:
            print(f"Body preview: {body[:500]}")
    except Exception as e:
        print(f"Error: {e}")
