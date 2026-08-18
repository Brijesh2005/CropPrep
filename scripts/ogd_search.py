"""
Use datagovindia to search for and download DK Crop Survey datasets.
"""
import os
import sys
import json
import time
import pandas as pd
from datagovindia import DataGovIndia

API_KEY = "579b464db66ec23bdd000001a7bbca880cfc4e2f728566029e246b63"
OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize
print("Initializing DataGovIndia client...")
datagovin = DataGovIndia(api_key=API_KEY)

# Step 1: Search for DK crop survey resources
print("\n=== Searching for 'crop survey dakshina kannada' ===")
try:
    search_results = datagovin.search("crop survey dakshina kannada")
    print(f"Found {len(search_results)} results")
    if len(search_results) > 0:
        print(search_results.columns.tolist())
        print(search_results.to_string(max_rows=50, max_colwidth=80))
except Exception as e:
    print(f"Search error: {e}")

print("\n=== Searching for 'crop survey karnataka dakshina' ===")
try:
    search_results2 = datagovin.search("crop survey karnataka dakshina")
    print(f"Found {len(search_results2)} results")
    if len(search_results2) > 0:
        print(search_results2.to_string(max_rows=50, max_colwidth=80))
except Exception as e:
    print(f"Search error: {e}")

print("\n=== Searching for 'crop survey karnataka' ===")
try:
    search_results3 = datagovin.search("crop survey karnataka")
    print(f"Found {len(search_results3)} results")
    if len(search_results3) > 0:
        print(search_results3.to_string(max_rows=100, max_colwidth=80))
except Exception as e:
    print(f"Search error: {e}")
