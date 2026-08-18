"""
R5.2.6 PART A: Discover additional DK hobli resources via targeted brute-force.

Pattern: base64(24 + hobli_code(2) + 11 + season_code(2))
Known: 11=MangaluruA, 14=Mulki, 22=Panemangaluru, 31=Beltangadi, 32=Kokkada, 52=Panja

Try codes 10-60 with suffixes 81,91,82,92 to find new hoblis.
"""
import subprocess, base64, json, time, os

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KNOWN_CODES = {11, 14, 22, 31, 32, 52}
SUFFIXES = [81, 91, 82, 92]  # season+year combinations

def test_key(numeric_str):
    """Test if a numeric key returns valid data."""
    b64 = base64.b64encode(numeric_str.encode()).decode()
    url = f"https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key={b64}"
    try:
        result = subprocess.run(
            ["curl", "-s", "-m", "12", url],
            capture_output=True, text=True, timeout=15
        )
        data = json.loads(result.stdout)
        if isinstance(data, list) and len(data) > 10:
            hobli = data[0].get("Hobli_Name", "?")
            taluk = data[0].get("Taluk_Name", "?")
            district = data[0].get("District_Name", "?")
            season = data[0].get("Season", "?")
            years = data[0].get("Years", "?")
            crops = set(r.get("Cropname", "") for r in data[:500])
            return {
                "key": numeric_str,
                "b64": b64,
                "hobli": hobli,
                "taluk": taluk,
                "district": district,
                "season": season,
                "years": years,
                "count": len(data),
                "sample_crops": list(crops)[:10],
            }
    except:
        pass
    return None

# Generate candidates
candidates = []
for code in range(10, 61):
    if code in KNOWN_CODES:
        continue  # skip already-downloaded
    for suffix in SUFFIXES:
        key = f"24{code:02d}11{suffix}"
        candidates.append(key)

print(f"Testing {len(candidates)} candidate keys...")
print(f"Known codes (skipped): {sorted(KNOWN_CODES)}")

discovered = []
for i, key in enumerate(candidates):
    result = test_key(key)
    if result:
        dk = "dakshina" in result["district"].lower() or "kannada" in result["district"].lower()
        print(f"  [{i+1}/{len(candidates)}] {key}: {result['hobli']}/{result['taluk']}/{result['district']} "
              f"({result['season']}/{result['years']}) = {result['count']} records "
              f"{'DK!' if dk else 'NOT-DK'}")
        if dk:
            discovered.append(result)
    else:
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(candidates)}] testing...")
    time.sleep(0.3)

print(f"\nDiscovered DK resources: {len(discovered)}")
for d in discovered:
    print(f"  {d['key']} -> {d['hobli']}/{d['taluk']} ({d['season']}/{d['years']}) = {d['count']} records")
    print(f"    Sample crops: {d['sample_crops']}")

# Save discovery results
with open(os.path.join(OUTPUT_DIR, "hobli_discovery_r5_2_6.json"), "w", encoding="utf-8") as f:
    json.dump(discovered, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUTPUT_DIR}/hobli_discovery_r5_2_6.json")
