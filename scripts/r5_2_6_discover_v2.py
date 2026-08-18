import subprocess, base64, json, time

# Keys that returned large responses (truncated JSON) — might be valid
# Also try more codes for Puttur, Kadaba, Karkala, Vitla
targets = [
    # The ones that had large responses (errors with data)
    '24121181', '24131181', '24211181', '24231181',
    '24331191', '24511181',
    # Try different middle digits for underexplored taluks
    '24411181', '24421181', '24431181', '24441181',
    '24541181', '24551181',
    '24631181', '24641181', '24651181',
    # Rabi variants for underexplored
    '24411191', '24421191', '24431191', '24441191',
    '24541191', '24551191',
    '24631191', '24641191', '24651191',
    # Try with different suffix patterns
    '24331181', '24341181',
    '24521191', '24521192',
    # Try broader range for Puttur/Kadaba
    '24451181', '24461181', '24471181', '24481181',
    '24451191', '24461191', '24471191', '24481191',
]

print("Testing {} keys (larger responses + new patterns)...".format(len(targets)))
found = []
errors_with_data = []

for key in targets:
    b64 = base64.b64encode(key.encode()).decode()
    url = "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=" + b64
    try:
        r = subprocess.run(["curl", "-s", "-m", "15", url], capture_output=True, text=True, timeout=18)
        stdout = r.stdout
        if not stdout or len(stdout) < 50:
            print("  Empty: {}".format(key))
            time.sleep(0.3)
            continue
        
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            # Large response but truncated — try to extract info
            if len(stdout) > 1000:
                # Try to find Hobli_Name in the raw text
                import re
                hobli_match = re.search(r'"Hobli_Name"\s*:\s*"([^"]+)"', stdout)
                taluk_match = re.search(r'"Taluk_Name"\s*:\s*"([^"]+)"', stdout)
                dist_match = re.search(r'"District_Name"\s*:\s*"([^"]+)"', stdout)
                season_match = re.search(r'"Season"\s*:\s*"([^"]+)"', stdout)
                years_match = re.search(r'"Years"\s*:\s*"([^"]+)"', stdout)
                if hobli_match and dist_match:
                    info = {
                        "key": key,
                        "hobli": hobli_match.group(1),
                        "taluk": taluk_match.group(1) if taluk_match else "?",
                        "district": dist_match.group(1),
                        "season": season_match.group(1) if season_match else "?",
                        "years": years_match.group(1) if years_match else "?",
                        "count_estimate": len(stdout) // 200,  # rough estimate
                        "truncated": True,
                        "response_size": len(stdout),
                    }
                    dk = "dakshina" in info["district"].lower()
                    if dk:
                        errors_with_data.append(info)
                        print("  DK(truncated)! {}: {}/{} ({}/{}) ~{} records, {} bytes".format(
                            key, info["hobli"], info["taluk"], info["season"], info["years"],
                            info["count_estimate"], info["response_size"]))
                    else:
                        print("  Non-DK(truncated): {}: {}/{}".format(key, info["hobli"], info["district"]))
                else:
                    print("  Parse error: {} ({} bytes)".format(key, len(stdout)))
            else:
                print("  Small error: {} ({} bytes)".format(key, len(stdout)))
            time.sleep(0.3)
            continue
        
        if isinstance(data, list) and len(data) > 5:
            h = data[0]
            dk = "dakshina" in h.get("District_Name", "").lower()
            crops = list(set(x.get("Cropname", "") for x in data[:500]))[:15]
            info = {
                "key": key, "hobli": h.get("Hobli_Name", ""),
                "taluk": h.get("Taluk_Name", ""), "district": h.get("District_Name", ""),
                "season": h.get("Season", ""), "years": h.get("Years", ""),
                "count": len(data), "crops": crops, "truncated": False,
            }
            if dk:
                found.append(info)
                print("  DK! {}: {}/{} ({}/{}) = {} records".format(
                    key, h.get("Hobli_Name", "?"), h.get("Taluk_Name", "?"),
                    h.get("Season", "?"), h.get("Years", "?"), len(data)))
                print("    Crops: {}".format(crops))
            else:
                print("  Non-DK: {}: {}/{}".format(key, h.get("Hobli_Name", "?"), h.get("District_Name", "?")))
        else:
            print("  Small/empty: {} ({} items)".format(key, len(data) if isinstance(data, list) else 0))
    except Exception as e:
        print("  Error: {}: {}".format(key, str(e)[:60]))
    time.sleep(0.5)

print("\nDK resources found (parsed): {}".format(len(found)))
print("DK resources found (truncated): {}".format(len(errors_with_data)))

all_found = found + errors_with_data
with open("D:/CropPrep/govt_crop_survey_data/hobli_discovery_r5_2_6.json", "w") as f:
    json.dump(all_found, f, indent=2)
print("Saved {} resources.".format(len(all_found)))
