import subprocess, base64, json, time

targets = [
    '24121181', '24131181', '24211181', '24231181',
    '24331181', '24341181', '24411181', '24421181',
    '24511181', '24531181', '24611181', '24621181',
    '24121191', '24131191', '24211191', '24231191',
    '24331191', '24341191', '24411191', '24421191',
    '24511191', '24531191', '24611191', '24621191',
    '24411182', '24421182', '24331182', '24341182',
    '24411192', '24421192', '24331192', '24341192',
    '24511182', '24531182', '24611182', '24621182',
    '24121182', '24131182', '24211182', '24231182',
]

print("Testing {} targeted keys...".format(len(targets)))
found = []
for key in targets:
    b64 = base64.b64encode(key.encode()).decode()
    url = "https://kodi.karnataka.gov.in/Crop_Survey/api/CropSurvey/Getdata?key=" + b64
    try:
        r = subprocess.run(["curl", "-s", "-m", "10", url], capture_output=True, text=True, timeout=12)
        data = json.loads(r.stdout)
        if isinstance(data, list) and len(data) > 10:
            h = data[0]
            dk = "dakshina" in h.get("District_Name", "").lower()
            crops = list(set(x.get("Cropname", "") for x in data[:200]))[:15]
            info = {
                "key": key, "hobli": h.get("Hobli_Name", ""),
                "taluk": h.get("Taluk_Name", ""), "district": h.get("District_Name", ""),
                "season": h.get("Season", ""), "years": h.get("Years", ""),
                "count": len(data), "crops": crops,
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
            print("  Empty: {}".format(key))
    except Exception as e:
        print("  Error: {}: {}".format(key, str(e)[:50]))
    time.sleep(0.5)

print("\nDK resources found: {}".format(len(found)))
with open("D:/CropPrep/govt_crop_survey_data/hobli_discovery_r5_2_6.json", "w") as f:
    json.dump(found, f, indent=2)
print("Saved.")
