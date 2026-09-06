import json, re

arr = json.load(open('artifacts/r5_5_pull/logs_v4.json', encoding='utf-8'))
lines = []
for e in arr:
    lines.append((e.get('time', 0), e.get('stream_name', '?'), e.get('data', '')))

# 1. All training epoch rows
print("=== training epoch rows (v4) ===")
for t, s, d in lines:
    if '[training] epoch' in d:
        print("[%8.2fs] %s" % (t, d), end='')

# 2. Variant separators
print("\n=== variant markers ===")
for t, s, d in lines:
    if 'binary variant' in d or 'Phase 3' in d or 'Phase 5/9' in d or 'Phase 10' in d or 'Phase 11' in d or 'Phases 12-14' in d:
        print("[%8.2fs] %s" % (t, d), end='')

# 3. Class separator / normalization block between 547 and 989 (image stats printed)
print("\n=== image stats block (phase 5/9) ===")
capture = False
depth = 0
buf = []
for t, s, d in lines:
    if '=== Phase 5/9' in d:
        capture = True
        continue
    if '=== Phase 3' in d:
        break
    if capture:
        buf.append(d.strip())
# re-parse the likely JSON object boundaries
text = "\n".join(buf)
# find first '{' and last '}'
i = text.find('{')
j = text.rfind('}')
if i >= 0 and j > i:
    frag = text[i:j+1]
    try:
        obj = json.loads(frag)
        print(json.dumps(obj, indent=2))
    except Exception as ex:
        print("parse failed:", ex)
        print(frag[:3000])