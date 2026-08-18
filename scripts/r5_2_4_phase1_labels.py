"""
R5.2.4 — Government Crop Survey Integration and STAM Matching

Phase 1: Generate crop label mapping from real OGD data.
"""
import sys
sys.path.insert(0, "D:/CropPrep")

from collections import Counter
import json
import os

from shared.enums.crop_taxonomy import (
    resolve_all_ogd_labels,
    write_label_mapping_csv,
    resolve_crop_label,
    LabelMatchStatus,
)
from shared.enums import CropType

OUTPUT_DIR = "D:/CropPrep/govt_crop_survey_data"
CROP_PREP = "D:/CropPrep"

# Load real OGD data
all_ogd = []
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.startswith("ogd_") and f.endswith(".json") and not f.startswith("_"):
        path = os.path.join(OUTPUT_DIR, f)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list) and len(data) > 0:
            all_ogd.extend(data)

print(f"Loaded {len(all_ogd):,} OGD records")

# Count unique crop names
ogd_crops = Counter(row.get("Cropname", "?") for row in all_ogd)
print(f"Unique crop names: {len(ogd_crops)}")

# Resolve all labels
print("\n=== Resolving all OGD crop labels ===")
resolutions = resolve_all_ogd_labels(ogd_crops)

# Print results grouped by status
by_status = {}
for r in resolutions:
    by_status.setdefault(r.status, []).append(r)

for status in LabelMatchStatus:
    items = by_status.get(status, [])
    if items:
        total_count = sum(ogd_crops.get(r.source_crop, 0) for r in items)
        print(f"\n{status.value.upper()} ({len(items)} names, {total_count:,} records):")
        for r in items[:20]:
            count = ogd_crops.get(r.source_crop, 0)
            print(f"  '{r.source_crop}' -> {r.crop_type.value} ({count:,} records)")
        if len(items) > 20:
            print(f"  ... and {len(items)-20} more")

# Write CSV
csv_path = os.path.join(CROP_PREP, "government_crop_label_mapping.csv")
write_label_mapping_csv(resolutions, csv_path, ogd_crops)
print(f"\nWrote: {csv_path}")

# Summary stats
print("\n=== Resolution Summary ===")
total_records = 0
matched_records = 0
for r in resolutions:
    count = ogd_crops.get(r.source_crop, 0)
    total_records += count
    if r.status in (LabelMatchStatus.EXACT, LabelMatchStatus.ALIAS, LabelMatchStatus.NORMALIZED):
        matched_records += count
print(f"Total OGD records: {total_records:,}")
print(f"Records with valid crop type: {matched_records:,} ({100*matched_records/total_records:.1f}%)")
print(f"Records out of scope (fallow/NA/harvest): {total_records - matched_records:,}")

# Show which CropType members have data
print("\n=== CropType Member Coverage ===")
crop_counts = Counter()
for r in resolutions:
    if r.status in (LabelMatchStatus.EXACT, LabelMatchStatus.ALIAS, LabelMatchStatus.NORMALIZED):
        crop_counts[r.crop_type] += ogd_crops.get(r.source_crop, 0)

for ct in CropType:
    count = crop_counts.get(ct, 0)
    if count > 0:
        print(f"  {ct.value}: {count:,} records")
    else:
        print(f"  {ct.value}: 0 records")

# Validate enum is correct
print(f"\n=== CropType Enum Validation ===")
print(f"Total members: {len(CropType)}")
for ct in CropType:
    print(f"  {ct.name} = '{ct.value}' (member index {list(CropType).index(ct)})")
