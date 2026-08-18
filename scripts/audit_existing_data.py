"""Audit existing CropPrep data for baseline counts."""
import csv
import collections
import json

# Read data_season.csv
with open("D:/CropPrep/Tabular_Datasets/data_season.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print(f"Unique years: {sorted(set(r['Year'] for r in rows))}")
print(f"Unique seasons: {sorted(set(r['Season'] for r in rows))}")
print(f"Unique locations: {sorted(set(r['Location'] for r in rows))}")

# Crop distribution
crops = collections.Counter(r['Crops'] for r in rows)
print("\nCrop distribution (all):")
for c, n in crops.most_common():
    print(f"  {c}: {n}")

# Mangalore crop distribution
mang = [r for r in rows if r['Location'] == 'Mangalore']
print(f"\nMangalore rows: {len(mang)}")
mang_crops = collections.Counter(r['Crops'] for r in mang)
print("Mangalore crop distribution:")
for c, n in mang_crops.most_common():
    print(f"  {c}: {n}")

# DK-adjacent locations (Mangalore, Kodagu, Chikmangaluru, Hassan, Kasaragodu, Madikeri)
dk_locs = ['Mangalore', 'Kodagu', 'Chikmangaluru', 'Hassan', 'Kasaragodu', 'Madikeri']
dk_rows = [r for r in rows if r['Location'] in dk_locs]
print(f"\nDK-area rows: {len(dk_rows)}")
dk_crops = collections.Counter(r['Crops'] for r in dk_rows)
print("DK-area crop distribution:")
for c, n in dk_crops.most_common():
    print(f"  {c}: {n}")

# DK-Features row counts
for year in range(2018, 2025):
    fn = f"D:/CropPrep/Tabular_Datasets/DK_Features_{year} (1).csv" if year == 2024 else f"D:/CropPrep/Tabular_Datasets/DK_Features_{year}.csv"
    try:
        with open(fn, "r") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f"DK_Features_{year}: {count} rows")
    except FileNotFoundError:
        try:
            fn2 = f"D:/CropPrep/Tabular_Datasets/DK_Features_{year} (1).csv"
            with open(fn2, "r") as f:
                reader = csv.DictReader(f)
                count = sum(1 for _ in reader)
            print(f"DK_Features_{year} (1): {count} rows")
        except FileNotFoundError:
            print(f"DK_Features_{year}: NOT FOUND")

# Readiness report
with open("D:/CropPrep/training/artifacts/readiness_report/readiness_report.json", "r") as f:
    report = json.load(f)
print(f"\nReadiness report:")
print(f"  crop_training_samples: {report['criteria']['D']['crop_training_samples']}")
print(f"  yield_training_samples: {report['criteria']['D']['yield_training_samples']}")
print(f"  train_crop_labeled: {report['criteria']['E']['train_crop_labeled']}")
print(f"  val_crop_labeled: {report['criteria']['E']['val_crop_labeled']}")
print(f"  test_crop_labeled: {report['criteria']['E']['test_crop_labeled']}")
print(f"  overall: {report['overall']}")
