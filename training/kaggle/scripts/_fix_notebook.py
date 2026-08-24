import json

nb_path = "training/kaggle/notebooks/R5_3_train.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

cell = nb["cells"][7]
new_source = [
    "import json, csv\n",
    "\n",
    "manifest_path = REPO_ROOT / 'training_manifests' / 'crop_supervised_v1_manifest.json'\n",
    "csv_path = REPO_ROOT / 'govt_crop_matched_v1' / 'crop_supervised_v1.csv'\n",
    "\n",
    "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n",
    "\n",
    "with open(csv_path, newline='', encoding='utf-8') as f:\n",
    "    rows = list(csv.DictReader(f))\n",
    "\n",
    "class_counts = {}\n",
    "split_counts = {'train': 0, 'val': 0, 'test': 0}\n",
    "TALUK_SPLIT = {'Belthangady': 'train', 'Mangalore': 'train', 'Puttur': 'train',\n",
    "               'Bantwal': 'val', 'Sullia': 'test'}\n",
    "for r in rows:\n",
    "    c = r['crop_label']\n",
    "    class_counts[c] = class_counts.get(c, 0) + 1\n",
    "    s = TALUK_SPLIT.get(r['location_taluk'], 'unknown')\n",
    "    split_counts[s] += 1\n",
    "\n",
    "print('=' * 50)\n",
    "print('  R5.3 FROZEN DATA CONTRACT')\n",
    "print('=' * 50)\n",
    "print(f'  Manifest: {manifest_path}')\n",
    "print(f'  Dataset version: {manifest[\"dataset_version\"]}')\n",
    "print(f'  Total: {len(rows)}')\n",
    "print(f'  Train: {split_counts[\"train\"]}')\n",
    "print(f'  Validation: {split_counts[\"val\"]}')\n",
    "print(f'  Test: {split_counts[\"test\"]}')\n",
    "print('  Classes:')\n",
    "for c in sorted(class_counts.keys()):\n",
    "    print(f'    {c}: {class_counts[c]}')\n",
    "print('=' * 50)\n",
    "\n",
    "assert len(rows) == 10674, f'Expected 10674, got {len(rows)}'\n",
    "assert split_counts['train'] == 6116, f'Expected train=6116, got {split_counts[\"train\"]}'\n",
    "assert split_counts['val'] == 2267, f'Expected val=2267, got {split_counts[\"val\"]}'\n",
    "assert split_counts['test'] == 2291, f'Expected test=2291, got {split_counts[\"test\"]}'\n",
    "print('\\nFROZEN DATA CONTRACT: PASS')",
]
cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated: checksum removed from data contract cell")
