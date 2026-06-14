from pathlib import Path
import csv
import json

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")

for rel in [
    "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv",
    "data/baselines/target_matrix_10d.csv",
    "data/curated_molecule_clean_v1/target_matrix_10d_molecule_clean.csv",
    "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv",
    "data/electronic_features/xtb_tasks_AL08_20260605.tsv",
    "data/electronic_features/xtb_missing_AL08_20260605_summary.json",
]:
    p = ROOT / rel
    if not p.exists():
        print(rel, "MISSING")
        continue
    if p.suffix.lower() == ".json":
        try:
            print(rel, json.dumps(json.loads(p.read_text()), ensure_ascii=False)[:500])
        except Exception as exc:
            print(rel, "JSON_ERR", type(exc).__name__)
        continue
    with p.open(newline="", encoding="utf-8") as fh:
        n = sum(1 for _ in fh) - 1
    print(rel, n)

p = ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
if p.exists():
    counts = {}
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mol = (row.get("Molecule") or row.get("Name") or "").strip()
            pref = mol.split("_Target_")[0] if "_Target_" in mol else "BASE"
            counts[pref] = counts.get(pref, 0) + 1
    print("official_db_prefix_counts", counts)
