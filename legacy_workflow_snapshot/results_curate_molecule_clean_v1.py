#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build conflict-filtered clean molecule-level dataset.

Input:
  data/curated_molecule_unique_v1/

Output:
  data/curated_molecule_clean_v1/

Logic:
  - Keep molecule-unique rows whose original duplicate group is unique or low-conflict.
  - Quarantine entire original duplicate groups if any target range exceeds target-specific conflict thresholds.
  - Do not average labels.
  - Do not use legacy 3672-row target_matrix_9d or 2d_feature_matrix.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
SRC = ROOT / "data" / "curated_molecule_unique_v1"
OUT = ROOT / "data" / "curated_molecule_clean_v1"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TRAIN_FRAC = 0.81
CALIB_FRAC = 0.09

TARGET_COLS = [
    "Density_calc(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
]

# Conservative first-pass conflict thresholds.
# These are not chemical truth thresholds; they are data-cleaning thresholds for duplicate groups.
CONFLICT_THRESHOLDS = {
    "Density_calc(g/cm3)": 0.03,
    "Heat_of_Formation(kcal/mol)": 5.0,
    "HOMO_LUMO_Gap(eV)": 0.20,
    "SAscore": 0.20,
    "VS_max": 5.0,
    "Sigma2_tot": 1.0e-4,
    "Nu": 0.03,
    "Trigger_Bond_Rho": 0.005,
    "Molecular_Weight": 1.0e-6,
}

required = [
    SRC / "old_dataset_molecule_unique.csv",
    SRC / "target_matrix_9targets_molecule_unique.csv",
    SRC / "xtb_features_molecule_unique.csv",
    SRC / "duplicate_group_audit_molecule_unique.csv",
]
for p in required:
    if not p.exists():
        raise SystemExit(f"[ERROR] missing required input: {p}")

curated = pd.read_csv(SRC / "old_dataset_molecule_unique.csv")
target = pd.read_csv(SRC / "target_matrix_9targets_molecule_unique.csv")
xtb = pd.read_csv(SRC / "xtb_features_molecule_unique.csv")
audit = pd.read_csv(SRC / "duplicate_group_audit_molecule_unique.csv")

if not (len(curated) == len(target) == len(xtb)):
    raise SystemExit(
        f"[ERROR] row mismatch: curated={len(curated)}, target={len(target)}, xtb={len(xtb)}"
    )

if "curated_row_index" not in curated.columns:
    raise SystemExit("[ERROR] curated_row_index missing in old_dataset_molecule_unique.csv")

if "kept_original_row_index" not in audit.columns:
    raise SystemExit("[ERROR] kept_original_row_index missing in duplicate audit")

# Mark conflict groups from original duplicate audit.
audit = audit.copy()

conflict_flags = []
conflict_reasons = []

for _, row in audit.iterrows():
    reasons = []
    n_rows = int(row.get("n_rows", 1))

    if n_rows <= 1:
        conflict_flags.append(False)
        conflict_reasons.append("")
        continue

    for target_name, threshold in CONFLICT_THRESHOLDS.items():
        range_col = f"{target_name}__range"
        if range_col not in audit.columns:
            continue
        val = row.get(range_col)
        try:
            val = float(val)
        except Exception:
            continue
        if np.isfinite(val) and val > threshold:
            reasons.append(f"{target_name}:range={val:g}>thr={threshold:g}")

    is_conflict = len(reasons) > 0
    conflict_flags.append(is_conflict)
    conflict_reasons.append("; ".join(reasons))

audit["label_conflict_quarantine"] = conflict_flags
audit["label_conflict_reason"] = conflict_reasons

# Map original kept row index -> quarantine flag.
conflict_original_indices = set(
    audit.loc[audit["label_conflict_quarantine"], "kept_original_row_index"]
    .astype(int)
    .tolist()
)

if "_original_row_index" not in curated.columns:
    raise SystemExit("[ERROR] _original_row_index missing in curated dataset")

curated["_label_conflict_quarantine"] = curated["_original_row_index"].astype(int).isin(conflict_original_indices)

clean = curated[~curated["_label_conflict_quarantine"]].copy().reset_index(drop=True)
quarantine = curated[curated["_label_conflict_quarantine"]].copy().reset_index(drop=True)

# Slice target and xTB using curated_row_index from molecule_unique table.
clean_unique_indices = clean["curated_row_index"].astype(int).tolist()
quarantine_unique_indices = quarantine["curated_row_index"].astype(int).tolist()

clean_target = target.iloc[clean_unique_indices].copy().reset_index(drop=True)
clean_xtb = xtb.iloc[clean_unique_indices].copy().reset_index(drop=True)

quarantine_target = target.iloc[quarantine_unique_indices].copy().reset_index(drop=True)
quarantine_xtb = xtb.iloc[quarantine_unique_indices].copy().reset_index(drop=True)

# Re-index clean rows.
clean.insert(0, "clean_row_index", np.arange(len(clean)))
clean_target.insert(0, "clean_row_index", np.arange(len(clean_target)))
clean_xtb.insert(0, "clean_row_index", np.arange(len(clean_xtb)))

quarantine.insert(0, "quarantine_row_index", np.arange(len(quarantine)))
quarantine_target.insert(0, "quarantine_row_index", np.arange(len(quarantine_target)))
quarantine_xtb.insert(0, "quarantine_row_index", np.arange(len(quarantine_xtb)))

# New split.
rng = np.random.default_rng(SEED)
indices = np.arange(len(clean))
rng.shuffle(indices)

n_total = len(indices)
n_train = int(round(n_total * TRAIN_FRAC))
n_calib = int(round(n_total * CALIB_FRAC))
n_val = n_total - n_train - n_calib

train = sorted(map(int, indices[:n_train]))
calib = sorted(map(int, indices[n_train:n_train+n_calib]))
val = sorted(map(int, indices[n_train+n_calib:]))

def example_ids(rows):
    out = []
    for i in rows:
        r = clean.iloc[i]
        mol = r["Molecule"] if "Molecule" in clean.columns else ""
        smi = r["SMILES"] if "SMILES" in clean.columns else ""
        orig = int(r["_original_row_index"])
        out.append(f"clean{int(i):08d}||orig{orig:08d}||{mol}||{smi}")
    return out

split = {
    "dataset_version": "curated_molecule_clean_v1",
    "parent_dataset_version": "curated_molecule_unique_v1",
    "dedup_key": "canonical_smiles_noiso",
    "label_conflict_policy": "duplicate groups exceeding target-specific range thresholds were quarantined",
    "seed": SEED,
    "val_fraction": 1.0 - TRAIN_FRAC - CALIB_FRAC,
    "specialist_calib_fraction": CALIB_FRAC,
    "train_core_row_indices": train,
    "calib_row_indices": calib,
    "val_row_indices": val,
    "train_core_example_ids": example_ids(train),
    "calib_example_ids": example_ids(calib),
    "val_example_ids": example_ids(val),
    "targets": TARGET_COLS,
    "xtb_feature_path": "../data/curated_molecule_clean_v1/xtb_features_molecule_clean.csv",
    "note": "Row indices refer to old_dataset_molecule_clean.csv."
}

# Save.
clean.to_csv(OUT / "old_dataset_molecule_clean.csv", index=False)
clean_target.to_csv(OUT / "target_matrix_9targets_molecule_clean.csv", index=False)
clean_xtb.to_csv(OUT / "xtb_features_molecule_clean.csv", index=False)

quarantine.to_csv(OUT / "quarantine_label_conflict_rows.csv", index=False)
quarantine_target.to_csv(OUT / "quarantine_label_conflict_target_matrix.csv", index=False)
quarantine_xtb.to_csv(OUT / "quarantine_label_conflict_xtb_features.csv", index=False)

audit.to_csv(OUT / "duplicate_group_audit_with_conflict_flags.csv", index=False)

with open(OUT / "train_calib_val_split_molecule_clean.json", "w") as f:
    json.dump(split, f, indent=2)

# Post-check.
n_noiso_dup = int(clean["_canonical_smiles_noiso"].duplicated().sum()) if "_canonical_smiles_noiso" in clean.columns else -1
n_iso_dup = int(clean["_canonical_smiles_iso"].duplicated().sum()) if "_canonical_smiles_iso" in clean.columns else -1

summary = {
    "unique_input_n": int(len(curated)),
    "clean_n": int(len(clean)),
    "quarantine_n": int(len(quarantine)),
    "quarantine_fraction_of_unique": float(len(quarantine) / len(curated)) if len(curated) else 0.0,
    "conflict_duplicate_groups_quarantined": int(audit["label_conflict_quarantine"].sum()),
    "canonical_noiso_duplicates_after_clean": n_noiso_dup,
    "canonical_iso_duplicates_after_clean": n_iso_dup,
    "clean_target_rows": int(len(clean_target)),
    "clean_xtb_rows": int(len(clean_xtb)),
    "train_core_n": len(train),
    "calib_n": len(calib),
    "val_n": len(val),
    "split_total": len(train) + len(calib) + len(val),
    "target_source": "data/curated_molecule_unique_v1/old_dataset_molecule_unique.csv",
    "xtb_source": "data/curated_molecule_unique_v1/xtb_features_molecule_unique.csv",
    "legacy_3672_target_matrix_used": False,
    "legacy_3672_2d_feature_matrix_used": False,
}

pd.DataFrame([summary]).to_csv(OUT / "clean_curation_summary.csv", index=False)
pd.DataFrame([
    {"target": k, "conflict_threshold": v}
    for k, v in CONFLICT_THRESHOLDS.items()
]).to_csv(OUT / "label_conflict_thresholds.csv", index=False)

print("===== curated_molecule_clean_v1 summary =====")
for k, v in summary.items():
    print(f"{k}: {v}")

print()
print("===== output files =====")
for p in sorted(OUT.glob("*")):
    print(p)
