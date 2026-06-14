#!/usr/bin/env python3
"""Prepare xTB task manifest for molecules missing from the current aligned xTB table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING = ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv"
DEFAULT_XTB = ROOT / "data" / "curated_molecule_clean_v1" / "xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv"
RAW_XYZ_DIR = ROOT / "data" / "raw_2100_xyz"
XTB_WORK_ROOT = ROOT / "xtb_calc"
EF_DIR = ROOT / "data" / "electronic_features"


def canonical_noiso(smiles: object) -> Optional[str]:
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def xtb_keys(df: pd.DataFrame) -> set[str]:
    candidates = [
        "_training_canonical_smiles_noiso",
        "_canonical_smiles_noiso",
        "Canonical_SMILES_NoIso",
        "canonical_smiles",
        "SMILES",
    ]
    for col in candidates:
        if col in df.columns:
            vals = df[col].map(canonical_noiso if col.lower() in {"smiles", "canonical_smiles"} else lambda x: str(x) if pd.notna(x) else None)
            return set(vals.dropna().astype(str))
    return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="Tag for output files, e.g. AL06_20260604")
    ap.add_argument("--training", default=str(DEFAULT_TRAINING))
    ap.add_argument("--aligned_xtb", default=str(DEFAULT_XTB))
    ap.add_argument("--raw_xyz_dir", default=str(RAW_XYZ_DIR))
    ap.add_argument("--work_root", default=str(XTB_WORK_ROOT))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--inventory", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()

    training_path = Path(args.training)
    xtb_path = Path(args.aligned_xtb)
    raw_xyz_dir = Path(args.raw_xyz_dir)
    work_root = Path(args.work_root)
    manifest = Path(args.manifest) if args.manifest else EF_DIR / f"xtb_tasks_{args.tag}.tsv"
    inventory = Path(args.inventory) if args.inventory else EF_DIR / f"xtb_missing_{args.tag}_inventory.csv"
    summary_path = Path(args.summary) if args.summary else EF_DIR / f"xtb_missing_{args.tag}_summary.json"

    train = pd.read_csv(training_path)
    if "SMILES" not in train.columns or "Molecule" not in train.columns:
        raise RuntimeError("Training table must contain SMILES and Molecule columns")
    train["_canonical_smiles_noiso_for_xtb"] = train["SMILES"].map(canonical_noiso)

    if xtb_path.exists() and xtb_path.stat().st_size > 0:
        known = xtb_keys(pd.read_csv(xtb_path))
    else:
        known = set()

    missing_rows = []
    tasks = []
    task_id = 0
    for idx, row in train.iterrows():
        key = row["_canonical_smiles_noiso_for_xtb"]
        if not key or key in known:
            continue
        mol = str(row["Molecule"])
        xyz = raw_xyz_dir / f"{mol}.xyz"
        workdir = work_root / mol
        rec = {
            "row_index": int(idx),
            "example_id": row.get("clean_row_index", idx),
            "Molecule": mol,
            "SMILES": row["SMILES"],
            "canonical_smiles_noiso": key,
            "xyz_path": str(xyz),
            "xyz_exists": bool(xyz.exists()),
            "workdir": str(workdir),
        }
        missing_rows.append(rec)
        if xyz.exists():
            task_id += 1
            tasks.append({
                "task_id": task_id,
                "row_index": int(idx),
                "example_id": rec["example_id"],
                "molecule": mol,
                "xyz_path": str(xyz),
                "workdir": str(workdir),
            })

    inventory.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    inv = pd.DataFrame(missing_rows)
    inv.to_csv(inventory, index=False)
    pd.DataFrame(tasks, columns=["task_id", "row_index", "example_id", "molecule", "xyz_path", "workdir"]).to_csv(manifest, sep="\t", index=False)

    summary = {
        "tag": args.tag,
        "training": str(training_path.relative_to(ROOT) if training_path.is_relative_to(ROOT) else training_path),
        "aligned_xtb": str(xtb_path.relative_to(ROOT) if xtb_path.is_relative_to(ROOT) else xtb_path),
        "training_rows": int(len(train)),
        "known_xtb_keys": int(len(known)),
        "missing_rows": int(len(inv)),
        "tasks_with_xyz": int(len(tasks)),
        "missing_xyz_rows": int((~inv["xyz_exists"]).sum()) if not inv.empty else 0,
        "manifest": str(manifest.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest),
        "inventory": str(inventory.relative_to(ROOT) if inventory.is_relative_to(ROOT) else inventory),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()