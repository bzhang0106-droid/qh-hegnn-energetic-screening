#!/usr/bin/env python3
"""Build the shared 10D target matrix from the official curated table.

This script is intentionally non-destructive: it does not edit the official
training table. It writes complete-case target matrices and a missing-row report
so downstream scripts can consume consistent target names.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv"
BASELINE_OUTPUT = ROOT / "data" / "baselines" / "target_matrix_10d.csv"
CURATED_OUTPUT = ROOT / "data" / "curated_molecule_clean_v1" / "target_matrix_10d_molecule_clean.csv"
MISSING_OUTPUT = ROOT / "data" / "baselines" / "target_matrix_10d_missing_rows.csv"
SUMMARY_OUTPUT = ROOT / "data" / "baselines" / "target_matrix_10d_summary.json"

TARGET_MAP: Dict[str, str] = {
    "Density": "Density_calc(g/cm3)",
    "Heat_of_Formation": "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap": "HOMO_LUMO_Gap(eV)",
    "SA_Score": "SAscore",
    "VS_max": "VS_max",
    "Sigma2_tot": "Sigma2_tot",
    "Nu": "Nu",
    "Trigger_Bond_Rho": "Trigger_Bond_Rho",
    "Molecular_Weight": "Molecular_Weight",
    "Vertical_BDE": "Vertical_BDE(kcal/mol)",
}

COMPAT_FULL_TARGETS = [
    "Density_calc(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "SA_Score",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
]

META_CANDIDATES = [
    "clean_row_index",
    "curated_row_index",
    "_original_row_index",
    "Molecule",
    "SMILES",
    "_canonical_smiles_noiso",
    "_formula",
    "Density_Source",
    "Density_Label_Source",
    "Density_Label_Is_True_Crystal",
    "BDE_Job_Dir",
    "BDE_Bond_Type",
    "BDE_Bond_i_1based",
    "BDE_Bond_j_1based",
    "BDE_Bond_WBO",
    "BDE_Parse_Status",
    "_BDE_Align_Source",
]


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Official curated table is missing required columns: {missing}")


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    required = sorted(set(TARGET_MAP.values()))
    require_columns(df, required)

    out = pd.DataFrame()
    out["Row_Index"] = np.arange(len(df), dtype=int)
    out["Example_ID"] = df.get("clean_row_index", pd.Series(np.arange(len(df)), index=df.index)).values

    for c in META_CANDIDATES:
        if c in df.columns:
            out[c] = df[c].values

    # Full physical names for training/evaluation scripts that use the official schema.
    for c in COMPAT_FULL_TARGETS:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # Short aliases for baseline scripts and manuscript-facing summary tables.
    for alias, source in TARGET_MAP.items():
        out[alias] = pd.to_numeric(df[source], errors="coerce")

    target_aliases = list(TARGET_MAP.keys())
    missing_mask = out[target_aliases].isna().any(axis=1)
    missing = out.loc[missing_mask, ["Row_Index", "Example_ID", "Molecule", "SMILES"] + target_aliases]
    complete = out.loc[~missing_mask].copy()

    BASELINE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CURATED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(BASELINE_OUTPUT, index=False)
    complete.to_csv(CURATED_OUTPUT, index=False)
    missing.to_csv(MISSING_OUTPUT, index=False)

    summary = {
        "input": str(INPUT.relative_to(ROOT)),
        "baseline_output": str(BASELINE_OUTPUT.relative_to(ROOT)),
        "curated_output": str(CURATED_OUTPUT.relative_to(ROOT)),
        "missing_output": str(MISSING_OUTPUT.relative_to(ROOT)),
        "input_rows": int(len(df)),
        "complete_rows": int(len(complete)),
        "missing_rows": int(len(missing)),
        "target_aliases": target_aliases,
        "target_sources": TARGET_MAP,
        "bde_status_counts": df.get("BDE_Parse_Status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict(),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[SAVE] {BASELINE_OUTPUT}")
    print(f"[SAVE] {CURATED_OUTPUT}")
    print(f"[SAVE] {MISSING_OUTPUT}")
    print(f"[SAVE] {SUMMARY_OUTPUT}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()