#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
13b_candidate_space_al_pareto.py

Purpose
-------
Regenerate candidate-space active-learning ranking using the Pareto-filtered
candidate pool instead of raw Surrogate_9D_Predictions.csv.

Main input:
  results/Pareto_Optimal_Candidates.csv

Outputs:
  results/model_optimization/candidate_space_active_learning_screening_pareto.csv
  results/model_optimization/bde_oracle_candidate_manifest_pareto.csv
  results/model_optimization/candidate_space_al_pareto_score_summary.csv

This script is a model-optimization diagnostic. Do not archive its output into
manuscript_npJ until the score columns are verified and the manifest is actually
used for oracle augmentation.
"""

from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "model_optimization"
OUTDIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = [
    ROOT / "results" / "final_model_release" / "clean_sanitized_v1" / "training_dataset_clean_v1.csv",
    ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv",
]

CANDIDATE_FILE = ROOT / "results" / "Pareto_Optimal_Candidates.csv"

SENS_TARGETS = {
    "Trigger_Bond_Rho": {"candidate_cols": ["Pred_Trigger_Bond_Rho", "Trigger_Bond_Rho"], "low_high": True},
    "HOMO_LUMO_Gap(eV)": {"candidate_cols": ["Pred_HOMO_LUMO_Gap(eV)", "HOMO_LUMO_Gap(eV)"], "low_high": True},
    "VS_max": {"candidate_cols": ["Pred_VS_max", "VS_max"], "low_high": False},
    "Sigma2_tot": {"candidate_cols": ["Pred_Sigma2_tot", "Sigma2_tot"], "low_high": False},
    "Nu": {"candidate_cols": ["Pred_Nu", "Nu"], "low_high": False},
}

TRAIN_TARGET_COLS = [
    "Trigger_Bond_Rho",
    "HOMO_LUMO_Gap(eV)",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Density_calc(g/cm3)",
    "Density_calibrated(g/cm3)",
    "SAscore",
    "SA_Score",
]


SMARTS = {
    "C_NO2": "[#6]-[N+](=O)[O-]",
    "Aromatic_C_NO2": "[c]-[N+](=O)[O-]",
    "N_Nitro_NNO2": "[#7]-[N+](=O)[O-]",
    "O_Nitrate_ONO2": "[#8]-[N+](=O)[O-]",
    "Azide": "[N-]=[N+]=N",
    "N_N_single": "[#7]-[#7]",
    "N_O_single": "[#7]-[#8]",
    "C_N_single": "[#6]-[#7]",
}

ELECTRONEGATIVITY = {
    1: 2.20, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44,
    9: 3.98, 15: 2.19, 16: 2.58, 17: 3.16,
}


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No training file found.")


def infer_smiles_col(df: pd.DataFrame) -> str:
    priority = [
        "SMILES",
        "smiles",
        "Smiles",
        "Canonical_SMILES",
        "Canonical_SMILES_NoIso",
        "Generated_SMILES",
        "Final_SMILES",
    ]
    for c in priority:
        if c in df.columns:
            sample = df[c].dropna().astype(str).head(50).tolist()
            ok = sum(Chem.MolFromSmiles(x) is not None for x in sample)
            if ok >= max(3, len(sample) // 3):
                return c

    # Fallback: scan object columns and choose the one RDKit parses best.
    best_col = None
    best_ok = -1
    for c in df.columns:
        if df[c].dtype != object:
            continue
        sample = df[c].dropna().astype(str).head(100).tolist()
        if not sample:
            continue
        ok = sum(Chem.MolFromSmiles(x) is not None for x in sample)
        if ok > best_ok:
            best_ok = ok
            best_col = c

    if best_col is None or best_ok < 3:
        raise ValueError("Cannot infer SMILES column.")
    return best_col


def infer_id_col(df: pd.DataFrame, smiles_col: str) -> str:
    for c in ["Candidate_ID", "Molecule", "Name", "ID", "Example_ID", "example_id"]:
        if c in df.columns and c != smiles_col:
            return c
    return "__row_index__"


def canonicalize(smi):
    if pd.isna(smi):
        return None, None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None, None
    can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return can, mol


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def scaffold(mol):
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return "NO_SCAFFOLD"
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=False)
    except Exception:
        return "SCAFFOLD_ERROR"


def norm01(s: pd.Series, higher_is_better=True):
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() < 2:
        y = pd.Series(np.zeros(len(s)), index=s.index)
    else:
        lo = x.quantile(0.01)
        hi = x.quantile(0.99)
        denom = hi - lo
        if not np.isfinite(denom) or denom == 0:
            y = pd.Series(np.zeros(len(s)), index=s.index)
        else:
            y = (x.clip(lo, hi) - lo) / denom
    if not higher_is_better:
        y = 1.0 - y
    return y.fillna(0.0)


def bond_priority(a1, a2, bond):
    nums = sorted([a1.GetAtomicNum(), a2.GetAtomicNum()])
    priority = 10.0
    if nums == [7, 7]:
        priority = 1.0
    elif nums == [7, 8]:
        priority = 1.5
    elif nums == [6, 7]:
        priority = 2.0
    elif nums == [6, 8]:
        priority = 3.0

    if bond.GetBondTypeAsDouble() <= 1.1:
        priority -= 0.5
    else:
        priority += 0.5
    if bond.IsInRing():
        priority += 1.0
    if bond.GetIsAromatic():
        priority += 1.0

    return priority


def trigger_descriptors(mol):
    desc = {}
    if mol is None:
        return {
            "Trigger_Candidate_Bond_Count": 0,
            "Weakest_Trigger_Bond_Proxy": np.nan,
            "Weakest_Trigger_Bond_Type": "INVALID",
            "Weakest_Trigger_Bond_AtomPair": "",
        }

    for name, smarts in SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        desc[f"Motif_{name}_Count"] = len(mol.GetSubstructMatches(patt)) if patt is not None else 0

    candidates = []
    for b in mol.GetBonds():
        a1 = b.GetBeginAtom()
        a2 = b.GetEndAtom()
        z1, z2 = a1.GetAtomicNum(), a2.GetAtomicNum()
        nums = sorted([z1, z2])

        if nums not in ([7, 7], [7, 8], [6, 7], [6, 8]):
            continue

        en_diff = abs(ELECTRONEGATIVITY.get(z1, 2.5) - ELECTRONEGATIVITY.get(z2, 2.5))
        formal_charge_abs = abs(a1.GetFormalCharge()) + abs(a2.GetFormalCharge())

        proxy = (
            bond_priority(a1, a2, b)
            + 0.25 * b.GetBondTypeAsDouble()
            - 0.20 * formal_charge_abs
            - 0.10 * en_diff
        )

        candidates.append({
            "bond_type": f"{a1.GetSymbol()}-{a2.GetSymbol()}",
            "atom_pair": f"{a1.GetIdx()}-{a2.GetIdx()}",
            "proxy": proxy,
            "bond_order": b.GetBondTypeAsDouble(),
            "is_ring": int(b.IsInRing()),
            "is_aromatic": int(b.GetIsAromatic()),
        })

    desc["Trigger_Candidate_Bond_Count"] = len(candidates)

    if not candidates:
        desc.update({
            "Weakest_Trigger_Bond_Proxy": np.nan,
            "Weakest_Trigger_Bond_Type": "NONE",
            "Weakest_Trigger_Bond_AtomPair": "",
            "Weakest_Trigger_Bond_Order": np.nan,
            "Weakest_Trigger_Bond_IsRing": np.nan,
            "Weakest_Trigger_Bond_IsAromatic": np.nan,
        })
        return desc

    best = sorted(candidates, key=lambda x: x["proxy"])[0]
    desc.update({
        "Weakest_Trigger_Bond_Proxy": best["proxy"],
        "Weakest_Trigger_Bond_Type": best["bond_type"],
        "Weakest_Trigger_Bond_AtomPair": best["atom_pair"],
        "Weakest_Trigger_Bond_Order": best["bond_order"],
        "Weakest_Trigger_Bond_IsRing": best["is_ring"],
        "Weakest_Trigger_Bond_IsAromatic": best["is_aromatic"],
    })
    return desc


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    train_file = first_existing(TRAIN_FILES)
    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(f"Missing candidate file: {CANDIDATE_FILE}")

    print(f"[INFO] Training data: {train_file}")
    train = pd.read_csv(train_file)
    train_smi_col = infer_smiles_col(train)
    print(f"[INFO] Training SMILES column: {train_smi_col}")

    train_fps = []
    train_scaffolds = set()
    valid_train_rows = []

    for i, row in train.iterrows():
        can, mol = canonicalize(row[train_smi_col])
        if mol is None:
            continue
        train_fps.append(fp(mol))
        train_scaffolds.add(scaffold(mol))
        rec = {"Canonical_SMILES_NoIso": can}
        for t in TRAIN_TARGET_COLS:
            if t in train.columns:
                rec[t] = row[t]
        valid_train_rows.append(rec)

    train_ref = pd.DataFrame(valid_train_rows)
    print(f"[INFO] Valid training molecules: {len(train_ref)}")

    cand = pd.read_csv(CANDIDATE_FILE)
    cand_smi_col = infer_smiles_col(cand)
    cand_id_col = infer_id_col(cand, cand_smi_col)
    print(f"[INFO] Candidate file: {CANDIDATE_FILE}")
    print(f"[INFO] Candidate rows: {len(cand)}")
    print(f"[INFO] Candidate SMILES column: {cand_smi_col}")
    print(f"[INFO] Candidate ID column: {cand_id_col}")

    out_rows = []
    for i, row in cand.iterrows():
        can, mol = canonicalize(row[cand_smi_col])
        if mol is None:
            continue

        cfp = fp(mol)
        sim = max(DataStructs.BulkTanimotoSimilarity(cfp, train_fps)) if train_fps else np.nan
        scaf = scaffold(mol)

        rec = row.to_dict()
        rec["Candidate_Row_Index"] = i
        rec["Candidate_ID"] = row[cand_id_col] if cand_id_col != "__row_index__" else i
        rec["Canonical_SMILES_NoIso"] = can
        rec["Murcko_Scaffold"] = scaf
        rec["NearestTrain_Tanimoto"] = sim
        rec["DistanceToTrain"] = 1.0 - sim if pd.notna(sim) else np.nan
        rec["UnseenScaffold"] = int(scaf not in train_scaffolds)
        rec.update(trigger_descriptors(mol))
        out_rows.append(rec)

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No valid candidate molecules parsed.")

    # Structural OOD score.
    out["Structural_OOD_Score"] = (
        0.70 * norm01(out["DistanceToTrain"], higher_is_better=True)
        + 0.30 * out["UnseenScaffold"].astype(float)
    )

    # Sensitivity risk and boundary score.
    risk_terms = []
    boundary_terms = []

    for train_t, meta in SENS_TARGETS.items():
        cand_col = find_col(out, meta["candidate_cols"])
        if cand_col is None or train_t not in train_ref.columns:
            print(f"[WARN] Missing sensitivity target or candidate column for {train_t}")
            continue

        train_vals = pd.to_numeric(train_ref[train_t], errors="coerce").dropna()
        if len(train_vals) < 20:
            print(f"[WARN] Too few training values for {train_t}")
            continue

        q33 = train_vals.quantile(0.33)
        q66 = train_vals.quantile(0.66)
        x = pd.to_numeric(out[cand_col], errors="coerce")

        if meta["low_high"]:
            risk = norm01(x, higher_is_better=False)
        else:
            risk = norm01(x, higher_is_better=True)

        scale = max(abs(q66 - q33), 1e-12)
        dist = np.minimum(abs(x - q33), abs(x - q66)) / scale
        boundary = 1.0 / (1.0 + dist)

        risk_terms.append(risk)
        boundary_terms.append(boundary.fillna(0.0))

    out["Sensitivity_Risk_Score"] = pd.concat(risk_terms, axis=1).mean(axis=1) if risk_terms else 0.0
    out["Sensitivity_Boundary_Score"] = pd.concat(boundary_terms, axis=1).mean(axis=1) if boundary_terms else 0.0

    # Pareto / screening score.
    pareto_terms = []

    # Main Pareto screening column from 05.
    if "Screening_Score_9Target" in out.columns:
        pareto_terms.append(norm01(out["Screening_Score_9Target"], higher_is_better=True))
    elif "Screening_Score_9D" in out.columns:
        pareto_terms.append(norm01(out["Screening_Score_9D"], higher_is_better=True))

    # Density surrogate.
    for c in ["Pred_Density_calibrated(g/cm3)", "Pred_Density_calc(g/cm3)", "Density_calibrated(g/cm3)", "Density_Calibrated(g/cm3)"]:
        if c in out.columns:
            pareto_terms.append(norm01(out[c], higher_is_better=True))
            break

    # Synthetic accessibility: lower is usually better.
    for c in ["Pred_SAscore", "SA_Score_RDKit", "Pareto_SA_Score", "SAscore", "SA_Score"]:
        if c in out.columns:
            pareto_terms.append(norm01(out[c], higher_is_better=False))
            break

    # Detonation properties, if present.
    for c in ["D_calibrated(km/s)", "D_calibrated(m/s)", "D_proxy(km/s)", "D_proxy_original(m/s)"]:
        if c in out.columns:
            pareto_terms.append(norm01(out[c], higher_is_better=True))
            break

    for c in ["P_calibrated(GPa)", "P_proxy(GPa)", "P_proxy_original(GPa)"]:
        if c in out.columns:
            pareto_terms.append(norm01(out[c], higher_is_better=True))
            break

    if pareto_terms:
        out["Pareto_Potential_Score"] = pd.concat(pareto_terms, axis=1).mean(axis=1)
    else:
        out["Pareto_Potential_Score"] = 0.0

    # Optional model uncertainty columns if they exist.
    uncertainty_cols = [c for c in out.columns if re.search(r"(std|uncert|uncertainty)", c, re.I)]
    if uncertainty_cols:
        out["Ensemble_Uncertainty_Score"] = pd.concat(
            [norm01(out[c], higher_is_better=True) for c in uncertainty_cols],
            axis=1
        ).mean(axis=1)
    else:
        out["Ensemble_Uncertainty_Score"] = 0.0

    # Final AL score.
    out["Candidate_AL_Score"] = (
        0.25 * out["Structural_OOD_Score"]
        + 0.20 * out["Sensitivity_Boundary_Score"]
        + 0.20 * out["Sensitivity_Risk_Score"]
        + 0.25 * out["Pareto_Potential_Score"]
        + 0.10 * out["Ensemble_Uncertainty_Score"]
    )

    # BDE priority.
    out["BDE_Oracle_Priority_Score"] = (
        0.35 * out["Candidate_AL_Score"]
        + 0.25 * norm01(out["Trigger_Candidate_Bond_Count"], higher_is_better=True)
        + 0.25 * norm01(out["Weakest_Trigger_Bond_Proxy"], higher_is_better=False)
        + 0.15 * out["Structural_OOD_Score"]
    )

    out = out.sort_values(
        ["Candidate_AL_Score", "BDE_Oracle_Priority_Score"],
        ascending=False
    ).reset_index(drop=True)

    out_path = OUTDIR / "candidate_space_active_learning_screening_pareto.csv"
    out.to_csv(out_path, index=False)

    manifest_cols = [
        "Candidate_ID",
        "Canonical_SMILES_NoIso",
        "Candidate_AL_Score",
        "BDE_Oracle_Priority_Score",
        "Screening_Score_9Target",
        "Screening_Score_9D",
        "Pareto_Potential_Score",
        "Structural_OOD_Score",
        "Sensitivity_Risk_Score",
        "Sensitivity_Boundary_Score",
        "NearestTrain_Tanimoto",
        "UnseenScaffold",
        "Trigger_Candidate_Bond_Count",
        "Weakest_Trigger_Bond_Type",
        "Weakest_Trigger_Bond_AtomPair",
        "Weakest_Trigger_Bond_Proxy",
        "Pred_Density_calc(g/cm3)",
        "Pred_HOMO_LUMO_Gap(eV)",
        "Pred_SAscore",
        "Pred_VS_max",
        "Pred_Sigma2_tot",
        "Pred_Nu",
        "Pred_Trigger_Bond_Rho",
    ]
    manifest_cols = [c for c in manifest_cols if c in out.columns]

    manifest = out[manifest_cols].head(100)
    manifest_path = OUTDIR / "bde_oracle_candidate_manifest_pareto.csv"
    manifest.to_csv(manifest_path, index=False)

    summary_rows = []
    for col in [
        "Candidate_AL_Score",
        "BDE_Oracle_Priority_Score",
        "Pareto_Potential_Score",
        "Structural_OOD_Score",
        "Sensitivity_Risk_Score",
        "Sensitivity_Boundary_Score",
        "NearestTrain_Tanimoto",
        "Trigger_Candidate_Bond_Count",
    ]:
        vals = pd.to_numeric(out[col], errors="coerce")
        summary_rows.append({
            "Column": col,
            "N": vals.notna().sum(),
            "Mean": vals.mean(),
            "Median": vals.median(),
            "Min": vals.min(),
            "Q25": vals.quantile(0.25),
            "Q75": vals.quantile(0.75),
            "Max": vals.max(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "candidate_space_al_pareto_score_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"[SAVE] {out_path}")
    print(f"[SAVE] {manifest_path}")
    print(f"[SAVE] {summary_path}")

    print("\n===== Score summary =====")
    print(summary.to_string(index=False))

    show_cols = [
        "Candidate_ID",
        "Candidate_AL_Score",
        "BDE_Oracle_Priority_Score",
        "Pareto_Potential_Score",
        "Structural_OOD_Score",
        "Sensitivity_Risk_Score",
        "Sensitivity_Boundary_Score",
        "NearestTrain_Tanimoto",
        "UnseenScaffold",
        "Trigger_Candidate_Bond_Count",
        "Weakest_Trigger_Bond_Type",
        "Weakest_Trigger_Bond_AtomPair",
        "Weakest_Trigger_Bond_Proxy",
        "Screening_Score_9D",
        "Pred_Density_calc(g/cm3)",
        "Pred_SAscore",
        "Pred_Trigger_Bond_Rho",
    ]
    show_cols = [c for c in show_cols if c in out.columns]

    print("\n===== Top 30 Pareto-aware AL candidates =====")
    print(out[show_cols].head(30).to_string(index=False))

    # Hard stop condition: Pareto score should not be all zero.
    if float(pd.to_numeric(out["Pareto_Potential_Score"], errors="coerce").fillna(0).max()) <= 0:
        raise RuntimeError(
            "Pareto_Potential_Score is still zero. Candidate AL is not valid for manuscript or BDE selection."
        )

    print("\n[DONE] Pareto-aware candidate-space AL completed.")


if __name__ == "__main__":
    main()
