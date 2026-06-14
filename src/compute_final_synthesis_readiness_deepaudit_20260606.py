#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DB_PATH = ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
PKG_DIR = ROOT / "manuscript_npJ/final_submission_package_AL08_20260605"
OUT_DIR = ROOT / "results/final_global_top20/synthesis_readiness"
TABLES_DIR = ROOT / "manuscript_npJ/final_tables_figures"
AUDIT_MD = ROOT / "manuscript_npJ/Final_Top20_Synthesis_Readiness_DeepAudit_20260606.md"
RUN_META = OUT_DIR / "synthesis_readiness_deepaudit_run_meta_20260606.json"

SCSCORE_ROOT = ROOT / "retrosynthesis_tools/scscore"
SYBA_ROOT = ROOT / "retrosynthesis_tools/syba"
RASCORE_ROOT = ROOT / "retrosynthesis_tools/RAscore"

BENCHMARKS = {
    "TNT": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "TATB": "Nc1c(N)c(N)c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "RDX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "HMX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "PETN": "C(CO[N+](=O)[O-])(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "NTO": "O=c1[nH]n[nH]c(=O)n1",
}

SMARTS = {
    "Nitro_or_Nitrate_Count": "[N+](=O)[O-]",
    "C_Nitro_Count": "[#6][N+](=O)[O-]",
    "N_Nitro_Nitramine_Count": "[#7][N+](=O)[O-]",
    "Nitrate_Ester_Count": "[OX2][N+](=O)[O-]",
    "Amino_Count": "[NX3;H2,H1;!$(NC=O)]",
    "Azo_Diazo_Count": "[NX2]=[NX2]",
    "Azide_Count": "[N-]=[N+]=N",
    "Peroxide_Like_Count": "[OX2][OX2]",
    "Furazan_Like_Count": "o1nncc1",
    "Tetrazole_Like_Count": "n1nnnc1",
}
PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in SMARTS.items()}


def mol_from_smiles(smiles: Any) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(smiles: Any) -> str:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""


def fp_from_mol(mol: Optional[Chem.Mol]):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def scaffold_smiles(mol: Optional[Chem.Mol]) -> str:
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def numeric(v: Any, default: float = math.nan) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def load_scscore():
    sys.path.insert(0, str(SCSCORE_ROOT))
    from scscore.standalone_model_numpy import SCScorer

    path = SCSCORE_ROOT / "models/full_reaxys_model_1024bool/model.ckpt-10654.as_numpy.json.gz"
    return SCScorer().restore(str(path))


def load_syba():
    sys.path.insert(0, str(SYBA_ROOT))
    from syba.syba import SybaClassifier

    scorer = SybaClassifier()
    scorer.fitDefaultScore()
    return scorer


def rascore_status_probe() -> Tuple[str, str]:
    sys.path.insert(0, str(RASCORE_ROOT))
    try:
        from RAscore import RAscore_XGB

        model_path = RASCORE_ROOT / "RAscore/models/XGB_chembl_ecfp_counts/model.pkl"
        scorer = RAscore_XGB.RAScorerXGB(model_path=str(model_path))
        _ = scorer.predict(BENCHMARKS["TNT"])
        return "available", "XGB_chembl_ecfp_counts"
    except Exception as exc:
        note = f"{type(exc).__name__}: {str(exc)[:220]}"
        note = " ".join(note.replace("\r", " ").replace("\n", " ").split())
        return (
            "not_computed_model_incompatible",
            note,
        )


def pubchem_get_text(url: str, timeout: int = 45) -> Tuple[str, str]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return "ok", resp.text.strip()
        return f"http_{resp.status_code}", resp.text[:160].replace("\n", " ")
    except Exception as exc:
        return f"error_{type(exc).__name__}", str(exc)[:160]


def pubchem_query(smiles: str, sleep_s: float = 0.25) -> Dict[str, Any]:
    enc = quote(smiles, safe="")
    exact_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{enc}/cids/TXT"
    sim_url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/smiles/"
        f"{enc}/cids/TXT?Threshold=90&MaxRecords=10"
    )
    exact_status, exact_text = pubchem_get_text(exact_url)
    time.sleep(sleep_s)
    sim_status, sim_text = pubchem_get_text(sim_url)
    time.sleep(sleep_s)

    def cids(status: str, text: str) -> List[str]:
        if status != "ok":
            return []
        out = []
        for x in text.splitlines():
            token = x.strip()
            if token.isdigit() and int(token) > 0:
                out.append(token)
        return out

    exact = cids(exact_status, exact_text)
    sim = cids(sim_status, sim_text)
    if exact_status == "ok" and not exact:
        exact_status = "ok_no_positive_cid"
    if sim_status == "ok" and not sim:
        sim_status = "ok_no_positive_cid"
    return {
        "PubChem_Exact_Status": exact_status,
        "PubChem_Exact_CID_Count": len(exact),
        "PubChem_Exact_CIDs": ";".join(exact[:10]),
        "PubChem_Similarity90_Status": sim_status,
        "PubChem_Similarity90_CID_Count": len(sim),
        "PubChem_Similarity90_CIDs": ";".join(sim[:10]),
    }


def benchmark_nearest(mol: Optional[Chem.Mol], benches: List[Tuple[str, str, Any]]) -> Tuple[str, str, float]:
    f = fp_from_mol(mol)
    if f is None:
        return "", "", math.nan
    best = ("", "", -1.0)
    for name, smi, bfp in benches:
        if bfp is None:
            continue
        sim = float(DataStructs.TanimotoSimilarity(f, bfp))
        if sim > best[2]:
            best = (name, smi, sim)
    return best


def nearest_pool(
    mol: Optional[Chem.Mol],
    pool: pd.DataFrame,
    fp_col: str,
    exclude_molecule: str = "",
) -> Dict[str, Any]:
    f = fp_from_mol(mol)
    if f is None or pool.empty:
        return {
            "Molecule": "",
            "SMILES": "",
            "Source_Group": "",
            "Tanimoto": math.nan,
        }
    best_idx = None
    best_sim = -1.0
    for idx, row in pool.iterrows():
        if str(row.get("Molecule", "")) == exclude_molecule:
            continue
        other_fp = row.get(fp_col)
        if other_fp is None:
            continue
        sim = float(DataStructs.TanimotoSimilarity(f, other_fp))
        if sim > best_sim:
            best_sim = sim
            best_idx = idx
    if best_idx is None:
        return {"Molecule": "", "SMILES": "", "Source_Group": "", "Tanimoto": math.nan}
    row = pool.loc[best_idx]
    return {
        "Molecule": row.get("Molecule", ""),
        "SMILES": row.get("SMILES", ""),
        "Source_Group": row.get("Source_Group", ""),
        "Tanimoto": best_sim,
    }


def substructure_counts(mol: Optional[Chem.Mol]) -> Dict[str, int]:
    out = {}
    for name, patt in PATTERNS.items():
        if mol is None or patt is None:
            out[name] = 0
        else:
            out[name] = len(mol.GetSubstructMatches(patt))
    return out


def formula_counts(mol: Optional[Chem.Mol]) -> Dict[str, int]:
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    if mol is None:
        return counts
    molh = Chem.AddHs(mol)
    for atom in molh.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts


def classify_tier(row: pd.Series) -> Tuple[str, str]:
    ai_solved = numeric(row.get("AiZynthFinder_Number_Of_Solved_Routes"), 0.0)
    ask_paths = numeric(row.get("ASKCOS_Pathways_Returned"), 0.0)
    pub_exact = numeric(row.get("PubChem_Exact_CID_Count"), 0.0)
    pub_sim = numeric(row.get("PubChem_Similarity90_CID_Count"), 0.0)
    syba = numeric(row.get("SYBA_Score"), math.nan)
    scscore = numeric(row.get("SCScore_Reaxys_1024"), math.nan)
    sa = numeric(row.get("SAscore"), numeric(row.get("SA_Score"), math.nan))
    base_sim = numeric(row.get("Nearest_BASE_Tanimoto"), math.nan)
    bench_sim = numeric(row.get("Nearest_Benchmark_Tanimoto"), math.nan)

    if ai_solved > 0 or ask_paths > 0:
        return "Tier_A_route_tool_supported", "At least one external route planner returned a complete route/pathway."
    if pub_exact > 0:
        return "Tier_B_registry_exact_hit_route_unresolved", "PubChem exact registry hit found, but ASKCOS/AiZynthFinder did not return a complete buyable route."
    if pub_sim > 0 and syba > 0 and scscore <= 3.5:
        return "Tier_B_near_analogue_supported_route_unresolved", "PubChem 90% similar analogue(s) plus favorable SYBA/SCScore, but no complete route."
    if syba > 0 and scscore <= 3.5 and (base_sim >= 0.45 or bench_sim >= 0.20) and sa <= 5.5:
        return "Tier_C_descriptor_plausible_route_unresolved", "Descriptor and similarity evidence are plausible, but external route planners did not solve the molecule."
    if syba > 0 or scscore <= 3.5 or sa <= 5.0:
        return "Tier_C_partial_descriptor_support_high_route_risk", "Some descriptor-level support exists, but route and precedent evidence remain weak."
    return "Tier_D_high_synthesis_risk", "No complete route and weak descriptor/precedent support under current evidence."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "main_tables").mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "si_tables").mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "supplementary_data").mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "code_release").mkdir(parents=True, exist_ok=True)
    (PKG_DIR / "internal_audit").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DB_PATH)
    if "Final_Detonation_Rank" not in df.columns:
        raise RuntimeError("Final_Detonation_Rank missing from official database")
    df["Final_Detonation_Rank"] = pd.to_numeric(df["Final_Detonation_Rank"], errors="coerce")
    df = df.sort_values("Final_Detonation_Rank").reset_index(drop=True)
    mols = [mol_from_smiles(s) for s in df["SMILES"]]

    print("Loading SCScore and SYBA scorers")
    scscore = load_scscore()
    syba = load_syba()
    ras_status, ras_note = rascore_status_probe()
    print("RAscore status", ras_status, ras_note[:120])

    sc_vals = []
    syba_vals = []
    syba_classes = []
    canonical = []
    scaffolds = []
    for i, (smi, mol) in enumerate(zip(df["SMILES"], mols), start=1):
        if i % 500 == 0:
            print("scored", i, "of", len(df), flush=True)
        canonical.append(canonical_smiles(smi))
        scaffolds.append(scaffold_smiles(mol))
        try:
            sc_vals.append(float(scscore.get_score_from_smi(str(smi))[1]))
        except Exception:
            sc_vals.append(math.nan)
        try:
            val = float(syba.predict(mol=mol)) if mol is not None else math.nan
        except Exception:
            val = math.nan
        syba_vals.append(val)
        syba_classes.append("easy_synthesis_like" if pd.notna(val) and val > 0 else "hard_synthesis_like")

    df["SCScore_Reaxys_1024"] = sc_vals
    df["SYBA_Score"] = syba_vals
    df["SYBA_Class"] = syba_classes
    df["RAscore_Value"] = np.nan
    df["RAscore_Status"] = ras_status
    df["RAscore_Note"] = ras_note
    df["Bemis_Murcko_Scaffold"] = scaffolds

    df["_fp"] = [fp_from_mol(m) for m in mols]
    bench_rows = []
    for name, smi in BENCHMARKS.items():
        bm = mol_from_smiles(smi)
        bench_rows.append((name, smi, fp_from_mol(bm)))

    base_pool = df[df["Source_Group"].astype(str).eq("BASE")][["Molecule", "SMILES", "Source_Group", "_fp"]].copy()
    full_pool = df[["Molecule", "SMILES", "Source_Group", "_fp"]].copy()
    top20_idx = df.index[df["Final_Detonation_Rank"].between(1, 20, inclusive="both")].tolist()

    top_records = []
    for idx in top20_idx:
        row = df.loc[idx]
        mol = mols[idx]
        molecule = str(row.get("Molecule", ""))
        bname, bsmi, bsim = benchmark_nearest(mol, bench_rows)
        near_base = nearest_pool(mol, base_pool, "_fp", exclude_molecule=molecule)
        near_full = nearest_pool(mol, full_pool, "_fp", exclude_molecule=molecule)
        counts = formula_counts(mol)
        groups = substructure_counts(mol)
        pubchem = pubchem_query(str(row.get("SMILES", "")))
        scaf = row.get("Bemis_Murcko_Scaffold", "")
        scaf_base_count = int((df["Source_Group"].astype(str).eq("BASE") & df["Bemis_Murcko_Scaffold"].eq(scaf)).sum()) if scaf else 0
        scaf_all_count = int(df["Bemis_Murcko_Scaffold"].eq(scaf).sum()) if scaf else 0

        rec: Dict[str, Any] = {
            "Final_Global_Rank": int(row["Final_Detonation_Rank"]),
            "Molecule": molecule,
            "SMILES": row.get("SMILES", ""),
            "Canonical_SMILES": row.get("_canonical_smiles_iso", "") or canonical_smiles(row.get("SMILES", "")),
            "Final_Detonation_D(km/s)": row.get("Final_Detonation_D(km/s)", np.nan),
            "Final_Detonation_P(GPa)": row.get("Final_Detonation_P(GPa)", np.nan),
            "Final_Detonation_Density_Used(g/cm3)": row.get("Final_Detonation_Density_Used(g/cm3)", np.nan),
            "Final_Detonation_HOF_Used(kcal/mol)": row.get("Final_Detonation_HOF_Used(kcal/mol)", np.nan),
            "SAscore": row.get("SAscore", row.get("SA_Score", np.nan)),
            "SCScore_Reaxys_1024": row.get("SCScore_Reaxys_1024", np.nan),
            "SYBA_Score": row.get("SYBA_Score", np.nan),
            "SYBA_Class": row.get("SYBA_Class", ""),
            "RAscore_Value": np.nan,
            "RAscore_Status": ras_status,
            "RAscore_Note": ras_note,
            "Bemis_Murcko_Scaffold": scaf,
            "Scaffold_Count_In_Final_DB": scaf_all_count,
            "Scaffold_Count_In_BASE": scaf_base_count,
            "Nearest_Benchmark": bname,
            "Nearest_Benchmark_SMILES": bsmi,
            "Nearest_Benchmark_Tanimoto": bsim,
            "Nearest_BASE_Molecule": near_base["Molecule"],
            "Nearest_BASE_SMILES": near_base["SMILES"],
            "Nearest_BASE_Tanimoto": near_base["Tanimoto"],
            "Nearest_FinalDB_Molecule": near_full["Molecule"],
            "Nearest_FinalDB_Source_Group": near_full["Source_Group"],
            "Nearest_FinalDB_Tanimoto": near_full["Tanimoto"],
            "Formula_C": counts["C"],
            "Formula_H": counts["H"],
            "Formula_N": counts["N"],
            "Formula_O": counts["O"],
            "N_C_Ratio": counts["N"] / max(counts["C"], 1),
            "O_C_Ratio": counts["O"] / max(counts["C"], 1),
            "ExactMolWt": Descriptors.ExactMolWt(mol) if mol is not None else np.nan,
            "Ring_Count": rdMolDescriptors.CalcNumRings(mol) if mol is not None else np.nan,
            "Bridgehead_Atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol) if mol is not None else np.nan,
            "Spiro_Atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol) if mol is not None else np.nan,
            "AiZynthFinder_Number_Of_Routes": row.get("AiZynthFinder_Number_Of_Routes", np.nan),
            "AiZynthFinder_Number_Of_Solved_Routes": row.get("AiZynthFinder_Number_Of_Solved_Routes", np.nan),
            "AiZynthFinder_Top_Score": row.get("AiZynthFinder_Top_Score", np.nan),
            "ASKCOS_Total_Iterations": row.get("ASKCOS_Total_Iterations", np.nan),
            "ASKCOS_Total_Reactions": row.get("ASKCOS_Total_Reactions", np.nan),
            "ASKCOS_Pathways_Returned": row.get("ASKCOS_Pathways_Returned", np.nan),
        }
        rec.update(groups)
        rec.update(pubchem)
        tier, rationale = classify_tier(pd.Series(rec))
        rec["Final_Synthesis_Readiness_Tier"] = tier
        rec["Final_Synthesis_Readiness_Rationale"] = rationale
        top_records.append(rec)

        for k, v in rec.items():
            if k in {
                "Final_Global_Rank",
                "Molecule",
                "SMILES",
                "Canonical_SMILES",
                "Final_Detonation_D(km/s)",
                "Final_Detonation_P(GPa)",
                "Final_Detonation_Density_Used(g/cm3)",
                "Final_Detonation_HOF_Used(kcal/mol)",
                "Formula_C",
                "Formula_H",
                "Formula_N",
                "Formula_O",
                "ExactMolWt",
                "Ring_Count",
                "Bridgehead_Atoms",
                "Spiro_Atoms",
            }:
                continue
            db_col = k
            df.loc[idx, db_col] = v

    top = pd.DataFrame(top_records).sort_values("Final_Global_Rank")
    score_cols = [
        "Final_Detonation_Rank",
        "Molecule",
        "SMILES",
        "Source_Group",
        "SAscore",
        "SCScore_Reaxys_1024",
        "SYBA_Score",
        "SYBA_Class",
        "RAscore_Value",
        "RAscore_Status",
        "Bemis_Murcko_Scaffold",
    ]
    scores = df[[c for c in score_cols if c in df.columns]].copy()

    top_csv = OUT_DIR / "Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv"
    scores_csv = OUT_DIR / "Table_Final_Database_Synthesizability_Scores_5432_20260606.csv"
    top.to_csv(top_csv, index=False)
    scores.to_csv(scores_csv, index=False)

    db_out = df.drop(columns=["_fp"], errors="ignore")
    db_out.to_csv(DB_PATH, index=False)
    frozen_pkg = PKG_DIR / "supplementary_data/Supplementary_Data_1_Final_Frozen_Database_5432.csv"
    al08_pkg = PKG_DIR / "supplementary_data/Supplementary_Data_1_Final_AL08_Database.csv"
    db_out.to_csv(frozen_pkg, index=False)
    db_out.to_csv(al08_pkg, index=False)

    main_table = PKG_DIR / "main_tables/Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv"
    si_scores = PKG_DIR / "si_tables/Table_Final_Database_Synthesizability_Scores_5432_20260606.csv"
    top.to_csv(main_table, index=False)
    scores.to_csv(si_scores, index=False)
    top.to_csv(TABLES_DIR / "Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv", index=False)
    scores.to_csv(TABLES_DIR / "Table_Final_Database_Synthesizability_Scores_5432_20260606.csv", index=False)

    tier_counts = top["Final_Synthesis_Readiness_Tier"].value_counts().to_dict()
    syba_counts = scores["SYBA_Class"].value_counts().to_dict()
    md = f"""# Final Top20 Synthesis-Readiness Deep Audit 2026-06-06

## Scope

- Database: final frozen 5432-molecule database sorted by `Final_Detonation_Rank`.
- Top20: final global Top20 from the full frozen database, not AL08-only.
- New evidence layers: SCScore, SYBA, PubChem exact/similarity registry search, internal BASE/benchmark nearest-neighbor similarity, scaffold recurrence, ASKCOS/AiZynthFinder route-audit fields.

## Main Result

- Top20 rows audited: {len(top)}.
- SCScore and SYBA were computed for all {len(scores)} final database rows.
- RAScore status: `{ras_status}`. Note: {ras_note}
- Top20 synthesis-readiness tiers: {json.dumps(tier_counts, ensure_ascii=False)}.
- Full-database SYBA classes: {json.dumps(syba_counts, ensure_ascii=False)}.

## Interpretation Boundary

The new evidence stack improves synthesizability analysis depth beyond SAscore alone, but it still does not prove experimental synthesizability. A route-proven claim would require either complete route planner solutions with validated purchasable precursors, expert retrosynthetic route design with literature precedents, or experimental synthesis.

## Outputs

- `{top_csv.relative_to(ROOT)}`
- `{scores_csv.relative_to(ROOT)}`
- `{main_table.relative_to(ROOT)}`
- `{si_scores.relative_to(ROOT)}`
- `{frozen_pkg.relative_to(ROOT)}`

## Manuscript-Ready Claim

The final global Top20 were subjected to a multi-layer synthesis-readiness audit combining heuristic complexity scores, fragment-based SYBA classification, Reaxys-trained SCScore, public-registry similarity checks, internal scaffold/nearest-neighbor evidence, and external ASKCOS/AiZynthFinder route-planning audits. The candidates should be described as high-performance computational leads with explicit synthesis-readiness tiers, not as experimentally route-proven materials.
"""
    AUDIT_MD.write_text(md)
    (PKG_DIR / "internal_audit/Final_Top20_Synthesis_Readiness_DeepAudit_20260606.md").write_text(md)

    meta = {
        "db_rows": int(len(df)),
        "db_columns": int(len(db_out.columns)),
        "top20_rows": int(len(top)),
        "outputs": {
            "top20_deep_audit": str(top_csv),
            "database_scores": str(scores_csv),
            "official_database": str(DB_PATH),
            "package_frozen_database": str(frozen_pkg),
            "audit_md": str(AUDIT_MD),
        },
        "tier_counts": tier_counts,
        "syba_counts": syba_counts,
        "rascore_status": ras_status,
        "rascore_note": ras_note,
    }
    RUN_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
