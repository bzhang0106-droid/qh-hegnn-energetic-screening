from __future__ import annotations

from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem


ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DB = ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
TARGET_MATRIX = ROOT / "data/curated_molecule_clean_v1/target_matrix_10d_molecule_clean.csv"
TOP20_FILES = [
    ROOT / "results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_ExternalRetrosynthesis_10D.csv",
    ROOT / "results/final_global_top20/synthesis_readiness/Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv",
    ROOT / "manuscript_npJ/final_tables_figures/Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv",
    ROOT / "manuscript_npJ/final_submission_package_AL08_20260605/main_tables/Table_Final_Global_Top20_Structure_Property_Synthesizability_ExternalRetrosynthesis_10D.csv",
    ROOT / "manuscript_npJ/final_submission_package_AL08_20260605/main_tables/Table_Final_Global_Top20_Synthesis_Readiness_DeepAudit_20260606.csv",
]
PKG = ROOT / "manuscript_npJ/final_submission_package_AL08_20260605"
MAJOR = ROOT / "manuscript_npJ/major_revision_20260607"
SI_TABLES = PKG / "si_tables"
SUPP_DATA = PKG / "supplementary_data"
INTERNAL = PKG / "internal_audit"
CODE_RELEASE = PKG / "code_release"

for p in [MAJOR, SI_TABLES, SUPP_DATA, INTERNAL, CODE_RELEASE]:
    p.mkdir(parents=True, exist_ok=True)


def count_chno(smiles: object) -> tuple[int, int, int, int, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return 0, 0, 0, 0, np.nan
    mol = Chem.AddHs(mol)
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    masses = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    mw = sum(masses[k] * v for k, v in counts.items())
    return counts["C"], counts["H"], counts["N"], counts["O"], mw


def heat_of_explosion_cal_g(c: int, h: int, n: int, o: int, mw: float, hof_kcal_mol: float) -> float:
    if mw <= 0 or not np.isfinite(mw) or not np.isfinite(hof_kcal_mol):
        return np.nan
    o_avail = float(o)
    h2o_moles = min(h / 2.0, o_avail)
    o_avail -= h2o_moles
    co_moles = min(float(c), o_avail)
    o_avail -= co_moles
    co2_moles = min(co_moles, o_avail)
    o_avail -= co2_moles
    co_moles -= co2_moles
    # Same product enthalpy convention as scripts/07_kamlet_jacobs_eval.py.
    hof_products = (h2o_moles * -57.8) + (co_moles * -26.4) + (co2_moles * -94.0)
    q_cal_g = (hof_kcal_mol - hof_products) / mw * 1000.0
    return float(q_cal_g) if q_cal_g > 0 else np.nan


def add_q_columns(df: pd.DataFrame, hof_col: str) -> pd.DataFrame:
    q_vals = []
    q_src = []
    for _, row in df.iterrows():
        smiles = row.get("SMILES", row.get("Canonical_SMILES", ""))
        c, h, n, o, mw = count_chno(smiles)
        hof = pd.to_numeric(pd.Series([row.get(hof_col, np.nan)]), errors="coerce").iloc[0]
        q = heat_of_explosion_cal_g(c, h, n, o, mw, hof)
        q_vals.append(q)
        q_src.append("Kamlet_Jacobs_product_enthalpy_from_final_HOF_formula" if np.isfinite(q) else "not_computed_missing_smiles_or_HOF")
    df["Final_Detonation_Q(cal/g)"] = q_vals
    df["Final_Detonation_Q(kJ/g)"] = pd.to_numeric(df["Final_Detonation_Q(cal/g)"], errors="coerce") * 0.004184
    df["Final_Detonation_Q_Source"] = q_src
    return df


def reorder_q_near_detonation(df: pd.DataFrame) -> pd.DataFrame:
    q_cols = ["Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"]
    cols = [c for c in df.columns if c not in q_cols]
    insert_after = "Final_Detonation_P(GPa)" if "Final_Detonation_P(GPa)" in cols else None
    if insert_after is None:
        return df[cols + q_cols]
    i = cols.index(insert_after) + 1
    return df[cols[:i] + q_cols + cols[i:]]


def update_top20(path: Path, q_lookup: pd.DataFrame) -> dict:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "status": "missing"}
    df = pd.read_csv(path)
    for c in ["Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    key = "Molecule"
    if key not in df.columns:
        return {"path": str(path.relative_to(ROOT)), "status": "no_molecule_column"}
    out = df.merge(q_lookup, on="Molecule", how="left")
    cols = list(out.columns)
    q_cols = ["Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"]
    cols = [c for c in cols if c not in q_cols]
    after_candidates = ["Final_Global_P(GPa)", "Final_Detonation_P(GPa)", "Final_Detonation_P(GPa)"]
    insert_after = next((c for c in after_candidates if c in cols), None)
    if insert_after:
        i = cols.index(insert_after) + 1
        out = out[cols[:i] + q_cols + cols[i:]]
    else:
        out = out[cols + q_cols]
    out.to_csv(path, index=False)
    return {"path": str(path.relative_to(ROOT)), "status": "updated", "rows": len(out), "q_nonmissing": int(out["Final_Detonation_Q(cal/g)"].notna().sum())}


def main() -> None:
    db = pd.read_csv(DB)
    hof_col = "Final_Detonation_HOF_Used(kcal/mol)" if "Final_Detonation_HOF_Used(kcal/mol)" in db.columns else "Heat_of_Formation(kcal/mol)"
    db = add_q_columns(db, hof_col)
    db = reorder_q_near_detonation(db)
    db.to_csv(DB, index=False)

    q_lookup = db[["Molecule", "Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"]].drop_duplicates("Molecule")
    q_lookup.to_csv(MAJOR / "Table_S_Final_Database_Q_HeatOfExplosion_Index_20260607.csv", index=False)
    q_lookup.to_csv(SI_TABLES / "Table_S_Final_Database_Q_HeatOfExplosion_Index_20260607.csv", index=False)

    tm_update = {"status": "not_modified", "reason": "Q is a derived K-J output, not an independent 10D training label."}
    if TARGET_MATRIX.exists():
        tm = pd.read_csv(TARGET_MATRIX)
        if "Final_Detonation_Q(cal/g)" in tm.columns:
            tm = tm.drop(columns=["Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"], errors="ignore")
        q_by_row = db[["Molecule", "SMILES", "Final_Detonation_Q(cal/g)", "Final_Detonation_Q(kJ/g)", "Final_Detonation_Q_Source"]]
        # Preserve a derived, non-training target matrix companion rather than altering the formal 10D label matrix.
        companion = tm.merge(q_by_row.drop_duplicates(["Molecule", "SMILES"]), on=["Molecule", "SMILES"], how="left")
        companion_path = ROOT / "data/curated_molecule_clean_v1/target_matrix_10d_plus_derived_Q_for_reporting.csv"
        companion.to_csv(companion_path, index=False)
        tm_update = {"status": "companion_written", "path": str(companion_path.relative_to(ROOT)), "rows": len(companion), "q_nonmissing": int(companion["Final_Detonation_Q(cal/g)"].notna().sum())}

    top_updates = [update_top20(p, q_lookup) for p in TOP20_FILES]

    # Refresh final frozen supplementary database in the package.
    for name in [
        "Supplementary_Data_1_Final_Frozen_Database_5432.csv",
        "Supplementary_Data_1_Final_AL08_Database.csv",
    ]:
        shutil.copy2(DB, SUPP_DATA / name)

    status = {
        "updated_at": "2026-06-07",
        "database": str(DB.relative_to(ROOT)),
        "database_rows": int(len(db)),
        "q_nonmissing": int(db["Final_Detonation_Q(cal/g)"].notna().sum()),
        "q_mean_cal_g": float(pd.to_numeric(db["Final_Detonation_Q(cal/g)"], errors="coerce").mean()),
        "q_top20_mean_cal_g": float(pd.to_numeric(db.sort_values("Final_Detonation_Rank").head(20)["Final_Detonation_Q(cal/g)"], errors="coerce").mean()),
        "target_matrix_policy": tm_update,
        "top20_updates": top_updates,
        "claim_boundary": "Q is reported as a derived Kamlet-Jacobs heat-of-explosion output sharing the same HOF/formula assumptions used for D and P. It is not added as an independent 11th training target in this update.",
    }
    status_path = MAJOR / "NPJ_Q_Output_Addition_Status_20260607.md"
    lines = [
        "# Heat of Explosion Q Output Addition",
        "",
        "Generated: 2026-06-07",
        "",
        "Q has been added as an explicit derived thermochemical output for final database and Top20 reporting.",
        "",
        f"- Final database rows: {len(db)}",
        f"- Non-missing Q rows: {status['q_nonmissing']}",
        f"- Mean Q over final database: {status['q_mean_cal_g']:.2f} cal g-1",
        f"- Mean Q over final Top20: {status['q_top20_mean_cal_g']:.2f} cal g-1",
        "",
        "## Policy",
        "",
        status["claim_boundary"],
        "",
        "## Target Matrix",
        "",
        json.dumps(tm_update, indent=2),
        "",
        "## Updated Top20 Tables",
        "",
    ]
    for item in top_updates:
        lines.append(f"- {item}")
    status_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(status_path, INTERNAL / status_path.name)

    script_path = ROOT / "scripts/add_final_heat_of_explosion_q_outputs_20260607.py"
    if script_path.exists():
        shutil.copy2(script_path, CODE_RELEASE / script_path.name)

    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
