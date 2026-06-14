from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DB = ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
TARGET = ROOT / "data/baselines/target_matrix_10d.csv"
XTB = ROOT / "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv"
REPORT = ROOT / "manuscript_npJ/Final_Database_Detonation_Sort_Report_20260605.md"


def mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def formula_counts(m) -> dict[str, int]:
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    if m is None:
        return counts
    mh = Chem.AddHs(m)
    for atom in mh.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts


def kj(c: int, h: int, n: int, o: int, mw: float, hof_kcal_mol: float, density: float) -> tuple[float, float]:
    if mw <= 0 or density <= 0 or not math.isfinite(hof_kcal_mol):
        return float("nan"), float("nan")
    o_avail = float(o)
    h2o = min(h / 2.0, o_avail)
    o_avail -= h2o
    h2 = (h / 2.0) - h2o
    co = min(float(c), o_avail)
    o_avail -= co
    co2 = min(co, o_avail)
    o_avail -= co2
    co -= co2
    o2 = max(o_avail, 0.0) / 2.0
    n2 = n / 2.0
    gas_moles = h2o + co + co2 + o2 + n2 + h2
    if gas_moles <= 0:
        return float("nan"), float("nan")
    gas_mass = h2o * 18.015 + co * 28.01 + co2 * 44.01 + o2 * 31.998 + n2 * 28.013 + h2 * 2.016
    N = gas_moles / mw
    M = gas_mass / gas_moles
    hof_products = h2o * -57.8 + co * -26.4 + co2 * -94.0
    q_heat = (hof_kcal_mol - hof_products) / mw * 1000.0
    if q_heat <= 0:
        return float("nan"), float("nan")
    D = 1.01 * (N * (M**0.5) * (q_heat**0.5)) ** 0.5 * (1.0 + 1.30 * density)
    P = 1.558 * density**2 * N * (M**0.5) * (q_heat**0.5)
    return round(float(D), 4), round(float(P), 4)


def source_prefix(mid: str) -> str:
    s = str(mid)
    for p in ["AL08", "AL07", "AL06", "AL05", "AL04", "GPT_AL"]:
        if s.startswith(p):
            return p
    return "BASE"


def add_detonation_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        m = mol(r.get("SMILES", ""))
        if m is None:
            rows.append((float("nan"), float("nan"), float("nan"), "invalid_smiles", float("nan"), "Kamlet-Jacobs_from_database_HOF_density"))
            continue
        counts = formula_counts(m)
        try:
            mw = float(r.get("Molecular_Weight", Descriptors.MolWt(m)))
        except Exception:
            mw = float(Descriptors.MolWt(m))

        density_source = "Density_calibrated(g/cm3)"
        density = pd.to_numeric(pd.Series([r.get("Density_calibrated(g/cm3)")]), errors="coerce").iloc[0]
        if not pd.notna(density):
            density_source = "Density_calc(g/cm3)"
            density = pd.to_numeric(pd.Series([r.get("Density_calc(g/cm3)")]), errors="coerce").iloc[0]
        hof = pd.to_numeric(pd.Series([r.get("Heat_of_Formation(kcal/mol)")]), errors="coerce").iloc[0]
        D, P = kj(counts["C"], counts["H"], counts["N"], counts["O"], mw, float(hof), float(density)) if pd.notna(density) and pd.notna(hof) else (float("nan"), float("nan"))
        rows.append((D, P, density, density_source, hof, "Kamlet-Jacobs_from_database_HOF_density"))

    add = pd.DataFrame(
        rows,
        columns=[
            "Final_Detonation_D(km/s)",
            "Final_Detonation_P(GPa)",
            "Final_Detonation_Density_Used(g/cm3)",
            "Final_Detonation_Density_Source",
            "Final_Detonation_HOF_Used(kcal/mol)",
            "Final_Detonation_Method",
        ],
    )
    # Remove stale generated columns before inserting fresh values.
    stale = [c for c in df.columns if c.startswith("Final_Detonation_") or c == "Source_Group"]
    df = df.drop(columns=stale, errors="ignore").reset_index(drop=True)
    df.insert(0, "Source_Group", df["Molecule"].map(source_prefix))
    for col in add.columns[::-1]:
        df.insert(5, col, add[col].values)
    df = df.sort_values(
        ["Final_Detonation_D(km/s)", "Final_Detonation_P(GPa)"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    df.insert(0, "Final_Detonation_Rank", range(1, len(df) + 1))
    return df


def alignment_report(df_sorted: pd.DataFrame) -> str:
    messages = []
    messages.append(f"Official database rows: {len(df_sorted)}")
    for label, path in [("target_matrix_10d", TARGET), ("xTB_aligned", XTB)]:
        if not path.exists():
            messages.append(f"{label}: MISSING")
            continue
        other = pd.read_csv(path, usecols=["Molecule"])
        overlap = int(df_sorted["Molecule"].isin(set(other["Molecule"])).sum())
        messages.append(f"{label}: rows={len(other)}, overlap_with_official_db={overlap}, missing={len(df_sorted)-overlap}")
    valid_d = int(df_sorted["Final_Detonation_D(km/s)"].notna().sum())
    messages.append(f"Rows with computed final detonation D/P: {valid_d}")
    messages.append("Top 20 after in-place database sorting:")
    cols = ["Final_Detonation_Rank", "Molecule", "Source_Group", "Final_Detonation_D(km/s)", "Final_Detonation_P(GPa)", "SMILES"]
    messages.append(df_sorted[cols].head(20).to_markdown(index=False))
    return "\n\n".join(messages) + "\n"


def main() -> None:
    df = pd.read_csv(DB)
    before_rows = len(df)
    sorted_df = add_detonation_columns(df)
    if len(sorted_df) != before_rows:
        raise RuntimeError(f"Row count changed unexpectedly: before={before_rows}, after={len(sorted_df)}")
    sorted_df.to_csv(DB, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("# Final Database Detonation Sort Report\n\n" + alignment_report(sorted_df), encoding="utf-8")
    print(f"[OK] Sorted official database in place: {DB}")
    print(f"[OK] Report: {REPORT}")
    print(f"[OK] Rows: {len(sorted_df)}")
    print(sorted_df[["Final_Detonation_Rank", "Molecule", "Source_Group", "Final_Detonation_D(km/s)", "Final_Detonation_P(GPa)"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
