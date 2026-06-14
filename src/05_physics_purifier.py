"""
05_physics_purifier.py

Manual HELS workflow physics-informed purifier.

Main corrections vs. the previous version:
1. Implements an actual 10-dimensional Pareto ranking over the 10 surrogate targets.
2. Keeps strict redlines for chemically meaningful failure modes.
3. Distinguishes molecular-volume-derived density proxy from true crystal density.
4. Recomputes RDKit SA score as a reliable synthesis-accessibility objective while preserving the model's predicted SAscore column.

10 Pareto dimensions:
Maximize: density proxy, heat of formation, HOMO-LUMO gap, trigger-bond rho, vertical weakest-bond BDE.
Minimize: SA score, VS_max, Sigma2_tot, Nu, molecular weight.

Important:
- This script only selects candidates for ORCA verification; it does not change the ORCA validation logic.
"""

import os
import argparse
import sys
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ImportError:
    class MockSAScorer:
        def calculateScore(self, mol):
            return 3.0
    sascorer = MockSAScorer()

INPUT_CANDIDATES = [
    "../results/Surrogate_10Target_Predictions.csv",
    "../results/Surrogate_10D_Predictions.csv",
    "../results/Surrogate_9Target_Predictions.csv",
    "../results/Surrogate_9D_Predictions.csv",
]
OUTPUT_CSV = "../results/Pareto_Optimal_Candidates.csv"
REJECTED_CSV = "../results/Purifier_Rejected_Candidates.csv"

EXISTING_DB_CANDIDATES = [
    "../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv",
    "../results/final_model_release/clean_sanitized_v1/training_dataset_clean_v1.csv",
]

# -------------------- physical redlines --------------------
# Density is not a guaranteed crystal density unless explicit crystal labels are supplied.
MIN_DENSITY_PROXY = 1.60
MIN_HOF = 0.0
MIN_GAP = 3.0
MIN_TRIGGER_RHO = 0.165
MAX_VS_MAX = 55.0
MAX_SA_SCORE = 5.5
MAX_MW = 520.0
MIN_BDE = 0.0
MAX_NC_RATIO = 3.0

# Optional hard limits for ESP variance/balance are deliberately left disabled.
# They remain Pareto objectives because literature thresholds are system-dependent.
MAX_SIGMA2 = None
MAX_NU = None

TARGET_NAMES = {
    "density": "Density_calc(g/cm3)",
    "hof": "Heat_of_Formation(kcal/mol)",
    "gap": "HOMO_LUMO_Gap(eV)",
    "sa_pred": "SAscore",
    "vsmax": "VS_max",
    "sigma2": "Sigma2_tot",
    "nu": "Nu",
    "rho": "Trigger_Bond_Rho",
    "mw": "Molecular_Weight",
    "bde": "Vertical_BDE(kcal/mol)",
}


def choose_input_path() -> Optional[str]:
    for path in INPUT_CANDIDATES:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


def resolve_col(df: pd.DataFrame, base_name: str) -> str:
    """Resolve either predicted or raw target column names."""
    candidates = [f"Pred_{base_name}", base_name]
    for c in candidates:
        if c in df.columns:
            return c

    # Mild fallback for old inconsistent headers.
    simplified = base_name.lower().replace("_", "").replace("(", "").replace(")", "").replace("/", "").replace(" ", "")
    for c in df.columns:
        c_s = c.lower().replace("pred_", "").replace("_", "").replace("(", "").replace(")", "").replace("/", "").replace(" ", "")
        if simplified in c_s or c_s in simplified:
            return c
    raise KeyError(f"无法在输入表中解析目标列: {base_name}")


def get_required_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = {key: resolve_col(df, name) for key, name in TARGET_NAMES.items()}
    return cols


def safe_sa_score(smiles: str) -> float:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return np.nan
        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


def structural_sanity(smiles: str) -> Dict[str, object]:
    """Hard chemistry filters before Pareto selection."""
    out = {"valid": False, "reason": "unknown", "canonical_smiles": None, "NC_ratio": np.nan}
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            out["reason"] = "invalid_smiles"
            return out
        if Chem.GetFormalCharge(mol) != 0:
            out["reason"] = "non_neutral"
            return out
        if Descriptors.NumRadicalElectrons(mol) > 0:
            out["reason"] = "radical"
            return out
        if len(Chem.GetMolFrags(mol)) > 1:
            out["reason"] = "disconnected_fragments"
            return out

        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        c_count = atoms.count("C")
        n_count = atoms.count("N")
        if c_count <= 0 or n_count <= 0:
            out["reason"] = "missing_C_or_N"
            return out
        nc_ratio = n_count / c_count
        out["NC_ratio"] = nc_ratio
        if nc_ratio > MAX_NC_RATIO:
            out["reason"] = "N/C_too_high"
            return out

        peroxide = Chem.MolFromSmarts("[OX2]-[OX2]")
        if peroxide is not None and mol.HasSubstructMatch(peroxide):
            out["reason"] = "peroxide"
            return out

        out.update({"valid": True, "reason": "pass", "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)})
        return out
    except Exception as exc:
        out["reason"] = f"exception:{exc}"
        return out


def get_pareto_front(df: pd.DataFrame, max_cols: Iterable[str], min_cols: Iterable[str], eps: float = 1e-12) -> pd.DataFrame:
    """Return rank-0 non-dominated front. NaN rows should be removed before calling."""
    max_cols = list(max_cols)
    min_cols = list(min_cols)
    scores = np.zeros((len(df), len(max_cols) + len(min_cols)), dtype=float)

    for i, col in enumerate(max_cols):
        scores[:, i] = -df[col].astype(float).values
    for i, col in enumerate(min_cols):
        scores[:, len(max_cols) + i] = df[col].astype(float).values

    n = len(scores)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        diff = scores - scores[i]
        dominated_by_other = np.any(np.all(diff <= eps, axis=1) & np.any(diff < -eps, axis=1))
        if dominated_by_other:
            is_pareto[i] = False
    return df.loc[is_pareto].copy()


def minmax_score(series: pd.Series, maximize: bool = True) -> pd.Series:
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return pd.Series(np.ones(len(values)) * 0.5, index=series.index)
    norm = (values - lo) / (hi - lo)
    return norm if maximize else (1.0 - norm)


def add_screening_score(df: pd.DataFrame, max_cols: List[str], min_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        df["Screening_Score_10Target"] = []
        return df
    score = pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    for col in max_cols:
        score += minmax_score(df[col], maximize=True)
    for col in min_cols:
        score += minmax_score(df[col], maximize=False)
    df["Screening_Score_10Target"] = score / (len(max_cols) + len(min_cols))
    return df



def load_existing_database_keys(paths: Optional[List[str]] = None) -> set:
    keys = set()
    for path in paths or EXISTING_DB_CANDIDATES:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if "Canonical_SMILES_NoIso" in d.columns:
            vals = d["Canonical_SMILES_NoIso"].dropna().astype(str)
            keys.update(vals.tolist())
        elif "SMILES" in d.columns:
            for smi in d["SMILES"].dropna().astype(str):
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    keys.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description="10-target physics-informed Pareto purifier with vertical BDE.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--rejected", default=REJECTED_CSV)
    parser.add_argument("--exclude_existing_db", action="store_true", default=True)
    parser.add_argument("--allow_existing_db_duplicates", action="store_true")
    args = parser.parse_args()

    print("==================================================")
    print("⚖️ 启动 10D Physics-Informed Pareto Purifier")
    print("==================================================")

    input_path = args.input or choose_input_path()
    if input_path is None:
        print(f"❌ 找不到 surrogate 预测文件。尝试路径: {INPUT_CANDIDATES}")
        return

    output_csv = args.output
    rejected_csv = args.rejected
    exclude_existing = args.exclude_existing_db and not args.allow_existing_db_duplicates

    df = pd.read_csv(input_path)
    if df.empty:
        print(f"⚠️ 输入文件为空: {input_path}")
        pd.DataFrame().to_csv(output_csv, index=False)
        return
    if "SMILES" not in df.columns:
        raise ValueError("输入预测表缺失 SMILES 列。")

    print(f"[INFO] 输入文件: {input_path} | 候选数: {len(df)}")
    cols = get_required_columns(df)
    print("[INFO] 9D 目标列映射:")
    for key, col in cols.items():
        print(f"  - {key:<10}: {col}")

    # 1. Structural sanity check and canonical deduplication.
    sanity_records = [structural_sanity(smi) for smi in df["SMILES"]]
    sanity_df = pd.DataFrame(sanity_records)
    df = pd.concat([df.reset_index(drop=True), sanity_df], axis=1)
    df["SA_Score_RDKit"] = df["SMILES"].apply(safe_sa_score)

    rejected = []
    df_bad_structure = df[~df["valid"]].copy()
    if not df_bad_structure.empty:
        df_bad_structure["Reject_Reason"] = df_bad_structure["reason"]
        rejected.append(df_bad_structure)

    df = df[df["valid"]].copy()
    df = df.drop_duplicates(subset=["canonical_smiles"], keep="first").reset_index(drop=True)

    if exclude_existing:
        existing_keys = load_existing_database_keys()
        if existing_keys:
            already = df["canonical_smiles"].isin(existing_keys)
            df_existing = df[already].copy()
            if not df_existing.empty:
                df_existing["Reject_Reason"] = "already_in_training_database"
                rejected.append(df_existing)
                print(f"[INFO] 已排除已有数据库重复分子: {len(df_existing)}")
            df = df[~already].copy().reset_index(drop=True)
        print(f"[INFO] Existing database duplicate exclusion: {exclude_existing}")

    # 2. Numeric cleaning.
    numeric_cols = list(cols.values()) + ["SA_Score_RDKit", "NC_ratio"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Pareto_SA_Score"] = df["SA_Score_RDKit"].where(df["SA_Score_RDKit"].notna(), df[cols["sa_pred"]])

    # 3. Physics-informed redlines.
    mask = (
        (df[cols["density"]] >= MIN_DENSITY_PROXY)
        & (df[cols["hof"]] >= MIN_HOF)
        & (df[cols["gap"]] >= MIN_GAP)
        & (df[cols["rho"]] >= MIN_TRIGGER_RHO)
        & (df[cols["vsmax"]] <= MAX_VS_MAX)
        & (df["Pareto_SA_Score"] <= MAX_SA_SCORE)
        & (df[cols["mw"]] <= MAX_MW)
        & (df[cols["bde"]] >= MIN_BDE)
    )
    if MAX_SIGMA2 is not None:
        mask &= df[cols["sigma2"]] <= MAX_SIGMA2
    if MAX_NU is not None:
        mask &= df[cols["nu"]] <= MAX_NU

    # Required for 10D Pareto: all objective values must be finite.
    pareto_objective_cols = [
        cols["density"],
        cols["hof"],
        cols["gap"],
        cols["rho"],
        cols["bde"],
        "Pareto_SA_Score",
        cols["vsmax"],
        cols["sigma2"],
        cols["nu"],
        cols["mw"],
    ]
    mask &= df[pareto_objective_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)

    df_reject_redline = df[~mask].copy()
    if not df_reject_redline.empty:
        df_reject_redline["Reject_Reason"] = "failed_redline_or_missing_10D_objective"
        rejected.append(df_reject_redline)

    df_pool = df[mask].copy()
    print(f"\n[INFO] 突破结构与物理红线生还者池: {len(df_pool)} 个分子。")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if rejected:
        pd.concat(rejected, ignore_index=True).to_csv(rejected_csv, index=False)
        print(f"[INFO] 被拒绝候选已保存: {rejected_csv}")

    if df_pool.empty:
        pd.DataFrame(columns=df.columns).to_csv(output_csv, index=False)
        print("⚠️ 物理漏斗全军覆没。请放宽红线或提高生成多样性。")
        return

    # 4. Actual 10D Pareto ranking.
    max_cols = [cols["density"], cols["hof"], cols["gap"], cols["rho"], cols["bde"]]
    min_cols = ["Pareto_SA_Score", cols["vsmax"], cols["sigma2"], cols["nu"], cols["mw"]]

    df_pareto = get_pareto_front(df_pool, max_cols=max_cols, min_cols=min_cols)
    df_pareto = add_screening_score(df_pareto, max_cols=max_cols, min_cols=min_cols)
    df_final = df_pareto.sort_values(by="Screening_Score_10Target", ascending=False).reset_index(drop=True)

    # Add human-readable metadata columns.
    df_final["Pareto_Maximize"] = ";".join(max_cols)
    df_final["Pareto_Minimize"] = ";".join(min_cols)
    df_final["Density_Label_Note"] = "Density is a molecular-volume-derived proxy unless crystal density is supplied."

    df_final.to_csv(output_csv, index=False)

    print("\n👑 终极预备役 TOP 10 (10D Pareto Rank-0 + normalized screening score):")
    cols_to_show = [
        "Molecule",
        cols["density"],
        cols["hof"],
        cols["gap"],
        cols["rho"],
        cols["bde"],
        "Pareto_SA_Score",
        "Screening_Score_10Target",
    ]
    print(df_final.head(10)[cols_to_show].to_string(index=False))
    print(f"\n🏆 进入 ORCA 神谕候选队列: {len(df_final)} 个。已保存至 {output_csv}")


if __name__ == "__main__":
    main()
