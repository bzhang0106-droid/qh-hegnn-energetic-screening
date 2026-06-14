"""
07_kamlet_jacobs_eval.py

Manual HELS workflow Kamlet-Jacobs post-processing.

Main corrections vs. the previous version:
1. Does not falsely call molecular-volume-derived density a crystal-density truth.
2. If an explicit crystal-density column exists in the ORCA result table, it is used preferentially.
3. Otherwise, density is estimated from Multiwfn molecular volume and clearly marked as an estimate.
4. Keeps the ORCA validation logic unchanged; this script only post-processes completed ORCA outputs.
"""

import argparse
import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

ORACLE_CSV = "../data/final_verification_results.csv"
if not os.path.exists(ORACLE_CSV) and os.path.exists("../data/oracle_results.csv"):
    ORACLE_CSV = "../data/oracle_results.csv"

PREDICT_CSV = "../results/Pareto_Optimal_Candidates.csv"
OUTPUT_CSV = "../results/True_vs_Pred_Detonation.csv"
TEMP_CALC_DIR = "../temp_calc"
DENSITY_CALIBRATION_CSV = "../data/density_calibration/density_calibration_reference_8benchmarks_raw.csv"

EH_TO_KCAL = 627.509
BOHR3_TO_ANG3 = 0.148184711
AMU_PER_A3_TO_G_CM3 = 1.66053906660
C_CORRECTION = 171.3
REF = {
    "C": (-37.758548969687) + 0.00000000 + 0.00094421,
    "H2": (-1.160285873559) + 0.01006440 + 0.00094421,
    "N2": (-109.360948813260) + 0.00558518 + 0.00094421,
    "O2": (-150.114262063708) + 0.00372684 + 0.00094421,
}

CRYSTAL_DENSITY_COLUMNS = [
    "Crystal_Density(g/cm3)",
    "Experimental_Density(g/cm3)",
    "Density_crystal(g/cm3)",
    "rho_crystal",
]

BDE_ORACLE_COLUMNS = [
    "BDE_Job_Dir",
    "BDE_Bond_Type",
    "BDE_Bond_i_1based",
    "BDE_Bond_j_1based",
    "BDE_Bond_WBO",
    "Vertical_BDE_Eh",
    "Vertical_BDE(kcal/mol)",
    "Vertical_BDE_eV",
    "BDE_Parse_Status",
]


def is_reasonable_density(value: float) -> bool:
    return np.isfinite(value) and 0.8 <= value <= 3.2



def load_density_calibration(calibration_csv: str = DENSITY_CALIBRATION_CSV) -> Optional[Tuple[float, float, int]]:
    """
    Optional crystal-density calibration.

    Expected CSV columns can be any of the following pairs:
    - Density_proxy and Crystal_Density(g/cm3)
    - Density_calc(g/cm3) and Crystal_Density(g/cm3)
    - Density_proxy and Experimental_Density(g/cm3)

    The fitted equation is: rho_crystal_cal = a * rho_proxy + b.
    If the file is absent, the workflow keeps using the molecular-volume-derived
    density proxy and marks the density source explicitly.
    """
    if not os.path.exists(calibration_csv):
        fallback = "../data/density_calibration_reference.csv"
        if os.path.exists(fallback):
            calibration_csv = fallback
        else:
            return None
    try:
        df = pd.read_csv(calibration_csv)
    except Exception as exc:
        print(f"[WARN] 无法读取密度校准文件 {calibration_csv}: {exc}")
        return None

    x_candidates = ["Density_proxy", "Density_calc(g/cm3)", "Molecular_Volume_Density(g/cm3)"]
    y_candidates = ["Crystal_Density(g/cm3)", "Experimental_Density(g/cm3)", "rho_crystal"]
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if x_col is None or y_col is None:
        print(f"[WARN] 密度校准文件缺少列。需要 proxy 列之一 {x_candidates} 与 crystal 列之一 {y_candidates}")
        return None

    d = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d[x_col] > 0.8) & (d[x_col] < 3.2) & (d[y_col] > 0.8) & (d[y_col] < 3.2)]
    if len(d) < 3:
        print(f"[WARN] 有效密度校准样本不足: {len(d)}。至少需要 3 个。")
        return None

    a, b = np.polyfit(d[x_col].values, d[y_col].values, deg=1)
    print(f"[INFO] 启用密度校准: rho_crystal = {a:.4f} * rho_proxy + {b:.4f}  (n={len(d)})")
    return float(a), float(b), int(len(d))


def apply_density_calibration(proxy_density: float, calibration: Optional[Tuple[float, float, int]]) -> Optional[float]:
    if calibration is None or not is_reasonable_density(proxy_density):
        return None
    a, b, _ = calibration
    rho = a * float(proxy_density) + b
    return float(rho) if is_reasonable_density(rho) else None

def extract_orca_h298(molecule_id: str) -> Optional[float]:
    freq_out = os.path.join(TEMP_CALC_DIR, molecule_id, f"{molecule_id}_step2_freq.out")
    sp_out = os.path.join(TEMP_CALC_DIR, molecule_id, f"{molecule_id}_step3_dlpnomp2.out")
    h_corr, e_elec = None, None

    if os.path.exists(freq_out):
        with open(freq_out, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Thermal Enthalpy correction" in line:
                    match = re.search(r"([-+]?\d*\.\d+)\s*Eh", line)
                    if match:
                        h_corr = float(match.group(1))

    if os.path.exists(sp_out):
        with open(sp_out, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_low = line.lower()
                if "dlpno-mp2 total energy" in line_low:
                    match = re.search(r"([-+]?\d*\.\d+)", line)
                    if match:
                        e_elec = float(match.group(1))
            if e_elec is None:
                f.seek(0)
                for line in f:
                    if "FINAL SINGLE POINT ENERGY" in line:
                        try:
                            e_elec = float(line.strip().split()[-1])
                        except Exception:
                            pass

    if h_corr is not None and e_elec is not None:
        return e_elec + h_corr
    return None


def calc_kamlet_jacobs(c: int, h: int, n: int, o: int, mw: float, hof_kcal_mol: float, density: float) -> Tuple[float, float]:
    if mw <= 0 or density <= 0:
        return 0.0, 0.0

    o_avail = float(o)
    h2o_moles = min(h / 2.0, o_avail)
    o_avail -= h2o_moles
    h2_moles = (h / 2.0) - h2o_moles

    co_moles = min(float(c), o_avail)
    o_avail -= co_moles
    co2_moles = min(co_moles, o_avail)
    o_avail -= co2_moles
    co_moles -= co2_moles

    o2_moles = max(o_avail, 0.0) / 2.0
    n2_moles = n / 2.0
    total_gas_moles = h2o_moles + co_moles + co2_moles + o2_moles + n2_moles + h2_moles
    if total_gas_moles <= 0:
        return 0.0, 0.0

    gas_mass = (
        h2o_moles * 18.015
        + co_moles * 28.01
        + co2_moles * 44.01
        + o2_moles * 31.998
        + n2_moles * 28.013
        + h2_moles * 2.016
    )
    N = total_gas_moles / mw
    M = gas_mass / total_gas_moles
    hof_products = (h2o_moles * -57.8) + (co_moles * -26.4) + (co2_moles * -94.0)
    q_heat = (hof_kcal_mol - hof_products) / mw * 1000.0
    if q_heat <= 0:
        return 0.0, 0.0

    D = 1.01 * (N * (M ** 0.5) * (q_heat ** 0.5)) ** 0.5 * (1.0 + 1.30 * density)
    P = 1.558 * (density ** 2) * N * (M ** 0.5) * (q_heat ** 0.5)
    return round(D, 2), round(P, 1)


def count_elements_from_smiles(smiles: str) -> Tuple[int, int, int, int, float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"SMILES 解析失败: {smiles}")
    mol_h = Chem.AddHs(mol)
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    mw = sum({"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}[k] * v for k, v in counts.items())
    return counts["C"], counts["H"], counts["N"], counts["O"], mw


def infer_volume_and_density(row: pd.Series, mw: float) -> Tuple[Optional[float], Optional[float], str]:
    """
    Prefer explicit crystal density if present. Otherwise estimate from Multiwfn volume.
    Returns density, volume_ang3, source_label.
    """
    for col in CRYSTAL_DENSITY_COLUMNS:
        if col in row.index:
            try:
                rho = float(row[col])
                if is_reasonable_density(rho):
                    return rho, None, f"explicit_crystal_density:{col}"
            except Exception:
                pass

    raw_volume = None
    for col in ["Volume_Ang3", "Volume", "Molecular_Volume_Ang3"]:
        if col in row.index:
            try:
                raw_volume = float(row[col])
                break
            except Exception:
                pass

    if raw_volume is None or not np.isfinite(raw_volume) or raw_volume <= 0:
        return None, None, "missing_volume"

    # Candidate 1: raw value already in Angstrom^3.
    vol_a = raw_volume
    rho_a = mw * AMU_PER_A3_TO_G_CM3 / vol_a if vol_a > 0 else np.nan

    # Candidate 2: raw value in Bohr^3; convert to Angstrom^3.
    vol_b = raw_volume * BOHR3_TO_ANG3
    rho_b = mw * AMU_PER_A3_TO_G_CM3 / vol_b if vol_b > 0 else np.nan

    candidates = []
    if is_reasonable_density(rho_a):
        candidates.append((rho_a, vol_a, "Multiwfn_molecular_volume_estimate_Ang3"))
    if is_reasonable_density(rho_b):
        candidates.append((rho_b, vol_b, "Multiwfn_molecular_volume_estimate_Bohr3_converted"))

    if candidates:
        # Pick the one closest to typical energetic molecular density ~1.8 g/cm3.
        return min(candidates, key=lambda x: abs(x[0] - 1.8))

    # Fall back to the old heuristic but mark it as low confidence.
    if raw_volume > 500:
        vol = raw_volume * BOHR3_TO_ANG3
        rho = mw * AMU_PER_A3_TO_G_CM3 / vol
        return rho, vol, "low_confidence_volume_Bohr3_converted"
    rho = mw * AMU_PER_A3_TO_G_CM3 / raw_volume
    return rho, raw_volume, "low_confidence_volume_Ang3_assumed"


def resolve_pred_col(df: pd.DataFrame, base: str) -> Optional[str]:
    for c in [f"Pred_{base}", base]:
        if c in df.columns:
            return c
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Kamlet-Jacobs post-processing with explicit density-source tracking.")
    parser.add_argument(
        "--density_mode",
        choices=["auto", "proxy", "calibrated"],
        default="auto",
        help="auto: explicit crystal density > calibrated proxy > proxy; proxy: use molecular-volume proxy; calibrated: require calibration for proxy densities.",
    )
    parser.add_argument("--density_calibration_csv", default=DENSITY_CALIBRATION_CSV)
    parser.add_argument("--oracle_csv", default=ORACLE_CSV, help="ORCA oracle/verification CSV to post-process.")
    parser.add_argument("--predict_csv", default=PREDICT_CSV, help="Candidate prediction CSV used for SMILES-level alignment.")
    parser.add_argument("--output_csv", default=OUTPUT_CSV, help="Output CSV for K-J truth-vs-prediction results.")
    args = parser.parse_args()

    calibration = load_density_calibration(args.density_calibration_csv)
    oracle_csv = args.oracle_csv
    predict_csv = args.predict_csv
    output_csv = args.output_csv

    if not os.path.exists(oracle_csv) or not os.path.exists(predict_csv):
        print(f"[错误] 找不到神谕真值数据({ORACLE_CSV})或预测帕累托文件({PREDICT_CSV})。")
        return

    df_oracle = pd.read_csv(oracle_csv)
    df_pred = pd.read_csv(predict_csv)
    if df_oracle.empty or df_pred.empty:
        print("[错误] 输入表为空，无法结算。")
        return

    dens_col = resolve_pred_col(df_pred, "Density_calc(g/cm3)")
    hof_col = resolve_pred_col(df_pred, "Heat_of_Formation(kcal/mol)")
    keep_cols = ["SMILES"]
    rename_map = {}
    if dens_col:
        keep_cols.append(dens_col)
        rename_map[dens_col] = "Pred_Density_calc(g/cm3)"
    if hof_col:
        keep_cols.append(hof_col)
        rename_map[hof_col] = "Pred_Heat_of_Formation(kcal/mol)"

    df_pred_sub = df_pred[keep_cols].copy().rename(columns=rename_map)
    df_merged = pd.merge(df_oracle, df_pred_sub, on="SMILES", how="inner")
    print(f"[INFO] 成功对齐 {len(df_merged)} 个神谕分子。")

    results, dropped_count = [], 0
    for _, row in df_merged.iterrows():
        mol_id = str(row["Molecule"])
        smi = str(row["SMILES"])

        try:
            c, h, n, o, mw_formula = count_elements_from_smiles(smi)
        except Exception:
            dropped_count += 1
            continue
        mw = float(row.get("MW", mw_formula)) if pd.notna(row.get("MW", np.nan)) else mw_formula

        h298_eh = extract_orca_h298(mol_id)
        if h298_eh is None:
            dropped_count += 1
            continue

        elem_sum = c * REF["C"] + (h / 2.0) * REF["H2"] + (n / 2.0) * REF["N2"] + (o / 2.0) * REF["O2"]
        true_hof = (h298_eh - elem_sum) * EH_TO_KCAL + c * C_CORRECTION

        density_proxy, volume_ang3, density_source_raw = infer_volume_and_density(row, mw)
        if density_proxy is None or not np.isfinite(density_proxy) or density_proxy <= 0:
            dropped_count += 1
            continue

        # Choose the density used for the headline K-J metrics.
        calibrated_density = None
        is_explicit_crystal = str(density_source_raw).startswith("explicit_crystal_density")
        if not is_explicit_crystal:
            calibrated_density = apply_density_calibration(float(density_proxy), calibration)

        if args.density_mode == "proxy":
            density_used = float(density_proxy)
            density_type = density_source_raw
        elif args.density_mode == "calibrated":
            if is_explicit_crystal:
                density_used = float(density_proxy)
                density_type = density_source_raw
            elif calibrated_density is not None:
                density_used = calibrated_density
                density_type = f"calibrated_density_from_proxy:{density_source_raw}"
            else:
                dropped_count += 1
                continue
        else:  # auto
            if is_explicit_crystal:
                density_used = float(density_proxy)
                density_type = density_source_raw
            elif calibrated_density is not None:
                density_used = calibrated_density
                density_type = f"calibrated_density_from_proxy:{density_source_raw}"
            else:
                density_used = float(density_proxy)
                density_type = density_source_raw

        pred_hof = float(row.get("Pred_Heat_of_Formation(kcal/mol)", np.nan))
        D_proxy, P_proxy = calc_kamlet_jacobs(c, h, n, o, mw, true_hof, float(density_proxy))
        D_true, P_true = calc_kamlet_jacobs(c, h, n, o, mw, true_hof, float(density_used))

        rec = {
            "Molecule": mol_id,
            "SMILES": smi,
            "Oracle_Density": round(float(density_used), 3),
            "Oracle_Density_Type": density_type,
            "Density_Proxy(g/cm3)": round(float(density_proxy), 3),
            "Density_Calibrated(g/cm3)": round(float(calibrated_density), 3) if calibrated_density is not None else np.nan,
            "Density_Mode": args.density_mode,
            "Oracle_Molecular_Volume_Ang3": round(float(volume_ang3), 3) if volume_ang3 is not None else np.nan,
            "Dens_Src": density_type,
            "Oracle_HOF(kcal/mol)": round(true_hof, 1),
            "Pred_HOF(kcal/mol)": round(pred_hof, 1) if np.isfinite(pred_hof) else np.nan,
            "HOF_Src": "DLPNO-MP2 真值",
            "HOF_Method_Detail": "DLPNO-MP2 single point + B3LYP-D3BJ thermal enthalpy correction",
            "D_proxy(km/s)": D_proxy,
            "P_proxy(GPa)": P_proxy,
            "Oracle_D(km/s)": D_true,
            "Oracle_P(GPa)": P_true,
        }
        for col in BDE_ORACLE_COLUMNS:
            if col in row.index:
                rec[col] = row.get(col, np.nan)
        results.append(rec)

    print(f"\n[汇总] 共清洗掉 {dropped_count} 个残缺或不可结算分子。")
    if not results:
        print("⚠️ 没有有效 K-J 结算结果。")
        return

    df_final = pd.DataFrame(results).sort_values(by="Oracle_D(km/s)", ascending=False)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)

    print("🏆 终极爆轰性能排行榜:")
    for i, (_, row) in enumerate(df_final.head(5).iterrows()):
        print(
            f"  No.{i+1} [{row['Molecule']}] | D: {row['Oracle_D(km/s)']} km/s | "
            f"P: {row['Oracle_P(GPa)']} GPa | density source: {row['Oracle_Density_Type']}"
        )
    print(f"📂 结算结果已保存: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
