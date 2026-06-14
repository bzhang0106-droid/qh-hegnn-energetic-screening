"""
08b_ultimate_merge.py

Manual HELS workflow database merge script.

Main corrections vs. the previous version:
1. Always backs up and writes the active clean training dataset after a successful merge; previously it wrote only when rows were dropped.
2. Requires the full 10-target feature set, including vertical BDE, for training consistency.
3. Preserves density-source metadata so molecular-volume-derived density is not confused with crystal density.
4. Uses step1 optimized geometry preferentially when extracting trigger-bond rho and archiving XYZ.

Important:
- Run this after 07_kamlet_jacobs_eval.py and 08a_run_multiwfn_critic2.py while temp_calc is still present.
"""

import glob
import math
import os
import re
import shutil
import sys
from typing import Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
from tqdm import tqdm

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except ImportError:
    print("[FATAL] 无法加载 RDKit 的 SA_Score 模块。请确保 RDKit 完整安装。")
    sys.exit(1)

TRUE_DATA_CSV = "../results/True_vs_Pred_Detonation.csv"
ORIGINAL_DATASET_CSV = "../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
XYZ_ARCHIVE_DIR = "../data/raw_2100_xyz"
TEMP_CALC_DIR = "../temp_calc"
ROLLING_BACKUP_DIR = "../data/backups"

TARGET_PROPS = [
    "Density_calc(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
]

BDE_OPTIONAL_PROPS = [
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


def latest_backup_path(src_path: str) -> str:
    src_name = os.path.basename(src_path)
    stem = src_name[:-4] if src_name.lower().endswith(".csv") else src_name
    return os.path.join(ROLLING_BACKUP_DIR, f"{stem}.latest_pre_merge.bak.csv")


def write_latest_backup(src_path: str, backup_path: Optional[str] = None) -> str:
    backup_path = backup_path or latest_backup_path(src_path)
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    tmp_path = f"{backup_path}.tmp"
    shutil.copy2(src_path, tmp_path)
    os.replace(tmp_path, backup_path)
    return backup_path


def extract_homo_lumo(mol_id: str) -> float:
    out_file = os.path.join(TEMP_CALC_DIR, mol_id, f"{mol_id}_step1_opt.out")
    if not os.path.exists(out_file):
        return float("nan")
    homo, lumo = None, None
    try:
        with open(out_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        # Prefer the last ORBITAL ENERGIES block.
        start_indices = [i for i, line in enumerate(lines) if line.strip() == "ORBITAL ENERGIES"]
        for start in reversed(start_indices):
            homo, lumo = None, None
            for line in lines[start + 4 : start + 800]:
                if line.strip() == "" or "----" in line:
                    if homo is not None:
                        break
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        occ, energy_ev = float(parts[1]), float(parts[3])
                    except ValueError:
                        continue
                    if occ > 0.001:
                        homo = energy_ev
                    elif occ <= 0.001 and lumo is None:
                        lumo = energy_ev
                        break
            if homo is not None and lumo is not None:
                return round(lumo - homo, 4)
    except Exception:
        pass
    return float("nan")


def parse_esp(esp_path: str) -> Tuple[float, float, float]:
    vs_max, sigma2, nu = float("nan"), float("nan"), float("nan")
    if not os.path.exists(esp_path):
        return vs_max, sigma2, nu
    try:
        with open(esp_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        match_max = re.search(r"Maximal value:\s*([-+]?\d*\.\d+)\s*kcal/mol", content, re.IGNORECASE)
        if match_max:
            vs_max = float(match_max.group(1))
        else:
            backup_max = re.search(r"Global surface maximum:\s*([-+]?\d*\.\d+)\s*a\.u\.", content, re.IGNORECASE)
            if backup_max:
                vs_max = float(backup_max.group(1)) * 627.509

        match_sigma = re.search(r"(?:Overall variance|Variance of ESP)[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content, re.IGNORECASE)
        if match_sigma:
            sigma2 = float(match_sigma.group(1))

        match_nu = re.search(r"Balance of charges[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", content, re.IGNORECASE)
        if match_nu:
            nu = float(match_nu.group(1))
    except Exception:
        pass
    return vs_max, sigma2, nu


def get_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def parse_critic2(report_path: str, xyz_path: str) -> float:
    if not os.path.exists(report_path) or not os.path.exists(xyz_path):
        return float("nan")
    atoms = []
    try:
        with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f.readlines()[2:]:
                p = line.split()
                if len(p) >= 4:
                    atoms.append((p[0].capitalize(), float(p[1]), float(p[2]), float(p[3])))
    except Exception:
        return float("nan")

    min_rho = 999.0
    try:
        with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "(3,-1)" in line and "bond" in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            cp_coord = (float(parts[3]), float(parts[4]), float(parts[5]))
                            rho = float(parts[7])
                        except ValueError:
                            continue
                        distances = sorted(
                            [(get_distance(cp_coord, (ax, ay, az)), elem) for elem, ax, ay, az in atoms],
                            key=lambda x: x[0],
                        )
                        if len(distances) >= 2:
                            bond = "".join(sorted([distances[0][1], distances[1][1]]))
                            if bond in ["CN", "NN"] and 0.10 < rho < min_rho:
                                min_rho = rho
        if min_rho != 999.0:
            return round(min_rho, 6)
    except Exception:
        pass
    return float("nan")


def calculate_sa_score(smiles: str) -> float:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return float(sascorer.calculateScore(mol))
    except Exception:
        pass
    return float("nan")


def choose_xyz_for_mol(mol_dir: str, mol_id: str) -> Optional[str]:
    preferred = os.path.join(mol_dir, f"{mol_id}_step1_opt.xyz")
    if os.path.exists(preferred):
        return preferred
    xyz_files = sorted(glob.glob(os.path.join(mol_dir, "*.xyz")))
    if not xyz_files:
        return None
    # Prefer optimized-looking xyz over init xyz.
    for path in xyz_files:
        name = os.path.basename(path).lower()
        if "opt" in name and "init" not in name:
            return path
    return xyz_files[0]


def main() -> None:
    print("==================================================")
    print("🔄 启动 Active Learning 真理融合引擎 (9D, manual workflow)")
    print("==================================================")

    if not os.path.exists(ORIGINAL_DATASET_CSV):
        print(f"❌ [致命错误] 找不到主数据库 {ORIGINAL_DATASET_CSV}，拒绝合并。")
        sys.exit(1)
    if not os.path.exists(TRUE_DATA_CSV):
        print(f"❌ [致命错误] 找不到 K-J 结算结果 {TRUE_DATA_CSV}。请先运行 07_kamlet_jacobs_eval.py。")
        sys.exit(1)
    if not os.path.exists(TEMP_CALC_DIR):
        print(f"❌ [致命错误] 找不到 {TEMP_CALC_DIR}。请在归档 temp_calc 之前运行本脚本。")
        sys.exit(1)

    os.makedirs(XYZ_ARCHIVE_DIR, exist_ok=True)
    df_old = pd.read_csv(ORIGINAL_DATASET_CSV)

    if "SAscore" not in df_old.columns:
        df_old["SAscore"] = float("nan")
    if df_old["SAscore"].isna().any():
        print("⏳ 正在为老数据库批量补全缺失 SAscore...")
        tqdm.pandas(desc="SAscore 补全")
        df_old["SAscore"] = df_old.progress_apply(
            lambda row: calculate_sa_score(str(row["SMILES"])) if pd.isna(row.get("SAscore")) else row["SAscore"],
            axis=1,
        )

    df_true = pd.read_csv(TRUE_DATA_CSV)
    if "HOF_Src" not in df_true.columns:
        print("❌ K-J 结算结果缺失 HOF_Src 列。")
        return

    df_valid = df_true[df_true["HOF_Src"] == "DLPNO-MP2 真值"].copy()
    actual_calc_dirs = set(os.listdir(TEMP_CALC_DIR))
    df_valid = df_valid[df_valid["Molecule"].astype(str).isin(actual_calc_dirs)]
    print(f"📦 物理缓存池校验完毕，准备固化 {len(df_valid)} 个实际完成计算的分子。")

    append_data = []
    for _, row in df_valid.iterrows():
        mol_id, smi = str(row["Molecule"]), str(row["SMILES"])
        mol_dir = os.path.join(TEMP_CALC_DIR, mol_id)
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.ExactMolWt(mol) if mol is not None else float("nan")

        data_dict = {
            "Molecule": mol_id,
            "SMILES": smi,
            "Density_calc(g/cm3)": row["Oracle_Density"],
            "Density_Source": row.get("Oracle_Density_Type", row.get("Dens_Src", "unknown")),
            "Heat_of_Formation(kcal/mol)": row["Oracle_HOF(kcal/mol)"],
            "HOMO_LUMO_Gap(eV)": extract_homo_lumo(mol_id),
            "SAscore": calculate_sa_score(smi),
            "Molecular_Weight": mw,
        }
        for col in BDE_OPTIONAL_PROPS:
            if col in row.index:
                data_dict[col] = row.get(col)

        vs_max, sigma2, nu = parse_esp(os.path.join(mol_dir, "esp_output.txt"))
        data_dict.update({"VS_max": vs_max, "Sigma2_tot": sigma2, "Nu": nu})

        xyz_path = choose_xyz_for_mol(mol_dir, mol_id)
        data_dict["Trigger_Bond_Rho"] = parse_critic2(os.path.join(mol_dir, "critic2_cpreport.out"), xyz_path) if xyz_path else float("nan")

        if xyz_path:
            try:
                shutil.copy2(xyz_path, os.path.join(XYZ_ARCHIVE_DIR, f"{mol_id}.xyz"))
            except Exception as exc:
                print(f"⚠️ 无法复制 XYZ ({mol_id}): {exc}")

        append_data.append(data_dict)

    if not append_data:
        print("⚠️ 提取后有效数据为零，流程中止。")
        return

    df_append = pd.DataFrame(append_data)
    df_merged = pd.concat([df_old, df_append], ignore_index=True)

    before_dedup = len(df_merged)
    if "SMILES" not in df_merged.columns:
        print("❌ 合并表缺失 SMILES 列。")
        return
    df_merged.drop_duplicates(subset=["SMILES"], keep="last", inplace=True)

    # Full 10D training matrix must be complete.
    for col in TARGET_PROPS:
        if col not in df_merged.columns:
            df_merged[col] = float("nan")
        df_merged[col] = pd.to_numeric(df_merged[col], errors="coerce")

    before_dropna = len(df_merged)
    df_merged.dropna(subset=TARGET_PROPS + ["SMILES"], inplace=True)
    df_merged = df_merged[
        (df_merged["Heat_of_Formation(kcal/mol)"] > -2000)
        & (df_merged["Heat_of_Formation(kcal/mol)"] < 3000)
        & (df_merged["Density_calc(g/cm3)"] > 0.5)
        & (df_merged["Density_calc(g/cm3)"] < 3.5)
        & (df_merged["Vertical_BDE(kcal/mol)"] > 0.0)
        & (df_merged["Vertical_BDE(kcal/mol)"] < 250.0)
    ].copy()
    df_merged.reset_index(drop=True, inplace=True)

    final_len = len(df_merged)
    dropped_count = before_dedup - final_len

    backup_name = write_latest_backup(ORIGINAL_DATASET_CSV)
    df_merged.to_csv(ORIGINAL_DATASET_CSV, index=False)

    print(f"🛡️ 核心数据库滚动备份已更新: {backup_name}")
    print(f"[INFO] 合并前总行数: {before_dedup} | 去重/清洗后: {final_len} | 清理行数: {dropped_count}")
    print(f"[INFO] 因 9D 缺失被剔除的行数约: {before_dropna - len(df_merged)}")
    print(f"✅ 数据已写回: {ORIGINAL_DATASET_CSV}")
    print("👉 下一步：重新运行 03_egnn_painn_train.py 训练更新后的 9D Hybrid EGNN。")


if __name__ == "__main__":
    main()
