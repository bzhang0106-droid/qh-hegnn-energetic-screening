import os
import sys
import shutil
import glob
import pandas as pd
import re
import math
import datetime
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import RDConfig, Descriptors
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    print("[FATAL] 无法加载 RDKit 的 SA_Score 模块。请确保 RDKit 完整安装。")
    sys.exit(1)

TRUE_DATA_CSV = "../results/True_vs_Pred_Detonation.csv"
ORIGINAL_DATASET_CSV = "../data/old_dataset.csv"
XYZ_ARCHIVE_DIR = "../data/raw_2100_xyz"
TEMP_CALC_DIR = "../temp_calc"

def extract_homo_lumo(mol_id):
    out_file = os.path.join(TEMP_CALC_DIR, mol_id, f"{mol_id}_step1_opt.out")
    if not os.path.exists(out_file): return float('nan')
    homo, lumo = None, None
    try:
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "ORBITAL ENERGIES" in line:
                    for j in range(i+4, i+500):
                        if j >= len(lines) or "------------" in lines[j] and j > i+10: break
                        parts = lines[j].split()
                        if len(parts) >= 4 and parts[0].isdigit():
                            occ, energy_ev = float(parts[1]), float(parts[3])
                            if occ > 0.0: homo = energy_ev
                            elif occ == 0.0 and lumo is None:
                                lumo = energy_ev
                                break
    except: pass
    if homo is not None and lumo is not None: return round(lumo - homo, 4)
    return float('nan')

def parse_esp(esp_path):
    vs_max, sigma2, nu = float('nan'), float('nan'), float('nan')
    try:
        with open(esp_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
        match_max = re.search(r'Maximal value:\s*([-+]?\d*\.\d+)\s*kcal/mol', content, re.IGNORECASE)
        if not match_max:
            backup_max = re.search(r'Global surface maximum:\s*([-+]?\d*\.\d+)\s*a\.u\.', content, re.IGNORECASE)
            if backup_max: vs_max = float(backup_max.group(1)) * 627.509
        else:
            vs_max = float(match_max.group(1))

        match_sigma = re.search(r'(?:Overall variance|Variance of ESP)[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
        if match_sigma: sigma2 = float(match_sigma.group(1))

        match_nu = re.search(r'Balance of charges[^:]*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', content, re.IGNORECASE)
        if match_nu: nu = float(match_nu.group(1))
    except: pass
    return vs_max, sigma2, nu

def get_distance(p1, p2): return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def parse_critic2(report_path, xyz_path):
    atoms = []
    try:
        with open(xyz_path, 'r') as f:
            for line in f.readlines()[2:]:
                p = line.split()
                if len(p) >= 4: atoms.append((p[0].capitalize(), float(p[1]), float(p[2]), float(p[3])))
    except: return float('nan')
    min_rho = 999.0
    try:
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "(3,-1)" in line and "bond" in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            cp_coord, rho = (float(parts[3]), float(parts[4]), float(parts[5])), float(parts[7])
                        except ValueError: continue
                        distances = sorted([(get_distance(cp_coord, (ax, ay, az)), elem) for elem, ax, ay, az in atoms], key=lambda x: x[0])
                        if len(distances) >= 2:
                            bond = "".join(sorted([distances[0][1], distances[1][1]]))
                            if bond in ["CN", "NN"] and 0.10 < rho < min_rho: min_rho = rho
        if min_rho != 999.0: return round(min_rho, 6)
    except: pass
    return float('nan')

def calculate_sa_score(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None: return sascorer.calculateScore(mol)
    except: pass
    return float('nan')

def main():
    print("==================================================")
    print("🔄 启动 Active Learning 大一统真理融合引擎 (9D 极速对齐版)")
    print("==================================================")

    if not os.path.exists(ORIGINAL_DATASET_CSV):
        print(f"❌ [致命错误] 找不到主本数据 {ORIGINAL_DATASET_CSV}，拒绝合并！")
        sys.exit(1)

    df_old = pd.read_csv(ORIGINAL_DATASET_CSV)

    if 'SAscore' not in df_old.columns or df_old['SAscore'].isna().any():
        print("⏳ 正在为老数据库批量计算缺失的 SAscore 特征...")
        tqdm.pandas(desc="SAscore 补全")
        df_old['SAscore'] = df_old.progress_apply(
            lambda row: calculate_sa_score(str(row['SMILES'])) if pd.isna(row.get('SAscore')) else row['SAscore'],
            axis=1
        )

    df_true = pd.read_csv(TRUE_DATA_CSV)
    df_valid = df_true[df_true['HOF_Src'] == 'DLPNO-MP2 真值'].copy()

    actual_calc_dirs = set(os.listdir(TEMP_CALC_DIR))
    df_valid = df_valid[df_valid['Molecule'].isin(actual_calc_dirs)]
    print(f"📦 物理缓存池校验完毕，正在提取并固化 {len(df_valid)} 个实际完成计算的分子...")

    append_data = []
    for _, row in df_valid.iterrows():
        mol_id, smi = row['Molecule'], str(row['SMILES'])
        mol_dir = os.path.join(TEMP_CALC_DIR, mol_id)
        
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.ExactMolWt(mol) if mol else float('nan')

        data_dict = {
            'Molecule': mol_id, 'SMILES': smi,
            'Density_calc(g/cm3)': row['Oracle_Density'],
            'Heat_of_Formation(kcal/mol)': row['Oracle_HOF(kcal/mol)'],
            'HOMO_LUMO_Gap(eV)': extract_homo_lumo(mol_id),
            'SAscore': calculate_sa_score(smi),
            'Molecular_Weight': mw
        }

        vs_max, sigma2, nu = parse_esp(os.path.join(mol_dir, "esp_output.txt"))
        data_dict.update({'VS_max': vs_max, 'Sigma2_tot': sigma2, 'Nu': nu})

        xyz_files = glob.glob(os.path.join(mol_dir, "*.xyz"))
        data_dict['Trigger_Bond_Rho'] = parse_critic2(os.path.join(mol_dir, "critic2_cpreport.out"), xyz_files[0]) if xyz_files else float('nan')

        if xyz_files:
            try: shutil.copy2(xyz_files[0], os.path.join(XYZ_ARCHIVE_DIR, f"{mol_id}.xyz"))
            except Exception as e: print(f"⚠️ 无法复制 XYZ: {e}")

        append_data.append(data_dict)

    if not append_data:
        print("⚠️ 提取后有效数据为零，流程中止。")
        return

    df_append = pd.DataFrame(append_data)
    df_merged = pd.concat([df_old, df_append], ignore_index=True)

    initial_len = len(df_merged)
    df_merged.drop_duplicates(subset=['SMILES'], keep='last', inplace=True)

    target_cols = ['Density_calc(g/cm3)', 'Heat_of_Formation(kcal/mol)', 'VS_max', 'Trigger_Bond_Rho', 'HOMO_LUMO_Gap(eV)', 'SAscore', 'Molecular_Weight']
    existing_cols = [col for col in target_cols if col in df_merged.columns]
    df_merged.dropna(subset=existing_cols, inplace=True)

    df_merged = df_merged[(df_merged['Heat_of_Formation(kcal/mol)'] > -2000) & (df_merged['Heat_of_Formation(kcal/mol)'] < 3000)]
    df_merged.reset_index(drop=True, inplace=True)

    final_len = len(df_merged)
    dropped_count = initial_len - final_len

    if dropped_count > 0:
        print(f"[🛡️ 自动排毒] 拦截并销毁 {dropped_count} 个劣质/重复分子(含原库去重清理)。")

        backup_name = f"{ORIGINAL_DATASET_CSV}.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        shutil.copy2(ORIGINAL_DATASET_CSV, backup_name)
        print(f"🛡️ 核心数据库快照已生成: {backup_name}")
        df_merged.to_csv(ORIGINAL_DATASET_CSV, index=False)
        print(f"✅ 完美融合！当前数据库扩容至: {final_len} 个高能分子。")

if __name__ == "__main__":
    main()
