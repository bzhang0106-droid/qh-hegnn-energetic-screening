import os
import re
import pandas as pd
from rdkit import Chem

# 兼容不同的神谕结果命名
ORACLE_CSV = "../data/final_verification_results.csv" 
if not os.path.exists(ORACLE_CSV) and os.path.exists("../data/oracle_results.csv"):
    ORACLE_CSV = "../data/oracle_results.csv"

PREDICT_CSV = "../results/Pareto_Optimal_Candidates.csv"
OUTPUT_CSV = "../results/True_vs_Pred_Detonation.csv"
TEMP_CALC_DIR = "../temp_calc"

EH_TO_KCAL = 627.509
C_CORRECTION = 171.3
REF = {
    "C":  (-37.758548969687) + 0.00000000 + 0.00094421,
    "H2": (-1.160285873559)  + 0.01006440 + 0.00094421,
    "N2": (-109.360948813260)+ 0.00558518 + 0.00094421,
    "O2": (-150.114262063708)+ 0.00372684 + 0.00094421,
}

def extract_orca_h298(molecule_id):
    freq_out = os.path.join(TEMP_CALC_DIR, molecule_id, f"{molecule_id}_step2_freq.out")
    sp_out = os.path.join(TEMP_CALC_DIR, molecule_id, f"{molecule_id}_step3_dlpnomp2.out")
    h_corr, e_elec = None, None
    if os.path.exists(freq_out):
        with open(freq_out, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "Thermal Enthalpy correction" in line:
                    match = re.search(r'([-+]?\d*\.\d+)\s*Eh', line)
                    if match: h_corr = float(match.group(1))
    if os.path.exists(sp_out):
        with open(sp_out, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "DLPNO-MP2 TOTAL ENERGY" in line or "DLPNO-MP2 total energy" in line.lower():
                    match = re.search(r'([-+]?\d*\.\d+)', line)
                    if match: e_elec = float(match.group(1))
            if e_elec is None:
                f.seek(0)
                for line in f:
                    if "FINAL SINGLE POINT ENERGY" in line:
                        e_elec = float(line.strip().split()[-1])
    if h_corr is not None and e_elec is not None: return e_elec + h_corr
    return None

def calc_kamlet_jacobs(c, h, n, o, mw, hof_kcal_mol, density):
    if mw == 0: return 0.0, 0.0
    o_avail = o
    h2o_moles = min(h / 2, o_avail)
    o_avail -= h2o_moles
    h2_moles = (h / 2) - h2o_moles
    co_moles = min(c, o_avail)
    o_avail -= co_moles
    co2_moles = min(co_moles, o_avail)
    o_avail -= co2_moles
    co_moles -= co2_moles
    o2_moles = o_avail / 2
    n2_moles = n / 2
    total_gas_moles = h2o_moles + co_moles + co2_moles + o2_moles + n2_moles + h2_moles
    if total_gas_moles == 0: return 0.0, 0.0
    gas_mass = (h2o_moles * 18.015 + co_moles * 28.01 + co2_moles * 44.01 + 
                o2_moles * 31.998 + n2_moles * 28.013 + h2_moles * 2.016)
    N = total_gas_moles / mw
    M = gas_mass / total_gas_moles
    hof_products = (h2o_moles * -57.8) + (co_moles * -26.4) + (co2_moles * -94.0)
    q_heat = (hof_kcal_mol - hof_products) / mw * 1000
    if q_heat <= 0: return 0.0, 0.0
    D = 1.01 * (N * (M ** 0.5) * (q_heat ** 0.5)) ** 0.5 * (1 + 1.30 * density)
    P = 1.558 * (density ** 2) * N * (M ** 0.5) * (q_heat ** 0.5)
    return round(D, 2), round(P, 1)

def main():
    print("==================================================")
    print("💥 启动 Kamlet-Jacobs 爆轰真值结算器 (纯正物理参数版)")
    print("==================================================")
    
    if not os.path.exists(ORACLE_CSV) or not os.path.exists(PREDICT_CSV):
        print(f"[错误] 找不到神谕真值数据({ORACLE_CSV})或预测帕累托文件({PREDICT_CSV})！")
        return

    df_oracle = pd.read_csv(ORACLE_CSV)
    df_pred = pd.read_csv(PREDICT_CSV)
    
    dens_col = [c for c in df_pred.columns if 'Density' in c][0]
    hof_col  = [c for c in df_pred.columns if 'Heat_of_Formation' in c][0]
    
    # ⚠️ 彻底移除了 HELS_Score 的提取，防止 Key Error 报错
    df_pred_sub = df_pred[['SMILES', dens_col, hof_col]].copy()
    
    df_pred_sub.rename(columns={
        dens_col: 'Pred_Density_calc(g/cm3)',
        hof_col: 'Pred_Heat_of_Formation(kcal/mol)'
    }, inplace=True)
    
    df_merged = pd.merge(df_oracle, df_pred_sub, on='SMILES', how='inner')
    print(f"[INFO] 成功对齐 {len(df_merged)} 个神谕分子。")

    results, dropped_count = [], 0
    for idx, row in df_merged.iterrows():
        mol_id = row['Molecule']
        smi = str(row['SMILES'])
        
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        c, h, n, o = 0, 0, 0, 0
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym == 'C': c += 1
            elif sym == 'H': h += 1
            elif sym == 'N': n += 1
            elif sym == 'O': o += 1
        mw = float(row.get('MW', c * 12.011 + h * 1.008 + n * 14.007 + o * 15.999))
        
        h298_eh = extract_orca_h298(mol_id)
        if h298_eh is not None:
            elem_sum = c * REF["C"] + (h / 2.0) * REF["H2"] + (n / 2.0) * REF["N2"] + (o / 2.0) * REF["O2"]
            true_hof = (h298_eh - elem_sum) * EH_TO_KCAL + c * C_CORRECTION
            hof_source = "DLPNO-MP2 真值"
        else:
            dropped_count += 1
            continue

        vol_true = float(row.get('Volume_Ang3', 0.0))
        if vol_true > 500: vol_true = vol_true * 0.148184 
            
        if vol_true > 10.0:
            density_true = (mw * 1.660539) / vol_true
            dens_source = "Multiwfn 真值"
        else:
            dropped_count += 1
            continue

        pred_hof = float(row['Pred_Heat_of_Formation(kcal/mol)'])
        D_true, P_true = calc_kamlet_jacobs(c, h, n, o, mw, true_hof, density_true)
        
        results.append({
            'Molecule': mol_id, 'SMILES': smi,
            'Oracle_Density': round(density_true, 3), 'Dens_Src': dens_source,
            'Oracle_HOF(kcal/mol)': round(true_hof, 1), 'Pred_HOF(kcal/mol)': round(pred_hof, 1),
            'HOF_Src': hof_source, 'Oracle_D(km/s)': D_true, 'Oracle_P(GPa)': P_true
        })

    print(f"\n[汇总] 共清洗掉 {dropped_count} 个残缺分子。")
    if not results: return

    df_final = pd.DataFrame(results).sort_values(by='Oracle_D(km/s)', ascending=False)
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print("🏆 终极爆轰性能排行榜 (严格物理验证):")
    for i, (_, row) in enumerate(df_final.head(5).iterrows()):
        print(f"  No.{i+1} [{row['Molecule']}] | D: {row['Oracle_D(km/s)']} km/s | P: {row['Oracle_P(GPa)']} GPa")

if __name__ == "__main__":
    main()
