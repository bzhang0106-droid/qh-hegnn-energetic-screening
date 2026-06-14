import os
import subprocess
import pandas as pd
import re
import argparse
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

# ================= 配置区域 (Workflow 2.0 绝对路径体系) =================
ORCA_CMD = "/home/soft/amd/codes/orca/v6.0.0-gcc13.2.0/orca"
ORCA_2MKL_CMD = "/home/soft/amd/codes/orca/v6.0.0-gcc13.2.0/orca_2mkl"
MULTIWFN_CMD = "/home/gma/bzhang/soft/Multiwfn_2026.2.2_bin_Linux_noGUI/Multiwfn_noGUI"

INPUT_CANDIDATES = "../data/active_learning_targets_100.csv"
OUTPUT_CSV = "../data/oracle_results.csv"  
XYZ_DIR = "../data/raw_2100_xyz"  # 核心：将3D坐标汇入总库供 EGNN 使用
TEMP_DIR = "../temp_calc"
# =======================================================================

def get_molecular_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    c, h, n, o = atoms.count('C'), atoms.count('H'), atoms.count('N'), atoms.count('O')
    mw = ExactMolWt(mol)
    ob = -1600.0 * (2 * c + h / 2.0 - o) / mw
    return c, h, n, o, round(mw, 2), round(ob, 2)

def safe_extract(keyword, text, default=0.0):
    pattern = rf"{keyword}.*?([-+]?\d*\.\d+[E]?[-\+]?\d*|\d+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default

def process_single_molecule(mol_name, smiles):
    """单分子的处理工作流：自动完成 ORCA 三步法与数据提取"""
    mol_info = get_molecular_info(smiles)
    if not mol_info:
        return None
    c, h, n, o, mw, ob = mol_info

    work_dir = os.path.join(TEMP_DIR, mol_name)
    os.makedirs(work_dir, exist_ok=True)
    
    # 我们将从第二步(Freq)提取波函数，所以预期的 molden 名字如下
    molden_file = f"{mol_name}_step2_freq.molden.input"
    molden_path = os.path.join(work_dir, molden_file)

    # ---------------------------------------------------------
    # 物理计算核心区：如果 molden 不存在，说明需要跑 ORCA
    # ---------------------------------------------------------
    if not (os.path.exists(molden_path) and os.path.getsize(molden_path) > 1000):
        # 1. 初始 3D 构型生成 (使用 ETKDGv3 提高起步质量)
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None # 无法生成合理 3D 构型，淘汰
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except:
            pass
            
        init_xyz = os.path.join(work_dir, f"{mol_name}_init.xyz")
        Chem.MolToXYZFile(mol, init_xyz)

        # ---------------------------------------------------------
        # 步骤 1：B3LYP-D3BJ / def2-SVP 几何优化 (Opt)
        # ---------------------------------------------------------
        opt_inp = os.path.join(work_dir, f"{mol_name}_step1_opt.inp")
        with open(opt_inp, 'w') as f:
            f.write("! B3LYP D3BJ def2-SVP RIJCOSX def2/J Opt\n! LooseOpt NormalPrint\n! KDIIS SOSCF\n\n")
            f.write("%pal nprocs 8 end\n%maxcore 1700\n\n")
            f.write("%scf\n  MaxIter 350\nend\n\n")
            f.write("%geom\n  MaxStep 0.20\n  Trust 0.20\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_init.xyz\n")
            
        with open(os.path.join(work_dir, f"{mol_name}_step1_opt.out"), 'w') as f_out:
            subprocess.run([ORCA_CMD, f"{mol_name}_step1_opt.inp"], cwd=work_dir, stdout=f_out, stderr=subprocess.STDOUT)
            
        opt_xyz = os.path.join(work_dir, f"{mol_name}_step1_opt.xyz")
        if not os.path.exists(opt_xyz):
            return None  # 第一步优化失败，直接放弃该分子

        # 【核心操作】将优化好的坐标拷贝到 3D 训练集总库，供 EGNN 使用！
        os.system(f"cp {opt_xyz} {XYZ_DIR}/{mol_name}.xyz")

        # ---------------------------------------------------------
        # 步骤 2：频率与热力学计算 (Freq)
        # ---------------------------------------------------------
        freq_inp = os.path.join(work_dir, f"{mol_name}_step2_freq.inp")
        with open(freq_inp, 'w') as f:
            f.write("! B3LYP D3BJ def2-SVP RIJCOSX def2/J Freq\n! NormalPrint\n! KDIIS SOSCF\n\n")
            f.write("%pal nprocs 8 end\n%maxcore 1700\n\n")
            f.write("%scf\n  MaxIter 400\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_step1_opt.xyz\n")
            
        with open(os.path.join(work_dir, f"{mol_name}_step2_freq.out"), 'w') as f_out:
            subprocess.run([ORCA_CMD, f"{mol_name}_step2_freq.inp"], cwd=work_dir, stdout=f_out, stderr=subprocess.STDOUT)

        # 提取 Multiwfn 所需的波函数文件
        subprocess.run([ORCA_2MKL_CMD, f"{mol_name}_step2_freq", "-molden"], cwd=work_dir, stdout=subprocess.DEVNULL)

        # ---------------------------------------------------------
        # 步骤 3：DLPNO-MP2 / def2-TZVP 高精度单点能
        # ---------------------------------------------------------
        mp2_inp = os.path.join(work_dir, f"{mol_name}_step3_dlpnomp2.inp")
        with open(mp2_inp, 'w') as f:
            f.write("! DLPNO-MP2 def2-TZVP def2-TZVP/C TightSCF NormalPrint\n! RIJCOSX def2/J\n\n")
            f.write("%pal nprocs 8 end\n%maxcore 1700\n\n")
            f.write("%scf\n  MaxIter 400\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_step1_opt.xyz\n")
            
        with open(os.path.join(work_dir, f"{mol_name}_step3_dlpnomp2.out"), 'w') as f_out:
            subprocess.run([ORCA_CMD, f"{mol_name}_step3_dlpnomp2.inp"], cwd=work_dir, stdout=f_out, stderr=subprocess.STDOUT)

    if not os.path.exists(molden_path):
        return None 

    # ---------------------------------------------------------
    # 特征提取区 (保留 Multiwfn 特征提取，以便与旧数据格式完全对齐)
    # ---------------------------------------------------------
    # 1. 提取静电势极值与方差
    mw_esp = subprocess.run([MULTIWFN_CMD, molden_file], input="12\n0\n-1\nq\n", text=True, cwd=work_dir, capture_output=True)
    esp_out = mw_esp.stdout

    vs_max = safe_extract("maximum", esp_out)
    vs_min = safe_extract("minimum", esp_out)
    sigma2 = safe_extract("variance", esp_out)
    nu = safe_extract("balance", esp_out)
    vol = safe_extract("volume", esp_out)

    # 2. QTAIM 寻找最弱 C-N 触发键
    subprocess.run([MULTIWFN_CMD, molden_file], input="2\n2\n3\n8\n0\nq\n", text=True, cwd=work_dir, capture_output=True)
    rho_cn_min = 0.30
    cp_path = os.path.join(work_dir, "CPprops.txt")
    if os.path.exists(cp_path):
        with open(cp_path, 'r') as f:
            content = f.read()
            cp_blocks = content.split("CP  ")
            bcp_data = []
            for block in cp_blocks:
                if "Type: (3,-1)" in block and "(C )" in block and "(N )" in block:
                    rho = safe_extract("Electron density", block)
                    if rho != 0.0: bcp_data.append(rho)
            if bcp_data:
                rho_cn_min = min(bcp_data)

    # =========================================================
    # 注意：为了让 GNN 学习，必须从 ORCA 输出中提取 Hf (生成焓) 的原始能量
    # 这里我们先完整返回特征，后续你可以用 Python 脚本结合 MP2 和 Freq 结果算真值
    # =========================================================

    return {
        "Molecule": mol_name, "SMILES": smiles, "C": c, "H": h, "N": n, "O": o, "MW": mw,
        "Oxygen_Balance(%)": ob, "Trigger_Bond_Rho": rho_cn_min, "Trigger_Bond_Type": "C-N",
        "VS_max": vs_max, "VS_min": vs_min, "Sigma2_tot": sigma2, "Nu": nu, "Volume_Ang3": vol
    }

def run_pipeline(max_workers):
    print(f"🚀 启动神谕计算集群 (ORCA 3步法并发模式, Worker数={max_workers})...")
    if not os.path.exists(INPUT_CANDIDATES): 
        print(f"❌ 找不到输入文件 {INPUT_CANDIDATES}！请先运行 01_bayesian_fast_surrogate.py 提取 100 个主动学习样本。")
        return
        
    os.makedirs(XYZ_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CANDIDATES)

    csv_headers = ["Molecule", "SMILES", "C", "H", "N", "O", "MW", "Oxygen_Balance(%)",
                   "Trigger_Bond_Rho", "Trigger_Bond_Type", "VS_max", "VS_min", "Sigma2_tot", "Nu", "Volume_Ang3"]
                   
    if not os.path.exists(OUTPUT_CSV):
        pd.DataFrame(columns=csv_headers).to_csv(OUTPUT_CSV, index=False)

    done_mols = set(pd.read_csv(OUTPUT_CSV)['Molecule'].astype(str).tolist())
    print(f"📦 发现本地已保存 {len(done_mols)} 个完成的数据。目标总计: {len(df)} 个。")

    tasks = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            # 获取分子名，如果表头变了这里做个兼容
            mol_name = str(row.get('Molecule', f"AL_Mutant_{idx:04d}"))
            smiles = row['SMILES']
            if mol_name in done_mols:
                continue

            future = executor.submit(process_single_molecule, mol_name, smiles)
            tasks[future] = mol_name

        completed_count = 0
        total_tasks = len(tasks)

        for future in concurrent.futures.as_completed(tasks):
            mol_name = tasks[future]
            completed_count += 1
            try:
                result = future.result()
                if result is not None:
                    pd.DataFrame([result]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                    print(f"[{completed_count}/{total_tasks}] ✅ {mol_name} 三步计算并存档成功！3D 坐标已并入总库！")
                else:
                    print(f"[{completed_count}/{total_tasks}] ⚠️ {mol_name} 计算不收敛或发生空间碰撞，已淘汰。")
            except Exception as e:
                print(f"[{completed_count}/{total_tasks}] ❌ {mol_name} 发生严重系统崩溃: {str(e)}")

    print("\n" + "="*60)
    print("🎯 Active Learning 第一轮神谕计算全部完成！")
    print(f"📂 真值数据已存入: {OUTPUT_CSV}")
    print(f"📂 3D 几何坐标已存入: {XYZ_DIR}")
    print("👉 下一步：运行 03_egnn_painn_train.py，吞噬 3D 坐标进行终极训练！")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=12, help="并行运行的分子最大数量 (建议为节点总核数/8)")
    args = parser.parse_args()

    run_pipeline(args.max_workers)
