import os
import subprocess
import pandas as pd
import numpy as np
import re
import argparse
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt

# ================= 配置区域 =================
ORCA_CMD = "/home/soft/amd/codes/orca/v6.0.0-gcc13.2.0/orca"
ORCA_2MKL_CMD = "/home/soft/amd/codes/orca/v6.0.0-gcc13.2.0/orca_2mkl"
MULTIWFN_CMD = "/home/gma/bzhang/soft/Multiwfn_2026.2.2_bin_Linux_noGUI/Multiwfn_noGUI"

INPUT_CANDIDATES = "../data/active_learning_targets.csv"
OUTPUT_CSV = "../data/final_verification_results.csv"
XYZ_DIR = "../data/raw_2100_xyz"
TEMP_DIR = "../temp_calc"

def get_molecular_info(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise RuntimeError("SMILES 解析失败")
    mol = Chem.AddHs(mol)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    c, h, n, o = atoms.count('C'), atoms.count('H'), atoms.count('N'), atoms.count('O')
    mw = ExactMolWt(mol)
    ob = -1600.0 * (2 * c + h / 2.0 - o) / mw
    return c, h, n, o, round(mw, 2), round(ob, 2)

def safe_extract(keyword, text, default=0.0):
    match = re.search(rf"{keyword}.*?([-+]?\d*\.\d+[E]?[-\+]?\d*|\d+)", text, re.IGNORECASE)
    return float(match.group(1)) if match else default

def check_orca_normal_termination(out_file):
    if not os.path.exists(out_file): return False
    try:
        with open(out_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f.readlines()[-50:]:
                if "ORCA TERMINATED NORMALLY" in line: return True
    except: pass
    return False

def process_single_molecule(mol_name, smiles):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    orca_env = os.environ.copy()
    if 'LD_LIBRARY_PATH' in orca_env:
        clean_paths = [p for p in orca_env['LD_LIBRARY_PATH'].split(':') if 'miniconda3' not in p]
        orca_env['LD_LIBRARY_PATH'] = ':'.join(clean_paths)

    c, h, n, o, mw, ob = get_molecular_info(smiles)

    work_dir = os.path.join(TEMP_DIR, mol_name)
    os.makedirs(work_dir, exist_ok=True)

    molden_file = f"{mol_name}_step2_freq.molden.input"
    molden_path = os.path.join(work_dir, molden_file)
    opt_xyz = os.path.join(work_dir, f"{mol_name}_step1_opt.xyz")
    mp2_out = os.path.join(work_dir, f"{mol_name}_step3_dlpnomp2.out")

    if not (os.path.exists(molden_path) and os.path.getsize(molden_path) > 1000 and check_orca_normal_termination(mp2_out)):

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))

        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.useBasicKnowledge = False
        params.maxIterations = 1000
        params.randomSeed = 42

        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            raise RuntimeError("RDKit 无法生成 3D 拓扑(极度拥挤或物理不合理)")

        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=300)
            elif AllChem.UFFHasAllMoleculeParams(mol):
                AllChem.UFFOptimizeMolecule(mol, maxIters=300)
        except: pass 

        init_xyz = os.path.join(work_dir, f"{mol_name}_init.xyz")
        Chem.MolToXYZFile(mol, init_xyz)

        opt_inp = os.path.join(work_dir, f"{mol_name}_step1_opt.inp")
        opt_out = os.path.join(work_dir, f"{mol_name}_step1_opt.out")
        with open(opt_inp, 'w') as f:
            f.write("! B3LYP D3BJ def2-SVP RIJCOSX def2/J Opt\n! LooseOpt NormalPrint\n! KDIIS SOSCF\n\n%pal nprocs 8 end\n%maxcore 1700\n\n%scf\n  MaxIter 350\nend\n\n%geom\n  MaxStep 0.20\n  Trust 0.20\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_init.xyz\n")

        if not check_orca_normal_termination(opt_out):
            subprocess.run([ORCA_CMD, f"{mol_name}_step1_opt.inp"], cwd=work_dir, stdout=open(opt_out, 'w'), stderr=subprocess.STDOUT, env=orca_env)
            if not check_orca_normal_termination(opt_out):
                raise RuntimeError(f"ORCA 第一步 (几何优化) 崩溃！请查看日志: {opt_out}")

        freq_inp = os.path.join(work_dir, f"{mol_name}_step2_freq.inp")
        freq_out = os.path.join(work_dir, f"{mol_name}_step2_freq.out")
        with open(freq_inp, 'w') as f:
            f.write("! B3LYP D3BJ def2-SVP RIJCOSX def2/J Freq\n! NormalPrint\n! KDIIS SOSCF\n\n%pal nprocs 8 end\n%maxcore 1700\n\n%scf\n  MaxIter 400\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_step1_opt.xyz\n")

        if not check_orca_normal_termination(freq_out):
            subprocess.run([ORCA_CMD, f"{mol_name}_step2_freq.inp"], cwd=work_dir, stdout=open(freq_out, 'w'), stderr=subprocess.STDOUT, env=orca_env)
            if not check_orca_normal_termination(freq_out):
                raise RuntimeError(f"ORCA 第二步 (频率) 崩溃！请查看日志: {freq_out}")

        subprocess.run([ORCA_2MKL_CMD, f"{mol_name}_step2_freq", "-molden"], cwd=work_dir, stdout=subprocess.DEVNULL, env=orca_env)

        mp2_inp = os.path.join(work_dir, f"{mol_name}_step3_dlpnomp2.inp")
        with open(mp2_inp, 'w') as f:
            f.write("! DLPNO-MP2 def2-TZVP def2-TZVP/C TightSCF NormalPrint\n! RIJCOSX def2/J\n\n%pal nprocs 8 end\n%maxcore 1700\n\n%scf\n  MaxIter 400\nend\n\n")
            f.write(f"* xyzfile 0 1 {mol_name}_step1_opt.xyz\n")

        if not check_orca_normal_termination(mp2_out):
            subprocess.run([ORCA_CMD, f"{mol_name}_step3_dlpnomp2.inp"], cwd=work_dir, stdout=open(mp2_out, 'w'), stderr=subprocess.STDOUT, env=orca_env)
            if not check_orca_normal_termination(mp2_out):
                raise RuntimeError(f"ORCA 第三步 (MP2) 崩溃！请查看日志: {mp2_out}")

        os.system(f"cp {opt_xyz} {XYZ_DIR}/{mol_name}.xyz")

    try:
        mw_esp = subprocess.run([MULTIWFN_CMD, molden_file], input="12\n0\n-1\nq\n", text=True, cwd=work_dir, capture_output=True, timeout=300)
        vs_max = safe_extract("maximum", mw_esp.stdout)
        vs_min = safe_extract("minimum", mw_esp.stdout)
        sigma2 = safe_extract("variance", mw_esp.stdout)
        nu = safe_extract("balance", mw_esp.stdout)
        vol = safe_extract("volume", mw_esp.stdout)

        subprocess.run([MULTIWFN_CMD, molden_file], input="2\n2\n3\n8\n0\nq\n", text=True, cwd=work_dir, capture_output=True, timeout=300)
        rho_cn_min = 0.30
        cp_path = os.path.join(work_dir, "CPprops.txt")
        if os.path.exists(cp_path):
            with open(cp_path, 'r') as f:
                bcp_data = [safe_extract("Electron density", b) for b in f.read().split("CP  ") if "Type: (3,-1)" in b and "(C )" in b and "(N )" in b]
                bcp_data = [r for r in bcp_data if r != 0.0]
                if bcp_data: rho_cn_min = min(bcp_data)
    except Exception as e:
        raise RuntimeError(f"Multiwfn 执行失败: {str(e)}")

    return {
        "Molecule": mol_name, "SMILES": smiles, "C": c, "H": h, "N": n, "O": o, "MW": mw,
        "Oxygen_Balance(%)": ob, "Trigger_Bond_Rho": rho_cn_min, "Trigger_Bond_Type": "C-N",
        "VS_max": vs_max, "VS_min": vs_min, "Sigma2_tot": sigma2, "Nu": nu, "Volume_Ang3": vol
    }

def run_pipeline(max_workers, array_id, chunk_size):
    print(f"🚀 启动神谕计算集群 (X光透视+环境隔离版, 本节点最大并发={max_workers})...")
    if not os.path.exists(INPUT_CANDIDATES):
        print(f"❌ 找不到输入文件 {INPUT_CANDIDATES}！")
        return

    os.makedirs(XYZ_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    df_all = pd.read_csv(INPUT_CANDIDATES)
    csv_headers = ["Molecule", "SMILES", "C", "H", "N", "O", "MW", "Oxygen_Balance(%)",
                   "Trigger_Bond_Rho", "Trigger_Bond_Type", "VS_max", "VS_min", "Sigma2_tot", "Nu", "Volume_Ang3"]

    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        pd.DataFrame(columns=csv_headers).to_csv(OUTPUT_CSV, index=False)

    # 🌟 统一且唯一的切片逻辑
    if array_id is not None:
        start_idx = (array_id - 1) * chunk_size
        end_idx = start_idx + chunk_size
        df = df_all.iloc[start_idx:end_idx]
        print(f"📦 [阵列节点 {array_id}] 分配任务切片: 索引 {start_idx} 到 {end_idx-1}，共计 {len(df)} 个分子。")
        if len(df) == 0:
            print("✅ 本节点无分配任务，直接退出。")
            return
    else:
        df = df_all

    done_mols = set(pd.read_csv(OUTPUT_CSV)['Molecule'].astype(str).tolist())

    tasks = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in df.iterrows():
            mol_name = str(row.get('Molecule', f"AL_Mutant_{idx:04d}"))
            if mol_name in done_mols: continue
            tasks[executor.submit(process_single_molecule, mol_name, row['SMILES'])] = mol_name

        completed_count, total_tasks = 0, len(tasks)

        for future in concurrent.futures.as_completed(tasks):
            mol_name = tasks[future]
            completed_count += 1
            try:
                result = future.result()
                pd.DataFrame([result]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                print(f"[{completed_count}/{total_tasks}] ✅ {mol_name} 神谕结算存档！")
            except Exception as e:
                print(f"[{completed_count}/{total_tasks}] ❌ {mol_name} 计算中止: {str(e)}")

    print("\n==================================================")
    print("🎯 本节点的神谕计算份额全部交接完毕！")

# 🌟 唯一入口点，接管所有参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=8, help="并行运行的最大数量")
    parser.add_argument("--array_id", type=int, default=None, help="SLURM 阵列 ID")
    parser.add_argument("--chunk_size", type=int, default=10, help="每个节点切片的分子数量")
    args = parser.parse_args()
    
    run_pipeline(args.max_workers, args.array_id, args.chunk_size)
