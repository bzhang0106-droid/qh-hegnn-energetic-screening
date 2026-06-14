import os
import subprocess
import concurrent.futures
import glob

# ==========================================
# ⚙️ 路径与命令配置 (沿用你的绝对路径)
# ==========================================
WORK_DIR = "../temp_calc"
MULTIWFN_CMD = "/home/gma/bzhang/soft/Multiwfn_2026.2.2_bin_Linux_noGUI/Multiwfn_noGUI"
CRITIC2_CMD = "/home/gma/bzhang/critic2/build/src/critic2"
MAX_WORKERS = 8  # 考虑到是后处理，可以开 8 个并发

def process_molecule(mol_dir_path):
    mol_name = os.path.basename(mol_dir_path)
    
    # 自动寻找 molden 文件
    molden_files = glob.glob(os.path.join(mol_dir_path, "*.molden.input"))
    if not molden_files:
        return f"❌ {mol_name}: 找不到 molden.input 文件，跳过。"
    molden_file = molden_files[0]
    molden_filename = os.path.basename(molden_file)

    msg = f"[{mol_name}] "

    # --------------------------------------------------
    # 1. 运行 Multiwfn (ESP 计算)
    # --------------------------------------------------
    esp_out = os.path.join(mol_dir_path, "esp_output.txt")
    if not os.path.exists(esp_out):
        inp = "12\n0\n\n\n"
        bash_command = f"ulimit -s unlimited && export OMP_STACKSIZE=2G && {MULTIWFN_CMD} {molden_filename}"
        try:
            res_m = subprocess.run(bash_command, input=inp, cwd=mol_dir_path, text=True, 
                                   capture_output=True, timeout=1800, shell=True, executable='/bin/bash')
            with open(esp_out, "w", encoding="utf-8") as f:
                f.write(res_m.stdout)
                if res_m.stderr: f.write("\n\n=== STDERR ===\n" + res_m.stderr)
            msg += "✅ ESP完成 "
        except Exception as e:
            msg += f"❌ ESP失败({e}) "
    else:
        msg += "⚡ ESP已存在 "

    # --------------------------------------------------
    # 2. 运行 Critic2 (QTAIM 计算)
    # --------------------------------------------------
    critic_out = os.path.join(mol_dir_path, "critic2_cpreport.out")
    if not os.path.exists(critic_out):
        inp_content = f"molecule {molden_filename}\nauto\ncpreport\n"
        inp_file = os.path.join(mol_dir_path, "critic2_fast.inp")
        try:
            with open(inp_file, "w") as f: f.write(inp_content)
            res_c = subprocess.run([CRITIC2_CMD, "critic2_fast.inp"], cwd=mol_dir_path, 
                                   timeout=120, capture_output=True, text=True)
            with open(critic_out, "w", encoding="utf-8") as f:
                f.write(res_c.stdout or "")
            msg += "✅ QTAIM完成"
        except Exception as e:
            msg += f"❌ QTAIM失败({e})"
    else:
        msg += "⚡ QTAIM已存在"

    return msg

def main():
    print("==================================================")
    print("🚀 启动 5D 物理特征补全行动 (Multiwfn + Critic2)")
    print("==================================================")
    
    # 获取 temp_calc 下的所有分子目录
    dirs = [os.path.join(WORK_DIR, d) for d in os.listdir(WORK_DIR) if os.path.isdir(os.path.join(WORK_DIR, d))]
    
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in executor.map(process_molecule, dirs):
            completed += 1
            print(f"[{completed}/{len(dirs)}] {result}")
            
    print("\n🎉 物理特征补充完毕！准备进行终极数据融合！")

if __name__ == "__main__":
    main()
