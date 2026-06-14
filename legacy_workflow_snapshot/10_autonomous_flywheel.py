import os
import time
import subprocess
import re
import pandas as pd
from datetime import datetime

MAX_ITERATIONS = 5
REPORT_FILE = "HELS_Flywheel_Report.md"
METRICS_FILE = "Flywheel_Metrics_Tracking.csv"

def write_metrics_csv(iteration, max_d, eval_output):
    maes = re.findall(r'MAE:\s*([0-9\.]+)', eval_output) if eval_output else []
    metrics = {
        'Iteration': iteration, 'Max_D(km/s)': round(max_d, 2),
        'Density_MAE': float(maes[0]) if len(maes) > 0 else None,
        'HOF_MAE': float(maes[1]) if len(maes) > 1 else None,
        'Gap_MAE': float(maes[2]) if len(maes) > 2 else None,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    df = pd.DataFrame([metrics])
    if not os.path.exists(METRICS_FILE): df.to_csv(METRICS_FILE, index=False)
    else: df.to_csv(METRICS_FILE, mode='a', header=False, index=False)

def write_to_log(iteration, max_d, best_mol, eval_output):
    print(f"📝 正在将第 {iteration} 轮战报写入日志和追踪表...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    clean_eval = "\n".join([line for line in (eval_output or "").split('\n') if "MAE:" in line or "R2:" in line or "真实MAE" in line])
    if not clean_eval.strip(): clean_eval = "暂无精度评估数据或运行报错。"

    report_content = (
        f"\n## 🚀 第 {iteration} 轮主动学习纪元 (时间: {now})\n\n"
        f"### 🏆 终极造物\n* **最强分子代号:** `{best_mol}`\n* **真实爆速 (DLPNO-MP2):** **{max_d} km/s**\n\n"
        f"### 🔬 物理引擎 (8D-EGNN) 精度跟踪\n```text\n{clean_eval.strip()}\n```\n---\n"
    )
    try:
        with open(REPORT_FILE, 'a', encoding='utf-8') as f: f.write(report_content)
        write_metrics_csv(iteration, max_d, eval_output)
    except Exception as e: print(f"⚠️ 日志写入失败: {e}")

def run_cmd(cmd, desc):
    print(f"\n[{desc}] 正在执行: {cmd}", flush=True)
    cmd = cmd.replace("python ", "python -u ") if cmd.startswith("python ") else cmd
    if subprocess.run(cmd, shell=True, executable='/bin/bash').returncode != 0:
        print(f"❌ [致命错误] {desc} 执行失败，飞轮紧急停机！")
        exit(1)

def run_eval_and_capture():
    print("\n[大脑精度测算] 正在执行: python -u evaluate_3d.py", flush=True)
    try:
        output = subprocess.check_output("python -u evaluate_3d.py", shell=True, text=True, stderr=subprocess.STDOUT)
        print(output, flush=True) 
        return output
    except subprocess.CalledProcessError as e:
        return e.output

def submit_and_wait_slurm(submit_script, job_name):
    print(f"\n[SLURM 调度] 提交任务: {submit_script}", flush=True)
    job_id_match = re.search(r"Submitted batch job (\d+)", subprocess.check_output(f"sbatch {submit_script}", shell=True, text=True))
    if not job_id_match:
        print("❌ [致命错误] sbatch 提交失败或未返回 Job ID！")
        exit(1)
    job_id = job_id_match.group(1)
    print(f"✅ 成功接管 Job ID: {job_id}。进入挂起轮询模式...", flush=True)
    
    while True:
        squeue_out = subprocess.check_output(f"squeue -j {job_id} 2>/dev/null || echo 'DONE'", shell=True, text=True)
        if "DONE" in squeue_out or job_id not in squeue_out:
            print(f"🎉 Job {job_id} ({job_name}) 神谕计算完毕！", flush=True)
            break
        print(f"⏳ [等待中] {job_name} 仍在激烈运算... (60秒后重试)", flush=True)
        time.sleep(60)

def extract_and_log(iteration, eval_output):
    csv_path = "../results/True_vs_Pred_Detonation.csv" 
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty: write_to_log(iteration, df.iloc[0]['Oracle_D(km/s)'], df.iloc[0]['Molecule'], eval_output)
        except Exception as e: print(f"⚠️ 解析最终战报失败: {e}")

def main():
    print("==================================================")
    print("🛸 HELS Workflow 7.0 (8D 全景自适应版): 全自动飞轮引擎")
    print("==================================================")
    
    if not os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, 'w', encoding='utf-8') as f: f.write("# 🌌 HELS 8D 高能分子创世飞轮战报\n\n")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n" + "▼"*50 + f"\n🔥 开启第 {iteration} 轮主动学习进化\n" + "▼"*50, flush=True)
        
        # 1. 大脑更新与推断
        run_cmd("python 03_egnn_painn_train.py", "训练 8D EGNN 全景物理大脑")
        eval_output = run_eval_and_capture()
        run_cmd("rm -f ../results/Final_5D_Top_Candidates.csv ../data/final_verification_results.csv ../results/Pareto_Optimal_Candidates.csv ../data/active_learning_targets.csv", "清空历史残留推断文件")
        
        # 安全归档
        if os.path.exists("../temp_calc") and len(os.listdir("../temp_calc")) > 0:
            bak_dir = f"../temp_calc_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_cmd(f"mv ../temp_calc {bak_dir} && mkdir ../temp_calc", f"归档上一轮神谕计算残骸至: {bak_dir}")
        else:
            run_cmd("mkdir -p ../temp_calc", "初始化神谕目录")

        run_cmd("python 04_ultimate_discovery.py", "64核 CPU 并发极速推断")
        
        # 2. 筛选与投递
        run_cmd("python 05_physics_purifier.py", "6D 帕累托终极筛选 (附带 SA 合成护盾)")
        run_cmd("python 05b_extract_candidates.py", "提取本轮神谕靶标") 
        target_csv = "../data/active_learning_targets.csv"
        if not os.path.exists(target_csv) or len(pd.read_csv(target_csv)) == 0:
            print("⚠️ 本轮无分子突破帕累托漏斗，直接跳过神谕计算与入库环节。")
            continue  # 直接进入下一轮播种
            
        num_mols = len(pd.read_csv(target_csv))
        chunk_size = 8
        import math
        num_arrays = math.ceil(num_mols / chunk_size)
        
        print(f"📊 [智能调度] 本轮共计 {num_mols} 个神谕靶标。判定策略：分配为 {num_arrays} 个阵列 (每节点上限 {chunk_size} 个)。")
        
        # 现场动态生成专属的 SLURM 提交脚本
        with open("submit_verify_dynamic.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=ORCA_Verify_Smart\n")
            f.write("#SBATCH --partition=long,plat,epyc\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --output=orca_array_%A_%a.log\n")
            f.write("#SBATCH --error=orca_array_%A_%a.err\n")
            
            if num_arrays > 1:
                f.write(f"#SBATCH --array=1-{num_arrays}\n")
                f.write("#SBATCH --ntasks=64\n")
            else:
                cores_needed = num_mols * 8
                workers = num_mols
                f.write(f"#SBATCH --ntasks={cores_needed}\n")
                
            f.write("\nsource /etc/profile\n")
            f.write("module purge\nmodule load openmpi-v4.1.6-gcc13.2.0\n")
            f.write("source /home/gma/bzhang/software/miniconda3/bin/activate energetic_gnn\n")
            f.write("cd /home/gma/bzhang/bzhang/Workflow2.0/scripts\n\n")
            
            if num_arrays > 1:
                f.write(f"python -u 02c_orca_verification.py --max_workers 8 --array_id $SLURM_ARRAY_TASK_ID --chunk_size {chunk_size}\n")
            else:
                f.write(f"python -u 02c_orca_verification.py --max_workers {workers}\n")
                
        # 投递刚刚写好的动态脚本
        submit_and_wait_slurm("submit_verify_dynamic.sh", f"ORCA 智能神谕阵列 ({num_arrays} 节点并发)")

        # 3. 结算与沉淀
        run_cmd("python 07_kamlet_jacobs_eval.py", "K-J 爆轰结算")
        run_cmd("python 08a_run_multiwfn_critic2.py", "Multiwfn & Critic2 宏观感度提取")
        extract_and_log(iteration, eval_output)
        csv_path = "../results/True_vs_Pred_Detonation.csv"
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            run_cmd("python 08b_ultimate_merge.py", "核心物理数据纯净入库")
        else:
            print("⚠️ 警告：本轮 ORCA 神谕计算无有效物理产出，跳过核心数据入库。")
        
        # 4. 飞轮闭环与下一轮动态播种
        run_cmd("python check_leakage.py", "化学空间诊断与反馈")
        
        temp = 1.25
        explore_cmd = ""
        if os.path.exists("../results/collapse_ratio.txt"):
            try:
                with open("../results/collapse_ratio.txt", 'r') as f:
                    ratio = float(f.read().strip())
                if ratio > 50.0:
                    temp = 1.45
                    explore_cmd = "--explore"
                    print(f"\n🚨 [飞轮警报] 捕捉到严重空间坍塌 (同质化率: {ratio}%)！强制启动超大温度跃迁模式...")
            except: pass
        
        if iteration < MAX_ITERATIONS:
            print("\n🚀 [飞轮播种] 正在将大模型生成任务离线调度至 GPU 节点...")
            submit_and_wait_slurm(f"run_00.slurm {temp} {explore_cmd}", "大模型高能突变生成")
        
        print(f"\n🏆 第 {iteration} 轮完美闭环！10 秒后进入下一次进化...", flush=True)
        time.sleep(10)

if __name__ == "__main__":
    main()
