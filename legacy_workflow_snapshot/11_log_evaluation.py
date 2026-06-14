import os
import argparse
import re
import subprocess
import pandas as pd
from datetime import datetime
import shutil

REPORT_FILE = "HELS_Flywheel_Report.md"
METRICS_FILE = "Flywheel_Metrics_Tracking.csv"

def run_eval_and_capture():
    print("[精度测算] 正在执行 9D 模型验证...", flush=True)
    try:
        output = subprocess.check_output("python -u evaluate_3d.py", shell=True, text=True, stderr=subprocess.STDOUT)
        print(output, flush=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"❌ 验证执行失败:\n{e.output}")
        return e.output

def write_metrics_csv(iteration, max_d, eval_output):
    maes = re.findall(r'MAE:\s*([0-9\.]+)', eval_output) if eval_output else []
    metrics = {
        'Iteration': iteration, 
        'Max_D(km/s)': round(max_d, 2) if max_d else None,
        'Density_MAE': float(maes[0]) if len(maes) > 0 else None,
        'HOF_MAE': float(maes[1]) if len(maes) > 1 else None,
        'Gap_MAE': float(maes[2]) if len(maes) > 2 else None,
        'SA_MAE': float(maes[3]) if len(maes) > 3 else None,
        'VS_max_MAE': float(maes[4]) if len(maes) > 4 else None,
        'Sigma2_MAE': float(maes[5]) if len(maes) > 5 else None,
        'Nu_MAE': float(maes[6]) if len(maes) > 6 else None,
        'Rho_MAE': float(maes[7]) if len(maes) > 7 else None,
        'MW_MAE': float(maes[8]) if len(maes) > 8 else None,
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    df = pd.DataFrame([metrics])
    if not os.path.exists(METRICS_FILE): df.to_csv(METRICS_FILE, index=False)
    else: df.to_csv(METRICS_FILE, mode='a', header=False, index=False)

def archive_temp_calc():
    """归档神谕计算的临时残骸，保障下一轮的纯净"""
    temp_dir = "../temp_calc"
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bak_dir = f"../temp_calc_bak_{timestamp}"
        print(f"\n📦 [归档程序] 正在将本轮神谕计算临时文件夹归档至: {bak_dir}")
        try:
            shutil.move(temp_dir, bak_dir)
            os.makedirs(temp_dir, exist_ok=True)
            print("✨ 原始 temp_calc 目录已清空重置，飞轮已准备好迎接下一轮进化！")
        except Exception as e:
            print(f"⚠️ 归档失败，请手动检查权限或路径: {e}")
    else:
        print("\n📭 [归档程序] temp_calc 为空或不存在，无需执行归档操作。")
        os.makedirs(temp_dir, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, required=True, help="当前手动执行的轮数")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"📊 开始生成第 {args.iter} 轮手动纪元战报 (9D 架构)")
    print(f"==================================================")

    eval_output = run_eval_and_capture()
    clean_eval = "\n".join([line for line in (eval_output or "").split('\n') if "MAE:" in line or "R2:" in line])
    
    csv_path = "../results/True_vs_Pred_Detonation.csv"
    best_mol, max_d = "无神谕产出", None
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                df = df.sort_values(by='Oracle_D(km/s)', ascending=False)
                best_mol = df.iloc[0]['Molecule']
                max_d = df.iloc[0]['Oracle_D(km/s)']
        except Exception as e:
            print(f"⚠️ 解析神谕结算文件失败: {e}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = (
        f"\n## 🚀 第 {args.iter} 轮人工干预纪元 (时间: {now})\n\n"
        f"### 🏆 本轮终极造物\n* **最强分子代号:** `{best_mol}`\n* **真实爆速 (DLPNO-MP2):** **{max_d if max_d else 'N/A'} km/s**\n\n"
        f"### 🔬 物理引擎 (9D-EGNN) 精度跟踪\n```text\n{clean_eval.strip() if clean_eval else '无有效测算数据'}\n```\n---\n"
    )

    try:
        with open(REPORT_FILE, 'a', encoding='utf-8') as f:
            f.write(report_content)
        write_metrics_csv(args.iter, max_d, eval_output)
        print(f"✅ 战报已成功写入 {REPORT_FILE} 与 {METRICS_FILE}。")
    except Exception as e:
        print(f"❌ 日志写入致命错误: {e}")

    # 触发自动归档补丁
    archive_temp_calc()

if __name__ == "__main__":
    main()
