"""
11_log_evaluation.py

Manual HELS workflow logging script.

Changes in this version:
1. Calls 09_eda_analysis.py --mode eval and --mode eda instead of deprecated evaluate_3d.py.
2. Does not archive ../temp_calc by default. Use --archive_temp_calc only after
   07_kamlet_jacobs_eval.py and 08b_ultimate_merge.py have completed.
3. Keeps the log terminology conservative: density-derived detonation metrics are
   labelled according to the density source reported by 07.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd

REPORT_FILE = "HELS_Clean_Sanitized_Workflow_Report.md"
METRICS_FILE = "Clean_Sanitized_Workflow_Metrics_Tracking.csv"
TRUE_VS_PRED_CSV = "../results/True_vs_Pred_Detonation.csv"
PRODUCTION_METRICS_CSV = "../results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv"
FINAL_MODEL_GROUP = "Final-Specialist-Hybrid-v2"


def run_eval_and_capture() -> str:
    print("[evaluation] Running 09_eda_analysis.py --mode eval ...", flush=True)
    cmd = [sys.executable, "-u", "09_eda_analysis.py", "--mode", "eval"]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        print(output, flush=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] evaluation failed:\n{e.output}")
        return e.output or ""



def run_eda_and_snapshot(iteration: int) -> str:
    print("[eda] Running 09_eda_analysis.py --mode eda ...", flush=True)
    cmd = [sys.executable, "-u", "09_eda_analysis.py", "--mode", "eda"]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        print(output, flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[WARN] EDA plotting failed:\n{e.output}")
        return e.output or ""

    source_to_label = {
        "01_feature_distributions_10D.png": "01_feature_distributions_10D",
        "02_density_hof_map.png": "02_density_hof_map",
        "03_correlation_matrix_10D.png": "03_correlation_matrix_10D",
    }
    out_dir = "../results/eda_plots"
    tag = f"iter{int(iteration):02d}"
    for src_name, label in source_to_label.items():
        src = os.path.join(out_dir, src_name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, f"{tag}_{label}.png")
        shutil.copy2(src, dst)
        print(f"[eda] Snapshot saved: {dst}")
    return output

def _metric_key(target: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(target)).strip("_")


def read_final_metrics() -> pd.DataFrame:
    if not os.path.exists(PRODUCTION_METRICS_CSV):
        return pd.DataFrame()
    df = pd.read_csv(PRODUCTION_METRICS_CSV)
    if "Model_Group" in df.columns:
        final = df[df["Model_Group"].eq(FINAL_MODEL_GROUP)].copy()
        if not final.empty:
            return final
    return df.copy()


def write_metrics_csv(iteration: int, max_d, eval_output: str) -> None:
    metrics = {
        "Iteration": iteration,
        "Max_D(km/s)": round(float(max_d), 2) if max_d is not None else None,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Architecture": "10D_BDE_Final_Specialist",
        "Metrics_Source": PRODUCTION_METRICS_CSV,
    }

    final_metrics = read_final_metrics()
    if not final_metrics.empty:
        for _, row in final_metrics.iterrows():
            key = _metric_key(row.get("Target", "Unknown"))
            for metric_name in ["MAE", "RMSE", "R2"]:
                if metric_name in row.index:
                    metrics[f"{key}_{metric_name}"] = row.get(metric_name)
    else:
        maes = re.findall(r"MAE:\s*([0-9\.eE+-]+)", eval_output or "")
        for i, value in enumerate(maes):
            metrics[f"Legacy_Target_{i+1}_MAE"] = float(value)

    new_row = pd.DataFrame([metrics])
    if os.path.exists(METRICS_FILE) and os.path.getsize(METRICS_FILE) > 0:
        old = pd.read_csv(METRICS_FILE)
        pd.concat([old, new_row], ignore_index=True, sort=False).to_csv(METRICS_FILE, index=False)
    else:
        new_row.to_csv(METRICS_FILE, index=False)

def archive_temp_calc() -> None:
    temp_dir = "../temp_calc"
    if os.path.exists(temp_dir) and len(os.listdir(temp_dir)) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak_dir = f"../temp_calc_bak_{timestamp}"
        print(f"\n📦 [归档程序] 正在将本轮神谕计算临时文件夹归档至: {bak_dir}")
        try:
            shutil.move(temp_dir, bak_dir)
            os.makedirs(temp_dir, exist_ok=True)
            print("✨ temp_calc 已归档并重建。")
        except Exception as e:
            print(f"⚠️ 归档失败，请手动检查权限或路径: {e}")
    else:
        print("\n📭 [归档程序] temp_calc 为空或不存在，无需归档。")
        os.makedirs(temp_dir, exist_ok=True)


def get_best_oracle() -> tuple[str, object, str]:
    best_mol, max_d, density_type = "无神谕产出", None, "N/A"
    if os.path.exists(TRUE_VS_PRED_CSV) and os.path.getsize(TRUE_VS_PRED_CSV) > 0:
        try:
            df = pd.read_csv(TRUE_VS_PRED_CSV)
            if not df.empty:
                d_col = "Oracle_D(km/s)"
                if d_col in df.columns:
                    df = df.sort_values(by=d_col, ascending=False)
                    best_mol = str(df.iloc[0].get("Molecule", "Unknown"))
                    max_d = df.iloc[0].get(d_col)
                    density_type = str(df.iloc[0].get("Oracle_Density_Type", "density_source_not_recorded"))
        except Exception as e:
            print(f"⚠️ 解析神谕结算文件失败: {e}")
    return best_mol, max_d, density_type



def write_workflow_state(iteration: int) -> None:
    """
    Record the completed manual AL iteration and the next iteration ID.
    05b_extract_candidates.py should read this file to generate unique
    ORCA target IDs such as AL04_Target_0001.
    """
    import json

    state_path = "../data/workflow_state.json"
    os.makedirs(os.path.dirname(state_path), exist_ok=True)

    state = {
        "last_completed_iter": int(iteration),
        "next_iter": int(iteration) + 1,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "id_rule": "AL{iter:02d}_Target_{index:04d}",
        "note": (
            "This state file is written by 11_log_evaluation.py after a full AL iteration. "
            "05b_extract_candidates.py should read next_iter to generate "
            "unique ORCA target IDs and avoid temp_calc directory conflicts."
        ),
        "architecture": "10D_BDE_Final_Specialist",
    }

    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"🧭 Workflow state updated: {state_path}")
    print(f"   last_completed_iter = {iteration}")
    print(f"   next_iter = {int(iteration) + 1}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, required=True, help="当前手动执行的轮数")
    parser.add_argument("--archive_temp_calc", action="store_true", help="显式归档 ../temp_calc。仅在 07 和 08b 完成后使用。")
    args = parser.parse_args()

    print("==================================================")
    print(f"📊 开始生成第 {args.iter} 轮手动纪元战报 (10D+BDE 架构)")
    print("==================================================")

    eval_output = run_eval_and_capture()
    run_eda_and_snapshot(args.iter)
    clean_eval = "\n".join([line for line in (eval_output or "").split("\n") if "MAE:" in line or "R2:" in line])
    best_mol, max_d, density_type = get_best_oracle()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = (
        f"\n## 🚀 第 {args.iter} 轮人工干预纪元 (时间: {now})\n\n"
        f"### 🏆 本轮终极造物\n"
        f"* **最强分子代号:** `{best_mol}`\n"
        f"* **最高爆速估计:** **{max_d if max_d is not None else 'N/A'} km/s**\n"
        f"* **密度来源:** `{density_type}`\n\n"
        f"### 🔬 物理引擎 (10D+BDE final-specialist) 精度跟踪\n"
        f"```text\n{clean_eval.strip() if clean_eval else '无有效测算数据'}\n```\n---\n"
    )

    try:
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(report_content)
        write_metrics_csv(args.iter, max_d, eval_output)
        print(f"✅ 战报已成功写入 {REPORT_FILE} 与 {METRICS_FILE}。")
        write_workflow_state(args.iter)
    except Exception as e:
        print(f"❌ 日志写入错误: {e}")

    if args.archive_temp_calc:
        archive_temp_calc()
    else:
        print("\n[INFO] 未归档 temp_calc。若确认 07 与 08b 已完成，可单独运行：")
        print(f"       /home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python -u 11_log_evaluation.py --iter {args.iter} --archive_temp_calc")


if __name__ == "__main__":
    main()
