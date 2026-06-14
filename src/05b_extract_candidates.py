#!/usr/bin/env python3
import os
import re
import json
import argparse
import datetime
import pandas as pd

PARETO_CSV = "../results/Pareto_Optimal_Candidates.csv"
OUTPUT_CSV = "../data/active_learning_targets.csv"
STATE_JSON = "../data/workflow_state.json"
TEMP_CALC_DIR = "../temp_calc"


def infer_next_iter_from_temp_calc():
    """
    Infer next AL iteration from existing temp_calc directories:
    AL04_Target_0001 -> next_iter at least 5.
    """
    max_iter = 0
    if os.path.isdir(TEMP_CALC_DIR):
        pat = re.compile(r"^AL(\d+)_Target_\d+$")
        for name in os.listdir(TEMP_CALC_DIR):
            m = pat.match(name)
            if m:
                max_iter = max(max_iter, int(m.group(1)))
    return max_iter + 1 if max_iter > 0 else 1


def read_next_iter():
    """
    Priority:
    1. workflow_state.json next_iter
    2. infer from ../temp_calc/ALxx_Target_yyyy
    """
    if os.path.exists(STATE_JSON):
        try:
            with open(STATE_JSON, "r", encoding="utf-8") as f:
                state = json.load(f)
            if "next_iter" in state:
                return int(state["next_iter"])
        except Exception:
            pass
    return infer_next_iter_from_temp_calc()


def prefix_exists(prefix):
    """
    Avoid conflict if ALxx_Target_* already exists in temp_calc.
    """
    if not os.path.isdir(TEMP_CALC_DIR):
        return False
    pat = re.compile(rf"^{re.escape(prefix)}_Target_\d+$")
    return any(pat.match(x) for x in os.listdir(TEMP_CALC_DIR))


def choose_safe_prefix(iter_id, user_prefix=None):
    if user_prefix:
        prefix = user_prefix
        if prefix_exists(prefix):
            raise RuntimeError(
                f"Prefix {prefix} already exists in {TEMP_CALC_DIR}. "
                f"Use a new --prefix or --iter."
            )
        return prefix, iter_id

    cur = int(iter_id)
    while True:
        prefix = f"AL{cur:02d}"
        if not prefix_exists(prefix):
            return prefix, cur
        cur += 1


def main():
    parser = argparse.ArgumentParser(
        description="Extract top candidates for ORCA verification with iteration-safe IDs."
    )
    parser.add_argument("--top_k", type=int, default=500)
    parser.add_argument("--input", default=PARETO_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--iter", type=int, default=None,
                        help="Manual AL iteration number, e.g. --iter 4 -> AL04_Target_0001.")
    parser.add_argument("--prefix", default=None,
                        help="Manual prefix, e.g. --prefix AL04. Overrides automatic prefix.")
    args = parser.parse_args()

    print("=" * 50)
    print(f"🎯 启动 ORCA 神谕靶标提取程序 (Top {args.top_k})")
    print("=" * 50)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input Pareto file not found: {args.input}")

    df = pd.read_csv(args.input)
    if len(df) == 0:
        raise RuntimeError("Pareto candidate table is empty.")

    if "SMILES" not in df.columns:
        raise RuntimeError("Input Pareto table must contain a SMILES column.")

    # Sort by current screening score if available.
    if "Screening_Score_10Target" in df.columns:
        df = df.sort_values("Screening_Score_10Target", ascending=False)
    elif "Screening_Score_10D" in df.columns:
        df = df.sort_values("Screening_Score_10D", ascending=False)
    elif "Screening_Score_9Target" in df.columns:
        df = df.sort_values("Screening_Score_9Target", ascending=False)
    elif "Screening_Score_9D" in df.columns:
        df = df.sort_values("Screening_Score_9D", ascending=False)
    elif "Pareto_Rank" in df.columns:
        df = df.sort_values("Pareto_Rank", ascending=True)

    selected = df.head(args.top_k).copy().reset_index(drop=True)

    # Determine safe AL prefix.
    iter_id = args.iter if args.iter is not None else read_next_iter()
    prefix, used_iter = choose_safe_prefix(iter_id, args.prefix)

    # Preserve original molecule ID.
    if "Molecule" in selected.columns:
        selected.insert(1, "Source_Molecule", selected["Molecule"])
    else:
        selected.insert(0, "Source_Molecule", [f"Candidate_{i:04d}" for i in range(1, len(selected) + 1)])

    selected["Molecule"] = [f"{prefix}_Target_{i:04d}" for i in range(1, len(selected) + 1)]

    keep_cols = ["Molecule", "Source_Molecule", "SMILES"]
    for col in [
        "Screening_Score_10Target",
        "Screening_Score_10D",
        "Screening_Score_9Target",
        "Pareto_SA_Score",
        "Pred_Vertical_BDE(kcal/mol)",
        "Vertical_BDE(kcal/mol)",
        "Pareto_Rank",
        "Density_Label_Note",
    ]:
        if col in selected.columns and col not in keep_cols:
            keep_cols.append(col)

    out = selected[keep_cols].copy()

    # Always include note about density proxy.
    if "Density_Label_Note" not in out.columns:
        out["Density_Label_Note"] = (
            "Density is a molecular-volume-derived proxy unless crystal density is supplied."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": args.input,
        "output": args.output,
        "top_k": int(args.top_k),
        "iteration": int(used_iter),
        "prefix": prefix,
        "n_targets": int(len(out)),
        "id_rule": f"{prefix}_Target_0001 ... {prefix}_Target_{len(out):04d}",
    }
    meta_path = os.path.splitext(args.output)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"🎉 已提取 {len(out)} 个 ORCA 靶标。")
    print(f"🧬 本轮唯一 ID 前缀: {prefix}")
    print(f"📂 输出文件: {args.output}")
    print(f"🧾 元数据文件: {meta_path}")
    print("👉 下一步：按当前手动流程提交 ORCA 验证脚本。")
    print()
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
