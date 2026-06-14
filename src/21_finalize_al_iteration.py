#!/usr/bin/env python3
"""Finalize one active-learning oracle round into canonical workflow files.

Policy:
- Tagged files are audit backups.
- Canonical files are updated after validation so downstream scripts stay simple.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)


def run(cmd: list[str], cwd: Path) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(cwd))


def copy_with_parent(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[COPY] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}", flush=True)


def csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    return max(0, sum(1 for _ in path.open(errors="replace")) - 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="AL prefix, e.g. AL06")
    ap.add_argument("--tag", required=True, help="Tag, e.g. AL06_20260604")
    ap.add_argument("--min_collected", type=int, default=1, help="Abort if collected oracle rows are below this number")
    ap.add_argument("--allow_missing_parts", action="store_true", help="Allow target molecules missing from all part CSVs")
    ap.add_argument("--skip_merge", action="store_true", help="Only collect and run 07; do not merge into official DB")
    args = ap.parse_args()

    scripts = ROOT / "scripts"
    data = ROOT / "data"
    results = ROOT / "results"

    tagged_oracle = data / f"final_verification_results_{args.tag}.csv"
    canonical_oracle = data / "final_verification_results.csv"
    predict_csv = results / f"Pareto_Optimal_Candidates_{args.tag}.csv"
    canonical_kj = results / "True_vs_Pred_Detonation.csv"
    tagged_kj = results / f"True_vs_Pred_Detonation_{args.tag}.csv"

    if not predict_csv.exists():
        raise FileNotFoundError(predict_csv)

    run([
        str(PY), "19_collect_al_orca_parts.py",
        "--prefix", args.prefix,
        "--tag", args.tag,
        "--output", str(tagged_oracle),
        "--sync_default",
    ], cwd=scripts)

    summary_path = tagged_oracle.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text())
    print("[SUMMARY]", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if summary.get("rows_collected", 0) < args.min_collected:
        raise RuntimeError(f"Collected only {summary.get('rows_collected', 0)} rows; expected at least {args.min_collected}")
    if summary.get("missing_rows", 0) and not args.allow_missing_parts:
        raise RuntimeError(
            f"{summary['missing_rows']} target molecules are missing from ORCA part outputs. "
            "Rerun missing slices or pass --allow_missing_parts for a partial merge."
        )

    run([
        str(PY), "07_kamlet_jacobs_eval.py",
        "--oracle_csv", str(canonical_oracle),
        "--predict_csv", str(predict_csv),
        "--output_csv", str(canonical_kj),
    ], cwd=scripts)

    if not canonical_kj.exists() or csv_rows(canonical_kj) == 0:
        raise RuntimeError(f"K-J canonical output is empty or missing: {canonical_kj}")
    copy_with_parent(canonical_kj, tagged_kj)

    if not args.skip_merge:
        before = csv_rows(ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv")
        run([str(PY), "08b_ultimate_merge.py"], cwd=scripts)
        after = csv_rows(ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv")
        print(f"[DB] official training rows before={before} after={after} delta={after-before}", flush=True)
        run([str(PY), "18_build_10d_target_matrix.py"], cwd=scripts)
        run([str(PY), "20_prepare_xtb_missing_for_current_db.py", "--tag", args.tag], cwd=scripts)

    print("[DONE] AL round finalized into canonical workflow files.", flush=True)


if __name__ == "__main__":
    main()