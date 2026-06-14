#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train-validation Tanimoto redundancy / similarity-leakage diagnostics.

Outputs:
  results/redundancy/val_train_tanimoto_max.csv
  results/redundancy/Table_Redundancy_Summary.csv
  results/redundancy/Table_Validation_SimilarityBin_Counts.csv
  results/redundancy/Figure_TrainVal_Tanimoto_Distribution.png/pdf

Optional, if validation predictions are available:
  results/redundancy/Table_Performance_by_SimilarityBin.csv
  results/redundancy/Figure_Error_vs_Tanimoto_HardTargets.png/pdf
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


TARGETS_HARD = [
    "HOMO_LUMO_Gap(eV)",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
]

SIM_BINS = [0.0, 0.4, 0.6, 0.8, 0.9, 1.0000001]
SIM_LABELS = ["[0.0,0.4)", "[0.4,0.6)", "[0.6,0.8)", "[0.8,0.9)", "[0.9,1.0]"]


def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def find_smiles_column(df):
    candidates = [
        "SMILES", "smiles", "canonical_smiles", "Canonical_SMILES",
        "mol_smiles", "structure_smiles", "SELFIES_SMILES"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        nc = norm_name(c)
        if "smiles" in nc:
            return c

    raise ValueError(
        "Could not find a SMILES column. Available columns:\n"
        + "\n".join(map(str, df.columns))
    )


def find_id_column(df):
    candidates = [
        "ID", "id", "Name", "name", "molecule_id", "mol_id",
        "candidate_id", "Candidate_ID", "sample_id", "Sample_ID"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        nc = norm_name(c)
        if nc in {"id", "name", "moleculeid", "molid", "candidateid", "sampleid"}:
            return c

    return None


def flatten_json_values(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            if isinstance(v, list):
                out.extend(v)
            elif isinstance(v, dict):
                out.extend(flatten_json_values(v))
        return out
    return []


def pick_split_keys(split_obj):
    """
    Flexible parser for common split JSON formats.

    Accepted examples:
      {"train": [...], "val": [...]}
      {"train_idx": [...], "val_idx": [...]}
      {"train_indices": [...], "validation_indices": [...]}
      {"Train": [...], "Val": [...]}
      {"split": {"train": [...], "val": [...]}}
    """
    if not isinstance(split_obj, dict):
        raise ValueError("Split JSON top-level object must be a dict.")

    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, list):
                    flat[key] = v
                elif isinstance(v, dict):
                    walk(key, v)

    walk("", split_obj)

    if not flat:
        raise ValueError("No list-like split entries found in split JSON.")

    train_keys = []
    val_keys = []

    for k in flat:
        nk = norm_name(k)
        if (
            "train" in nk
            and "val" not in nk
            and "valid" not in nk
            and "calib" not in nk
            and "test" not in nk
        ):
            train_keys.append(k)

        if (
            "val" in nk
            or "valid" in nk
            or nk.endswith("validationindices")
            or nk.endswith("validindices")
        ):
            val_keys.append(k)

    if not train_keys:
        raise ValueError(
            "Could not identify training split key. Candidate keys:\n"
            + "\n".join(flat.keys())
        )

    if not val_keys:
        raise ValueError(
            "Could not identify validation split key. Candidate keys:\n"
            + "\n".join(flat.keys())
        )

    # Prefer explicit train_core only if no ordinary train key exists.
    train_keys_sorted = sorted(
        train_keys,
        key=lambda k: (
            0 if norm_name(k).endswith("trainindices") else 1,
            0 if norm_name(k).endswith("trainidx") else 1,
            0 if norm_name(k).endswith("train") else 1,
            len(k),
        ),
    )

    val_keys_sorted = sorted(
        val_keys,
        key=lambda k: (
            0 if "valindices" in norm_name(k) else 1,
            0 if "validindices" in norm_name(k) else 1,
            0 if norm_name(k).endswith("val") else 1,
            len(k),
        ),
    )

    return train_keys_sorted[0], val_keys_sorted[0], flat[train_keys_sorted[0]], flat[val_keys_sorted[0]], flat


def convert_split_entries_to_positions(entries, df, id_col, split_name):
    """
    Convert split entries to integer row positions.
    Supports:
      - 0-based integer positions
      - 1-based integer positions
      - dataframe index values
      - string IDs matched to id_col
    """
    n = len(df)

    if len(entries) == 0:
        raise ValueError(f"{split_name} split is empty.")

    # Case 1: integer-like entries
    integer_like = True
    ints = []
    for x in entries:
        try:
            if isinstance(x, bool):
                integer_like = False
                break
            fx = float(x)
            ix = int(fx)
            if abs(fx - ix) > 1e-12:
                integer_like = False
                break
            ints.append(ix)
        except Exception:
            integer_like = False
            break

    if integer_like:
        arr = np.array(ints, dtype=int)

        # 0-based positions
        if arr.min() >= 0 and arr.max() < n:
            return arr.tolist()

        # 1-based positions
        if arr.min() >= 1 and arr.max() <= n:
            return (arr - 1).tolist()

        # Match dataframe index values
        index_to_pos = {int(idx): pos for pos, idx in enumerate(df.index) if str(idx).isdigit()}
        if all(ix in index_to_pos for ix in arr):
            return [index_to_pos[ix] for ix in arr]

        raise ValueError(
            f"{split_name} integer entries do not match 0-based positions, "
            f"1-based positions, or dataframe integer index."
        )

    # Case 2: string IDs
    if id_col is None:
        raise ValueError(
            f"{split_name} split entries are not integer-like, but no ID column was found."
        )

    id_to_pos = {str(v): i for i, v in enumerate(df[id_col].astype(str).tolist())}
    missing = [str(x) for x in entries if str(x) not in id_to_pos]
    if missing:
        raise ValueError(
            f"{split_name} contains IDs not found in column {id_col}. "
            f"First missing values: {missing[:10]}"
        )

    return [id_to_pos[str(x)] for x in entries]


def mol_from_smiles(smi):
    if not isinstance(smi, str) or not smi.strip():
        return None
    return Chem.MolFromSmiles(smi)


def fp_from_mol(mol, radius=2, nbits=2048):
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def detect_prediction_file(project_root):
    candidates = [
        "results/baselines/final_specialist_hybrid_v2_predictions.csv",
        "results/baselines/final_specialist_hybrid_v2_val_predictions.csv",
        "results/baselines/final_specialist_v2_predictions.csv",
        "results/baselines/final_specialist_v2_val_predictions.csv",
        "results/final_model_release/final_specialist_v2_candidate_predictions.csv",
        "results/final_model_release/final_specialist_v2_val_predictions.csv",
        "results/predictions/final_specialist_hybrid_v2_predictions.csv",
        "results/predictions/final_specialist_v2_predictions.csv",
    ]

    for c in candidates:
        p = project_root / c
        if p.exists():
            return p

    return None


def find_column_by_patterns(columns, target, roles):
    """
    Match columns with normalized target name and role hints.
    roles example: ["true", "actual", "y"]
    """
    nt = norm_name(target)
    cols = list(columns)

    # Exact normalized target can be truth column if no role suffix.
    for c in cols:
        nc = norm_name(c)
        if nc == nt and "true" in roles:
            return c

    scored = []
    for c in cols:
        nc = norm_name(c)
        if nt in nc or nc in nt:
            score = 0
            for r in roles:
                if r in nc:
                    score += 10
            if "pred" in roles and ("prediction" in nc or "predict" in nc):
                score += 10
            score += min(len(nt), len(nc))
            scored.append((score, c))

    if not scored:
        return None

    scored.sort(reverse=True)
    if scored[0][0] >= 10:
        return scored[0][1]

    return None


def align_predictions(pred_df, val_info_df, id_col, smiles_col):
    """
    Align prediction rows to validation rows.

    Priority:
      1. row_index / dataset_index / original_index column
      2. ID column
      3. SMILES column
      4. same length as validation table, assume val order
    """
    pcols_norm = {norm_name(c): c for c in pred_df.columns}

    index_keys = [
        "rowindex", "datasetindex", "originalindex", "dataindex",
        "index", "idx", "sampleindex"
    ]

    for key in index_keys:
        if key in pcols_norm:
            c = pcols_norm[key]
            pred_df = pred_df.copy()
            pred_df["_align_key"] = pred_df[c].astype(str)
            v = val_info_df.copy()
            v["_align_key"] = v["row_index"].astype(str)
            merged = v.merge(pred_df, on="_align_key", how="left", suffixes=("", "_pred"))
            return merged, f"row index column: {c}"

    if id_col is not None and id_col in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["_align_key"] = pred_df[id_col].astype(str)
        v = val_info_df.copy()
        v["_align_key"] = v[id_col].astype(str)
        merged = v.merge(pred_df, on="_align_key", how="left", suffixes=("", "_pred"))
        return merged, f"ID column: {id_col}"

    # SMILES alignment
    smi_pred_col = None
    for c in pred_df.columns:
        if "smiles" in norm_name(c):
            smi_pred_col = c
            break

    if smi_pred_col is not None:
        pred_df = pred_df.copy()
        pred_df["_align_key"] = pred_df[smi_pred_col].astype(str)
        v = val_info_df.copy()
        v["_align_key"] = v[smiles_col].astype(str)
        merged = v.merge(pred_df, on="_align_key", how="left", suffixes=("", "_pred"))
        return merged, f"SMILES column: {smi_pred_col}"

    if len(pred_df) == len(val_info_df):
        merged = pd.concat(
            [val_info_df.reset_index(drop=True), pred_df.reset_index(drop=True)],
            axis=1
        )
        return merged, "same length as validation set; assumed validation order"

    raise ValueError(
        "Could not align prediction file to validation molecules. "
        "Please include row_index, dataset_index, ID, or SMILES in the prediction file."
    )


def add_prediction_diagnostics(pred_file, val_info_df, smiles_col, id_col, outdir):
    pred_df = pd.read_csv(pred_file)
    merged, align_note = align_predictions(pred_df, val_info_df, id_col, smiles_col)

    rows = []
    plot_rows = []

    for target in TARGETS_HARD:
        true_col = find_column_by_patterns(
            merged.columns,
            target,
            roles=["true", "actual", "ytrue", "label", "target"]
        )
        pred_col = find_column_by_patterns(
            merged.columns,
            target,
            roles=["pred", "prediction", "ypred", "final"]
        )

        if true_col is None:
            # If the original validation info contains the target, use it as truth.
            for c in val_info_df.columns:
                if norm_name(c) == norm_name(target):
                    true_col = c
                    break

        if true_col is None or pred_col is None:
            rows.append({
                "target": target,
                "status": "skipped",
                "reason": f"could not identify true/pred columns; true={true_col}, pred={pred_col}",
                "alignment": align_note,
            })
            continue

        tmp = merged[["similarity_bin", "max_train_tanimoto", true_col, pred_col]].copy()
        tmp[true_col] = pd.to_numeric(tmp[true_col], errors="coerce")
        tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
        tmp = tmp.dropna(subset=[true_col, pred_col, "max_train_tanimoto"])

        if tmp.empty:
            rows.append({
                "target": target,
                "status": "skipped",
                "reason": "no numeric rows after dropping NA",
                "alignment": align_note,
            })
            continue

        tmp["abs_error"] = (tmp[pred_col] - tmp[true_col]).abs()
        tmp["sq_error"] = (tmp[pred_col] - tmp[true_col]) ** 2

        for bin_label, g in tmp.groupby("similarity_bin", observed=False):
            if len(g) == 0:
                continue

            y_true = g[true_col].to_numpy(dtype=float)
            y_pred = g[pred_col].to_numpy(dtype=float)

            ss_res = float(np.sum((y_pred - y_true) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2 = np.nan if ss_tot <= 1e-30 else 1.0 - ss_res / ss_tot

            rows.append({
                "target": target,
                "similarity_bin": str(bin_label),
                "n": int(len(g)),
                "mae": float(np.mean(np.abs(y_pred - y_true))),
                "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                "r2": r2,
                "true_column": true_col,
                "pred_column": pred_col,
                "alignment": align_note,
                "status": "ok",
            })

        for _, r in tmp.iterrows():
            plot_rows.append({
                "target": target,
                "max_train_tanimoto": r["max_train_tanimoto"],
                "abs_error": r["abs_error"],
            })

    perf = pd.DataFrame(rows)
    perf.to_csv(outdir / "Table_Performance_by_SimilarityBin.csv", index=False)

    plot_df = pd.DataFrame(plot_rows)
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for target, g in plot_df.groupby("target"):
            ax.scatter(g["max_train_tanimoto"], g["abs_error"], s=16, alpha=0.65, label=target)

        ax.set_xlabel("Maximum Tanimoto similarity to training set")
        ax.set_ylabel("Absolute validation error")
        ax.set_title("Error vs train-validation similarity for hard targets")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "Figure_Error_vs_Tanimoto_HardTargets.png", dpi=300)
        fig.savefig(outdir / "Figure_Error_vs_Tanimoto_HardTargets.pdf")
        plt.close(fig)

    return perf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/old_dataset.csv")
    parser.add_argument("--split", default="results/train_val_split_9d.json")
    parser.add_argument("--target_matrix", default="data/baselines/target_matrix_9d.csv")
    parser.add_argument("--pred_file", default=None)
    parser.add_argument("--outdir", default="results/redundancy")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--nbits", type=int, default=2048)
    args = parser.parse_args()

    project_root = Path(".").resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    split_path = Path(args.split)
    target_path = Path(args.target_matrix)

    df = pd.read_csv(dataset_path)
    smiles_col = find_smiles_column(df)
    id_col = find_id_column(df)

    # Attach target matrix columns if not already present and row counts match.
    if target_path.exists():
        target_df = pd.read_csv(target_path)
        if len(target_df) == len(df):
            for c in target_df.columns:
                if c not in df.columns:
                    df[c] = target_df[c]

    with open(split_path, "r", encoding="utf-8") as f:
        split_obj = json.load(f)

    train_key, val_key, train_entries, val_entries, all_keys = pick_split_keys(split_obj)

    train_pos = convert_split_entries_to_positions(train_entries, df, id_col, "train")
    val_pos = convert_split_entries_to_positions(val_entries, df, id_col, "validation")

    train_pos = sorted(set(train_pos))
    val_pos = sorted(set(val_pos))

    overlap = sorted(set(train_pos).intersection(set(val_pos)))
    if overlap:
        raise ValueError(f"Train/validation overlap detected in row positions: {overlap[:20]}")

    print("===== Split information =====")
    print(f"Dataset rows: {len(df)}")
    print(f"SMILES column: {smiles_col}")
    print(f"ID column: {id_col}")
    print(f"Train key: {train_key}; n={len(train_pos)}")
    print(f"Validation key: {val_key}; n={len(val_pos)}")
    print(f"All split list keys: {list(all_keys.keys())}")

    train_smiles = df.iloc[train_pos][smiles_col].astype(str).tolist()
    val_smiles = df.iloc[val_pos][smiles_col].astype(str).tolist()

    train_mols = [mol_from_smiles(s) for s in train_smiles]
    val_mols = [mol_from_smiles(s) for s in val_smiles]

    train_fps = [fp_from_mol(m, args.radius, args.nbits) for m in train_mols]
    val_fps = [fp_from_mol(m, args.radius, args.nbits) for m in val_mols]

    valid_train = [(i, fp) for i, fp in enumerate(train_fps) if fp is not None]
    if not valid_train:
        raise ValueError("No valid training fingerprints could be generated.")

    valid_train_indices_local = [x[0] for x in valid_train]
    valid_train_fps = [x[1] for x in valid_train]

    records = []

    for local_val_i, fp in enumerate(val_fps):
        row_index = val_pos[local_val_i]
        smi = val_smiles[local_val_i]

        base = {
            "validation_order": local_val_i,
            "row_index": row_index,
            smiles_col: smi,
        }

        if id_col is not None:
            base[id_col] = df.iloc[row_index][id_col]

        if fp is None:
            base.update({
                "max_train_tanimoto": np.nan,
                "mean_train_tanimoto": np.nan,
                "top_train_row_index": np.nan,
                "top_train_smiles": None,
                "top_train_id": None,
                "n_train_compared": len(valid_train_fps),
                "mol_parse_status": "failed",
            })
            records.append(base)
            continue

        sims = DataStructs.BulkTanimotoSimilarity(fp, valid_train_fps)
        sims = np.asarray(sims, dtype=float)

        max_j = int(np.argmax(sims))
        max_sim = float(sims[max_j])
        mean_sim = float(np.mean(sims))
        top_local_train_i = valid_train_indices_local[max_j]
        top_row_index = train_pos[top_local_train_i]

        base.update({
            "max_train_tanimoto": max_sim,
            "mean_train_tanimoto": mean_sim,
            "top_train_row_index": int(top_row_index),
            "top_train_smiles": df.iloc[top_row_index][smiles_col],
            "top_train_id": df.iloc[top_row_index][id_col] if id_col is not None else None,
            "n_train_compared": len(valid_train_fps),
            "mol_parse_status": "ok",
        })

        records.append(base)

    sim_df = pd.DataFrame(records)
    sim_df["similarity_bin"] = pd.cut(
        sim_df["max_train_tanimoto"],
        bins=SIM_BINS,
        labels=SIM_LABELS,
        include_lowest=True,
        right=False,
    )

    sim_out = outdir / "val_train_tanimoto_max.csv"
    sim_df.to_csv(sim_out, index=False)

    valid_sims = sim_df["max_train_tanimoto"].dropna().to_numpy(dtype=float)

    summary_rows = []
    summary_rows.append({"metric": "dataset_n", "value": len(df)})
    summary_rows.append({"metric": "train_n", "value": len(train_pos)})
    summary_rows.append({"metric": "validation_n", "value": len(val_pos)})
    summary_rows.append({"metric": "valid_validation_fingerprint_n", "value": int(len(valid_sims))})
    summary_rows.append({"metric": "invalid_validation_smiles_n", "value": int(sim_df["max_train_tanimoto"].isna().sum())})
    summary_rows.append({"metric": "max_similarity_mean", "value": float(np.mean(valid_sims))})
    summary_rows.append({"metric": "max_similarity_median", "value": float(np.median(valid_sims))})
    summary_rows.append({"metric": "max_similarity_std", "value": float(np.std(valid_sims, ddof=1)) if len(valid_sims) > 1 else np.nan})
    summary_rows.append({"metric": "max_similarity_min", "value": float(np.min(valid_sims))})
    summary_rows.append({"metric": "max_similarity_max", "value": float(np.max(valid_sims))})
    summary_rows.append({"metric": "fraction_ge_0.8", "value": float(np.mean(valid_sims >= 0.8))})
    summary_rows.append({"metric": "fraction_ge_0.9", "value": float(np.mean(valid_sims >= 0.9))})
    summary_rows.append({"metric": "fraction_eq_1.0_tol_1e-12", "value": float(np.mean(np.isclose(valid_sims, 1.0, atol=1e-12)))})
    summary_rows.append({"metric": "train_split_key", "value": train_key})
    summary_rows.append({"metric": "validation_split_key", "value": val_key})
    summary_rows.append({"metric": "smiles_column", "value": smiles_col})
    summary_rows.append({"metric": "id_column", "value": id_col if id_col is not None else ""})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(outdir / "Table_Redundancy_Summary.csv", index=False)

    bin_counts = (
        sim_df.groupby("similarity_bin", observed=False)
        .agg(
            n=("max_train_tanimoto", "count"),
            mean_max_tanimoto=("max_train_tanimoto", "mean"),
            median_max_tanimoto=("max_train_tanimoto", "median"),
            min_max_tanimoto=("max_train_tanimoto", "min"),
            max_max_tanimoto=("max_train_tanimoto", "max"),
        )
        .reset_index()
    )
    bin_counts.to_csv(outdir / "Table_Validation_SimilarityBin_Counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.hist(valid_sims, bins=np.linspace(0, 1, 31), edgecolor="black", linewidth=0.4)
    ax.axvline(0.8, linestyle="--", linewidth=1.0)
    ax.axvline(0.9, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Maximum Tanimoto similarity to training set")
    ax.set_ylabel("Validation molecule count")
    ax.set_title("Train-validation chemical similarity diagnostic")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "Figure_TrainVal_Tanimoto_Distribution.png", dpi=300)
    fig.savefig(outdir / "Figure_TrainVal_Tanimoto_Distribution.pdf")
    plt.close(fig)

    pred_file = Path(args.pred_file) if args.pred_file else detect_prediction_file(project_root)
    if pred_file is not None and pred_file.exists():
        print(f"===== Prediction diagnostics =====")
        print(f"Using prediction file: {pred_file}")
        try:
            perf = add_prediction_diagnostics(pred_file, sim_df, smiles_col, id_col, outdir)
            print(f"Wrote prediction/bin diagnostics with {len(perf)} rows.")
        except Exception as exc:
            msg = outdir / "PENDING_Error_vs_Tanimoto_README.txt"
            msg.write_text(
                "Prediction file was detected, but error-vs-similarity diagnostics failed.\n"
                f"Detected prediction file: {pred_file}\n"
                f"Error: {repr(exc)}\n"
                "Action: provide a validation prediction CSV with row_index/dataset_index/ID/SMILES "
                "and true/pred columns for hard targets.\n",
                encoding="utf-8"
            )
            print(f"[WARN] prediction diagnostics failed. See {msg}")
    else:
        msg = outdir / "PENDING_Error_vs_Tanimoto_README.txt"
        msg.write_text(
            "No validation prediction file was automatically detected.\n"
            "The train-validation Tanimoto similarity diagnostic has been completed.\n"
            "To generate error-vs-similarity diagnostics, rerun with:\n"
            "  python -u scripts/analyze_train_val_redundancy.py --pred_file PATH_TO_VAL_PREDICTIONS.csv\n\n"
            "Expected prediction CSV alignment columns: row_index, dataset_index, ID, or SMILES.\n"
            "Expected hard-target true/pred columns can use names such as:\n"
            "  HOMO_LUMO_Gap(eV)_true, HOMO_LUMO_Gap(eV)_pred\n"
            "  VS_max_true, VS_max_pred\n"
            "  Sigma2_tot_true, Sigma2_tot_pred\n"
            "  Nu_true, Nu_pred\n"
            "  Trigger_Bond_Rho_true, Trigger_Bond_Rho_pred\n",
            encoding="utf-8"
        )
        print(f"[INFO] No prediction file detected. See {msg}")

    print("===== Outputs =====")
    for p in sorted(outdir.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
