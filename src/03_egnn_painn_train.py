"""
03_egnn_painn_train.py

Clean HELS 10-target training script with aligned vertical BDE labels.

Core modes
----------
1. 2d_only
   2D-only baseline family using Morgan fingerprints + compact energetic descriptors.
   This is retained only as a manuscript baseline.

2. 3d_only
   3D-only EGNN baseline using atom type + coordinates + radius graph.
   This is retained only as a manuscript baseline.

3. final_specialist
   Final-Specialist v2:
   target-wise 2D+xTB specialist teacher ensemble + xTB-aware EGNN residual correction.
   Codex 2026-06-02 extension:
   optional weak-bond/BDE manifest descriptors, conservative residual gating,
   and npj/SI model-evidence exports.

Design notes
------------
- No validation leakage is used in final_specialist. The validation set is held out until final reporting.
- Teacher selection uses out-of-fold training-core predictions rather than a single calibration set.
- The residual EGNN receives both compact 2D descriptors and xTB global electronic descriptors.
- Per-target residual alpha is fitted only on the calibration subset with a conservative improvement gate.
- Density_calc(g/cm3) is treated as molecular-volume-derived proxy density unless true crystal density
  labels are explicitly supplied elsewhere.
- Optional weak-bond descriptors are disabled by default. Completed vertical BDE values are target labels and are never used as input features.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, radius_graph


# ==============================================================================
# Paths and constants
# ==============================================================================
OLD_CSV_PATH = "../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
XYZ_DIR = "../data/raw_2100_xyz"
MODEL_SAVE_PATH = "../results/clean_training/model_default.pth"
SPLIT_SAVE_PATH = "../results/clean_training/split_default.json"

TARGET_PROPS = [
    "Density_calc(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
]

CRYSTAL_DENSITY_CANDIDATE_COLS = [
    "Crystal_Density(g/cm3)",
    "Density_crystal(g/cm3)",
    "rho_crystal",
    "Density_calibrated(g/cm3)",
]

SENSITIVITY_TARGETS = {
    "HOMO_LUMO_Gap(eV)",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Vertical_BDE(kcal/mol)",
}

BDE_NUMERIC_CANDIDATE_COLS = [
    "Bond_WBO",
    "BDE_Bond_WBO",
    "BDE_Bond_i_1based",
    "BDE_Bond_j_1based",
    "Bond_Order",
    "Bond_IsRing",
    "Bond_IsAromatic",
    "Bond_Selection_Priority",
    "Parent_Charge",
    "Parent_Mult",
    "FragA_Charge",
    "FragB_Charge",
    "Frag_Mult",
    "FragA_NAtoms",
    "FragB_NAtoms",
]

# Completed BDE values are now supervised targets. Keep this empty so the
# optional weak-bond feature hook cannot leak the Vertical_BDE label into X.
BDE_VALUE_CANDIDATE_COLS: List[str] = []

BDE_BOND_TYPES = ["C-N", "N-N", "N-O", "O-O", "C-O", "C-C"]


PRETTY_TARGETS = {
    "Density_calc(g/cm3)": "Density",
    "Heat_of_Formation(kcal/mol)": "Heat_of_Formation",
    "HOMO_LUMO_Gap(eV)": "HOMO_LUMO_Gap",
    "SAscore": "SA_Score",
    "VS_max": "VS_max",
    "Sigma2_tot": "Sigma2_tot",
    "Nu": "Nu",
    "Trigger_Bond_Rho": "Trigger_Bond_Rho",
    "Molecular_Weight": "Molecular_Weight",
    "Vertical_BDE(kcal/mol)": "Vertical_BDE",
}

TWO_D_FEATURE_NAMES = [
    # original compact energetic descriptors
    "ExactMolWt",
    "NumHeteroatoms",
    "NumRings",
    "NumNitro",
    "NumNitrogen",
    "NumC_NO2",
    "NumN_NO2",
    "NumN_eq_N",
    "NumAzide",

    # sensitivity_v3 trigger-linkage / explosophore descriptors
    "Num_Nitramine_NNO2",
    "Num_Nitrate_Ester_ONO2",
    "Num_Nitroaromatic_CNO2",
    "Num_Gem_Dinitro",
    "Num_Furazan",
    "Num_Tetrazole",
    "Num_Triazole",
    "Num_Nitroso",
    "Num_N_Oxide",
    "Num_NitroAdjacentPairs",
    "Max_Nitro_Adjacency",
    "Nitro_Nitrogen_Ratio",
    "Nitro_Per_HeavyAtom",
    "Oxygen_Balance_100",
    "DBE",
    "Aromatic_Ring_Count",
    "Heteroaromatic_Ring_Count",
    "Nitrogen_Oxygen_Ratio",
    "Explosophore_Count",
    "Trigger_Linkage_Count",
]

ATOMIC_NUMS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17}



def apply_density_label_policy(df_all: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Optionally replace Density_calc(g/cm3) with true/calibrated crystal-density labels.

    This does not create crystal densities. It only uses a density column if the
    active clean dataset already contains it.
    """
    df_all = df_all.copy()
    used_col = "Density_calc(g/cm3)"
    mode = getattr(args, "density_label_mode", "proxy_only")

    if mode == "crystal_preferred":
        for col in CRYSTAL_DENSITY_CANDIDATE_COLS:
            if col in df_all.columns:
                vals = pd.to_numeric(df_all[col], errors="coerce")
                good = vals.notna() & np.isfinite(vals) & (vals > 0.5) & (vals < 3.5)
                if int(good.sum()) >= max(50, int(0.05 * len(df_all))):
                    df_all.loc[good, "Density_calc(g/cm3)"] = vals.loc[good]
                    used_col = col
                    print(
                        f"[INFO] Density label policy: crystal_preferred | "
                        f"using {col} for {int(good.sum())}/{len(df_all)} rows; "
                        f"remaining rows keep Density_calc(g/cm3)."
                    )
                    break
        else:
            print("[INFO] Density label policy: crystal_preferred requested, but no usable crystal-density column was found.")
    else:
        print("[INFO] Density label policy: proxy_only | using Density_calc(g/cm3).")

    df_all["_Density_Label_Column_Used"] = used_col
    df_all["_Density_Label_Mode"] = mode
    return df_all


def apply_target_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Adjust training defaults for balanced/sensitivity/density profiles."""
    profile = getattr(args, "target_profile", "balanced")

    if profile == "sensitivity":
        args.loss_weight_mode = "focus_electronic"
        args.teacher_top_k = max(args.teacher_top_k, 4)
        args.teacher_cv_folds = max(args.teacher_cv_folds, 5)
        args.include_mlp_teacher = True
        args.alpha_max = max(args.alpha_max, 0.75)
        args.alpha_step = min(args.alpha_step, 0.025)
        args.alpha_min_improvement = min(args.alpha_min_improvement, 0.002)
        print("[INFO] Target profile: sensitivity | stronger teacher CV/top-k + electronic loss focus.")
    elif profile == "density":
        args.loss_weight_mode = "no_mw"
        print("[INFO] Target profile: density | no_mw loss profile, density label policy controlled separately.")
    else:
        print("[INFO] Target profile: balanced.")

    return args



# ==============================================================================
# Reproducibility and metrics
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))



def _quantile_class(values: np.ndarray, q1: float, q2: float, low_is_high_risk: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if low_is_high_risk:
        cls = np.where(values <= q1, 2, np.where(values <= q2, 1, 0))
    else:
        cls = np.where(values >= q2, 2, np.where(values >= q1, 1, 0))
    return cls.astype(int)


def _classification_metrics(y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> Dict[str, float]:
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    y_pred_cls = np.asarray(y_pred_cls, dtype=int)
    acc = float(np.mean(y_true_cls == y_pred_cls))
    rows = {"accuracy": acc}

    # macro F1 without sklearn dependency
    f1s = []
    for c in [0, 1, 2]:
        tp = float(np.sum((y_true_cls == c) & (y_pred_cls == c)))
        fp = float(np.sum((y_true_cls != c) & (y_pred_cls == c)))
        fn = float(np.sum((y_true_cls == c) & (y_pred_cls != c)))
        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        rows[f"f1_class_{c}"] = float(f1)

    rows["macro_f1"] = float(np.mean(f1s))
    return rows


def sensitivity_classification_auxiliary_report(
    y_core: np.ndarray,
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    val_examples: List["RawExample"],
    output_dir: str,
    run_tag: str,
) -> pd.DataFrame:
    """Auxiliary classification diagnostics for sensitivity proxy targets.

    Classes:
      Trigger_Bond_Rho: low rho = high risk.
      Vertical_BDE: low weakest-bond BDE = high risk.
      VS_max, Sigma2_tot, Nu: high value = high risk.
      HOMO_LUMO_Gap: low gap = high risk.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_to_idx = {t: i for i, t in enumerate(TARGET_PROPS)}

    specs = [
        ("Trigger_Bond_Rho", True),
        ("Vertical_BDE(kcal/mol)", True),
        ("HOMO_LUMO_Gap(eV)", True),
        ("VS_max", False),
        ("Sigma2_tot", False),
        ("Nu", False),
    ]

    rows = []
    per_mol_rows = []

    for target, low_is_high_risk in specs:
        if target not in target_to_idx:
            continue
        j = target_to_idx[target]
        q1, q2 = np.quantile(y_core[:, j], [1.0 / 3.0, 2.0 / 3.0])

        true_cls = _quantile_class(y_val[:, j], q1, q2, low_is_high_risk=low_is_high_risk)
        pred_cls = _quantile_class(y_pred_val[:, j], q1, q2, low_is_high_risk=low_is_high_risk)
        met = _classification_metrics(true_cls, pred_cls)

        rows.append({
            "Target": PRETTY_TARGETS.get(target, target),
            "Class_Definition": "0=low-risk,1=mid,2=high-risk",
            "Low_Value_Is_High_Risk": bool(low_is_high_risk),
            "Q33_train_core": float(q1),
            "Q66_train_core": float(q2),
            "Accuracy": met["accuracy"],
            "Macro_F1": met["macro_f1"],
            "F1_low_risk_class0": met["f1_class_0"],
            "F1_mid_class1": met["f1_class_1"],
            "F1_high_risk_class2": met["f1_class_2"],
            "Val_N": int(len(y_val)),
        })

        for i, ex in enumerate(val_examples):
            per_mol_rows.append({
                "Molecule": ex.molecule,
                "SMILES": ex.smiles,
                "Target": PRETTY_TARGETS.get(target, target),
                "True_Value": float(y_val[i, j]),
                "Pred_Value": float(y_pred_val[i, j]),
                "True_Class": int(true_cls[i]),
                "Pred_Class": int(pred_cls[i]),
                "Class_Error": int(pred_cls[i] - true_cls[i]),
                "Abs_Error": float(abs(y_pred_val[i, j] - y_val[i, j])),
            })

    report = pd.DataFrame(rows)
    per_mol = pd.DataFrame(per_mol_rows)

    report_path = os.path.join(output_dir, f"sensitivity_classification_report_{run_tag}.csv")
    detail_path = os.path.join(output_dir, f"sensitivity_classification_per_molecule_{run_tag}.csv")

    report.to_csv(report_path, index=False)
    per_mol.to_csv(detail_path, index=False)

    print("\n📊 Sensitivity proxy classification auxiliary report:")
    print(report.to_string(index=False))
    print(f"📄 Sensitivity classification report saved: {report_path}")
    print(f"📄 Sensitivity per-molecule classification saved: {detail_path}")
    return report


def sensitivity_active_learning_report(
    y_core: np.ndarray,
    y_val: np.ndarray,
    teacher_val: np.ndarray,
    final_val: np.ndarray,
    val_examples: List["RawExample"],
    output_dir: str,
    run_tag: str,
    top_k: int = 80,
) -> pd.DataFrame:
    """Rank validation molecules by sensitivity uncertainty/error for AL planning.

    The score combines:
      - disagreement between final model and teacher model,
      - normalized absolute error on sensitivity proxy targets,
      - high-risk true/predicted class mismatch.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_to_idx = {t: i for i, t in enumerate(TARGET_PROPS)}
    sensitivity_targets = [
        "Trigger_Bond_Rho",
        "Vertical_BDE(kcal/mol)",
        "HOMO_LUMO_Gap(eV)",
        "VS_max",
        "Sigma2_tot",
        "Nu",
    ]
    indices = [target_to_idx[t] for t in sensitivity_targets if t in target_to_idx]
    if not indices:
        return pd.DataFrame()

    scale = np.nanstd(y_core[:, indices], axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)

    abs_err = np.abs(final_val[:, indices] - y_val[:, indices]) / scale.reshape(1, -1)
    disagreement = np.abs(final_val[:, indices] - teacher_val[:, indices]) / scale.reshape(1, -1)

    score = 0.70 * abs_err.mean(axis=1) + 0.30 * disagreement.mean(axis=1)

    rows = []
    for i, ex in enumerate(val_examples):
        row = {
            "Rank_Source": "validation_error_disagreement",
            "Molecule": ex.molecule,
            "SMILES": ex.smiles,
            "Sensitivity_AL_Score": float(score[i]),
            "Mean_Normalized_Abs_Error": float(abs_err[i].mean()),
            "Mean_Teacher_Final_Disagreement": float(disagreement[i].mean()),
        }
        for k, target in zip(indices, sensitivity_targets):
            pretty = PRETTY_TARGETS.get(target, target)
            row[f"True_{pretty}"] = float(y_val[i, k])
            row[f"Pred_{pretty}"] = float(final_val[i, k])
            row[f"AbsErrNorm_{pretty}"] = float(abs(final_val[i, k] - y_val[i, k]) / scale[list(indices).index(k)])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Sensitivity_AL_Score", ascending=False).reset_index(drop=True)
    df["AL_Rank"] = np.arange(1, len(df) + 1)
    out_path = os.path.join(output_dir, f"sensitivity_active_learning_candidates_{run_tag}.csv")
    df.head(top_k).to_csv(out_path, index=False)

    print("\n🎯 Sensitivity active-learning candidates from validation uncertainty/error:")
    print(df.head(min(20, top_k))[["AL_Rank", "Molecule", "Sensitivity_AL_Score", "Mean_Normalized_Abs_Error", "Mean_Teacher_Final_Disagreement"]].to_string(index=False))
    print(f"📄 Sensitivity AL candidate table saved: {out_path}")
    return df.head(top_k)



def metrics_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_group: str,
    model_name: str,
    train_n: int,
    val_n: int,
    split_id: str,
    input_features: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for j, target in enumerate(TARGET_PROPS):
        rows.append(
            {
                "Model_Group": model_group,
                "Model": model_name,
                "Target": PRETTY_TARGETS.get(target, target),
                "MAE": mae_np(y_true[:, j], y_pred[:, j]),
                "RMSE": rmse_np(y_true[:, j], y_pred[:, j]),
                "R2": r2_score_np(y_true[:, j], y_pred[:, j]),
                "Train_N": int(train_n),
                "Val_N": int(val_n),
                "Split_ID": split_id,
                "Input_Features": input_features,
                "Status": "Done",
            }
        )
    return pd.DataFrame(rows)


def _safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _ci95_mean(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean
    half_width = 1.96 * float(np.std(values, ddof=1)) / float(np.sqrt(values.size))
    return mean - half_width, mean + half_width


def _feature_category(name: str) -> str:
    if name.startswith("morgan_"):
        return "Morgan fingerprint"
    if name in TWO_D_FEATURE_NAMES:
        return "compact energetic descriptor"
    if name.startswith("eng_"):
        return "engineered xTB interaction"
    if name.startswith("bde_"):
        return "optional BDE/weak-bond descriptor"
    if name.startswith("xtb_"):
        return "raw xTB descriptor"
    return "other"


def build_validation_prediction_audit(
    y_true: np.ndarray,
    teacher_pred: np.ndarray,
    final_pred: np.ndarray,
    val_examples: List["RawExample"],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(val_examples):
        row: Dict[str, Any] = {
            "Row_Index": int(ex.row_index),
            "Molecule": ex.molecule,
            "SMILES": ex.smiles,
            "Example_ID": ex.example_id,
        }
        for j, target in enumerate(TARGET_PROPS):
            pretty = PRETTY_TARGETS.get(target, target)
            row[f"True_{pretty}"] = float(y_true[i, j])
            row[f"TeacherPred_{pretty}"] = float(teacher_pred[i, j])
            row[f"FinalPred_{pretty}"] = float(final_pred[i, j])
            row[f"TeacherAbsErr_{pretty}"] = float(abs(teacher_pred[i, j] - y_true[i, j]))
            row[f"FinalAbsErr_{pretty}"] = float(abs(final_pred[i, j] - y_true[i, j]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_error_quantile_report(
    y_true: np.ndarray,
    teacher_pred: np.ndarray,
    final_pred: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, pred in [
        ("2D+xTB teacher", teacher_pred),
        ("Final specialist", final_pred),
    ]:
        for j, target in enumerate(TARGET_PROPS):
            err = np.asarray(pred[:, j] - y_true[:, j], dtype=float)
            abs_err = np.abs(err)
            bias_ci_low, bias_ci_high = _ci95_mean(err)
            rows.append(
                {
                    "Model": model_name,
                    "Target": PRETTY_TARGETS.get(target, target),
                    "N": int(len(err)),
                    "Bias_Mean": float(np.mean(err)),
                    "Bias_CI95_Low": bias_ci_low,
                    "Bias_CI95_High": bias_ci_high,
                    "AbsErr_Median": float(np.quantile(abs_err, 0.50)),
                    "AbsErr_P75": float(np.quantile(abs_err, 0.75)),
                    "AbsErr_P90": float(np.quantile(abs_err, 0.90)),
                    "AbsErr_P95": float(np.quantile(abs_err, 0.95)),
                    "AbsErr_Max": float(np.max(abs_err)),
                }
            )
    return pd.DataFrame(rows)


def build_feature_manifest(
    teacher_feature_names: Sequence[str],
    xtb_feature_names: Sequence[str],
) -> pd.DataFrame:
    xtb_set = set(xtb_feature_names)
    rows: List[Dict[str, Any]] = []
    for i, name in enumerate(teacher_feature_names):
        used_by = "teacher_and_residual" if name in xtb_set else "teacher_only"
        rows.append(
            {
                "Feature_Index": int(i),
                "Feature_Name": name,
                "Feature_Category": _feature_category(name),
                "Used_By": used_by,
            }
        )
    return pd.DataFrame(rows)


def build_targetwise_policy_audit(
    teacher_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    teacher_bundles: Sequence["TargetTeacherBundle"],
    alpha_report: pd.DataFrame,
) -> pd.DataFrame:
    teacher = teacher_metrics[["Target", "MAE", "RMSE", "R2"]].rename(
        columns={"MAE": "Teacher_MAE", "RMSE": "Teacher_RMSE", "R2": "Teacher_R2"}
    )
    final = final_metrics[["Target", "MAE", "RMSE", "R2"]].rename(
        columns={"MAE": "Final_MAE", "RMSE": "Final_RMSE", "R2": "Final_R2"}
    )
    out = teacher.merge(final, on="Target", how="outer")
    out["Delta_R2_Final_minus_Teacher"] = out["Final_R2"] - out["Teacher_R2"]
    out["Delta_RMSE_Final_minus_Teacher"] = out["Final_RMSE"] - out["Teacher_RMSE"]
    selected = pd.DataFrame(
        [
            {
                "Target": PRETTY_TARGETS.get(tb.target, tb.target),
                "Teacher_Transform": tb.transform_name,
                "Selected_Teacher_Models": "+".join(tb.selected_names),
                "Teacher_OOF_R2": float(tb.oof_r2),
            }
            for tb in teacher_bundles
        ]
    )
    if len(selected):
        out = out.merge(selected, on="Target", how="left")
    if len(alpha_report):
        out = out.merge(alpha_report, on="Target", how="left")
    return out


def write_npj_claim_evidence_ledger(
    path: str,
    run_tag: str,
    target_audit_path: str,
    error_report_path: str,
    prediction_path: str,
    feature_manifest_path: str,
    bde_enabled: bool,
) -> None:
    bde_text = (
        "Optional BDE/weak-bond descriptors were enabled for this run. "
        "Claims must distinguish these training descriptors from completed candidate-level BDE oracle labels."
        if bde_enabled
        else "Optional BDE descriptors were not enabled for this run; BDE evidence remains an external oracle/planning artifact."
    )
    text = f"""# npj Model Evidence Ledger

Run tag: `{run_tag}`

## Evidence files

- Target-wise model audit: `{target_audit_path}`
- Validation error quantiles: `{error_report_path}`
- Validation predictions: `{prediction_path}`
- Feature manifest: `{feature_manifest_path}`

## Allowed manuscript wording

- The final specialist model uses target-wise 2D+xTB teacher selection and a calibrated residual EGNN branch.
- Improvements should be stated per target, using the target-wise audit table rather than a single global score.
- Density should be described as a proxy or crystal-preferred mixed label according to the recorded density label mode.
- Sensitivity-related targets should be described as electronic or trigger-bond proxies, not experimental impact sensitivity.

## Forbidden stronger wording

- Do not claim universal accuracy across all nine targets.
- Do not claim true crystal-density prediction when the label source is molecular-volume proxy density.
- Do not claim H50 or experimental impact-sensitivity prediction from these proxy targets.
- Do not describe candidate-level BDE evidence as available unless completed BDE oracle energies have been parsed.

## BDE/QTAIM status

{bde_text}

## Recommended npj use

Use this ledger as a claim gate before drafting Results and Supplementary Information. Each performance claim should point to one of the evidence files above and include the target, split, sample size, and uncertainty/quantile context where relevant.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export_npj_evidence_bundle(
    args: argparse.Namespace,
    y_val: np.ndarray,
    teacher_val: np.ndarray,
    final_val: np.ndarray,
    val_examples: List["RawExample"],
    teacher_metrics: pd.DataFrame,
    final_metrics: pd.DataFrame,
    teacher_bundles: Sequence["TargetTeacherBundle"],
    alpha_report: pd.DataFrame,
    teacher_feature_names: Sequence[str],
    xtb_feature_names: Sequence[str],
) -> Dict[str, str]:
    if not getattr(args, "export_npj_evidence", True):
        return {}

    out_dir = getattr(args, "npj_evidence_dir", "../manuscript_npJ/SI/model_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    run_tag = getattr(args, "run_tag", "final_specialist")

    prediction_path = os.path.join(out_dir, f"Supplementary_NPJ_Validation_Predictions_{run_tag}.csv")
    error_report_path = os.path.join(out_dir, f"Table_NPJ_Validation_Error_Quantiles_{run_tag}.csv")
    feature_manifest_path = os.path.join(out_dir, f"Table_NPJ_Feature_Manifest_{run_tag}.csv")
    target_audit_path = os.path.join(out_dir, f"Table_NPJ_Targetwise_Model_Audit_{run_tag}.csv")
    ledger_path = os.path.join(out_dir, f"Claim_Evidence_Ledger_{run_tag}.md")

    prediction_audit = build_validation_prediction_audit(y_val, teacher_val, final_val, val_examples)
    error_report = build_error_quantile_report(y_val, teacher_val, final_val)
    feature_manifest = build_feature_manifest(teacher_feature_names, xtb_feature_names)
    target_audit = build_targetwise_policy_audit(teacher_metrics, final_metrics, teacher_bundles, alpha_report)

    prediction_audit.to_csv(prediction_path, index=False)
    error_report.to_csv(error_report_path, index=False)
    feature_manifest.to_csv(feature_manifest_path, index=False)
    target_audit.to_csv(target_audit_path, index=False)
    write_npj_claim_evidence_ledger(
        ledger_path,
        run_tag=run_tag,
        target_audit_path=target_audit_path,
        error_report_path=error_report_path,
        prediction_path=prediction_path,
        feature_manifest_path=feature_manifest_path,
        bde_enabled=bool(getattr(args, "enable_bde_features", False)),
    )

    print("\n[NPJ] Evidence bundle exported:")
    for p in [target_audit_path, error_report_path, prediction_path, feature_manifest_path, ledger_path]:
        print(f"[NPJ] {p}")
    return {
        "target_audit": target_audit_path,
        "error_quantiles": error_report_path,
        "validation_predictions": prediction_path,
        "feature_manifest": feature_manifest_path,
        "claim_evidence_ledger": ledger_path,
    }


def safe_std(tensor: torch.Tensor) -> torch.Tensor:
    std = tensor.std(dim=0)
    std[std == 0] = 1.0
    std[torch.isnan(std)] = 1.0
    return std


# ==============================================================================
# Molecular featurization
# ==============================================================================
def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def make_example_id(row_index: int, molecule: str, smiles: str) -> str:
    return f"row{int(row_index):08d}||{molecule}||{canonicalize_smiles(smiles)}"


def _count_substructure(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def _safe_smarts_count(mol: Chem.Mol, smarts: str) -> int:
    try:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt, uniquify=True))
    except Exception:
        return 0


def _calc_dbe(mol: Chem.Mol) -> float:
    """Approximate double-bond equivalent from molecular formula."""
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        counts = {el: 0 for el in ["C", "H", "N", "F", "Cl", "Br", "I"]}
        for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
            if el in counts:
                counts[el] += int(num) if num else 1
        c = counts["C"]
        h = counts["H"]
        n = counts["N"]
        x = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
        return float((2 * c + 2 + n - h - x) / 2.0)
    except Exception:
        return 0.0


def _nitro_adjacency_features(mol: Chem.Mol) -> Tuple[int, int]:
    """Count nitro groups attached to the same atom or nearby atom neighborhood.

    Returns:
      total_adjacent_pairs, max_nitro_adjacency_per_atom
    """
    nitro = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
    if nitro is None:
        return 0, 0

    nitro_matches = mol.GetSubstructMatches(nitro, uniquify=True)
    nitro_n_atoms = set()
    for m in nitro_matches:
        # first atom of the SMARTS match is usually nitro N
        if len(m):
            nitro_n_atoms.add(int(m[0]))

    attachment_counts = {}
    for n_idx in nitro_n_atoms:
        atom = mol.GetAtomWithIdx(n_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() not in (7, 8):
                aidx = int(nbr.GetIdx())
                attachment_counts[aidx] = attachment_counts.get(aidx, 0) + 1
            elif nbr.GetAtomicNum() == 7:
                aidx = int(nbr.GetIdx())
                attachment_counts[aidx] = attachment_counts.get(aidx, 0) + 1

    max_adj = max(attachment_counts.values()) if attachment_counts else 0
    total_pairs = sum(v * (v - 1) // 2 for v in attachment_counts.values() if v >= 2)
    return int(total_pairs), int(max_adj)


def extract_2d_features(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [0.0] * len(TWO_D_FEATURE_NAMES)

    nitro = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
    num_nitro = _safe_smarts_count(mol, nitro)
    num_n = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    num_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    heavy = max(1, mol.GetNumHeavyAtoms())

    # Original descriptors
    exact_mw = float(Descriptors.ExactMolWt(mol))
    num_hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    num_rings = float(rdMolDescriptors.CalcNumRings(mol))
    num_c_no2 = float(_safe_smarts_count(mol, f"[#6]-{nitro}"))
    num_n_no2 = float(_safe_smarts_count(mol, f"[#7]-{nitro}"))
    num_n_eq_n = float(_safe_smarts_count(mol, "[#7]=[#7]"))
    num_azide = float(_safe_smarts_count(mol, "[N]=[N+]=[N-]"))

    # Explosophore / trigger-linkage descriptors
    num_nitramine = float(_safe_smarts_count(mol, f"[NX3,NX4]-{nitro}"))
    num_nitrate_ester = float(_safe_smarts_count(mol, "[OX2]-[N+](=O)[O-]"))
    num_nitroaromatic = float(_safe_smarts_count(mol, f"[a]-{nitro}"))
    num_gem_dinitro = float(_safe_smarts_count(mol, f"[#6]({nitro})({nitro})"))
    num_furazan = float(_safe_smarts_count(mol, "c1nonc1") + _safe_smarts_count(mol, "C1=NON=C1"))
    num_tetrazole = float(_safe_smarts_count(mol, "c1nnnn1") + _safe_smarts_count(mol, "C1=NN=NN1"))
    num_triazole = float(_safe_smarts_count(mol, "c1nncn1") + _safe_smarts_count(mol, "C1=NNC=N1"))
    num_nitroso = float(_safe_smarts_count(mol, "[N!+]=O"))
    num_n_oxide = float(_safe_smarts_count(mol, "[n+][O-]") + _safe_smarts_count(mol, "[N+][O-]"))

    nitro_adj_pairs, max_nitro_adj = _nitro_adjacency_features(mol)
    nitro_n_ratio = float(num_nitro / max(num_n, 1))
    nitro_heavy_ratio = float(num_nitro / heavy)

    # OB100 = 1600 * (O - 2C - H/2) / MW
    c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    h = sum(1 for atom in Chem.AddHs(mol).GetAtoms() if atom.GetAtomicNum() == 1)
    oxygen_balance_100 = float(1600.0 * (num_o - 2.0 * c - 0.5 * h) / max(exact_mw, 1e-6))

    dbe = float(_calc_dbe(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    heteroaromatic_rings = float(_safe_smarts_count(mol, "[a]1[a][a][a][a]1"))
    n_o_ratio = float(num_n / max(num_o, 1))

    explosophore_count = float(
        num_nitro
        + num_azide
        + num_nitramine
        + num_nitrate_ester
        + num_nitroso
        + num_n_oxide
    )
    trigger_linkage_count = float(
        num_c_no2
        + num_n_no2
        + num_nitramine
        + num_nitrate_ester
        + num_n_eq_n
        + num_azide
        + num_gem_dinitro
    )

    vals = [
        exact_mw,
        num_hetero,
        num_rings,
        float(num_nitro),
        float(num_n),
        num_c_no2,
        num_n_no2,
        num_n_eq_n,
        num_azide,

        num_nitramine,
        num_nitrate_ester,
        num_nitroaromatic,
        num_gem_dinitro,
        num_furazan,
        num_tetrazole,
        num_triazole,
        num_nitroso,
        num_n_oxide,
        float(nitro_adj_pairs),
        float(max_nitro_adj),
        nitro_n_ratio,
        nitro_heavy_ratio,
        oxygen_balance_100,
        dbe,
        aromatic_rings,
        heteroaromatic_rings,
        n_o_ratio,
        explosophore_count,
        trigger_linkage_count,
    ]

    if len(vals) != len(TWO_D_FEATURE_NAMES):
        raise RuntimeError(f"2D feature length mismatch: {len(vals)} vs {len(TWO_D_FEATURE_NAMES)}")
    return [float(x) if np.isfinite(x) else 0.0 for x in vals]



def morgan_fp(smiles: str, fp_size: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros((fp_size,), dtype=np.float32)
    if mol is None:
        return arr
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def load_xyz_graph(mol_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    xyz_path = os.path.join(XYZ_DIR, f"{mol_name}.xyz")
    if not os.path.exists(xyz_path):
        return None, None

    z: List[int] = []
    pos: List[List[float]] = []
    try:
        with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None, None
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] in ATOMIC_NUMS:
                z.append(ATOMIC_NUMS[parts[0]])
                pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None, None

    if not z:
        return None, None
    return torch.tensor(z, dtype=torch.long), torch.tensor(pos, dtype=torch.float)


@dataclass
class RawExample:
    row_index: int
    example_id: str
    molecule: str
    smiles: str
    z: torch.Tensor
    pos: torch.Tensor
    y_raw: torch.Tensor
    x2d_raw: torch.Tensor


class EnergeticDataset(Dataset):
    def __init__(self, data_list: List[Data]):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def build_raw_examples(df_all: pd.DataFrame) -> List[RawExample]:
    mol_col = "Molecule" if "Molecule" in df_all.columns else "Moleule"
    examples: List[RawExample] = []

    for row_idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="内存挂载 3D 拓扑"):
        mol_name = str(row[mol_col]).replace(".xyz", "").replace(".out", "").strip()
        smiles = str(row["SMILES"])
        example_id = make_example_id(int(row_idx), mol_name, smiles)
        z, pos = load_xyz_graph(mol_name)
        if z is None or pos is None:
            continue

        y_raw = torch.tensor(row[TARGET_PROPS].astype(float).values, dtype=torch.float)
        x2d_raw = torch.tensor(extract_2d_features(smiles), dtype=torch.float)
        if torch.isnan(y_raw).any() or torch.isnan(x2d_raw).any():
            continue

        examples.append(
            RawExample(
                row_index=int(row_idx),
                example_id=example_id,
                molecule=mol_name,
                smiles=smiles,
                z=z,
                pos=pos,
                y_raw=y_raw,
                x2d_raw=x2d_raw,
            )
        )
    return examples


# ==============================================================================
# Native EGNN modules
# ==============================================================================
def native_scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    count = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index_expanded, torch.ones_like(src))
    return out / count.clamp(min=1)


class NativeEGNNLayer(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1, bias=False),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))

        coord_msg = coord_diff * self.coord_mlp(m_ij)
        pos_aggr = native_scatter_mean(coord_msg, row, dim_size=pos.size(0))
        pos_out = pos + pos_aggr

        m_aggr = native_scatter_mean(m_ij, row, dim_size=h.size(0))
        h_out = h + self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        return h_out, pos_out


class EGNNBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, radius: float = 4.0, max_neighbors: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.node_emb = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([NativeEGNNLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_emb(z)
        edge_index = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.max_neighbors)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return global_mean_pool(h, batch)



class True3DOnlyEGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_targets: int = 9):
        super().__init__()
        self.backbone = EGNNBackbone(hidden_dim=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_targets),
        )

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, x_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
        h_graph = self.backbone(z, pos, batch)
        return self.mlp(h_graph)


class XTBResidualEGNN(nn.Module):
    """xTB-aware residual EGNN used only by final_specialist v2."""

    def __init__(self, hidden_dim: int = 128, num_targets: int = 9, num_2d: int = 9, num_xtb: int = 1):
        super().__init__()
        self.backbone = EGNNBackbone(hidden_dim=hidden_dim)
        self.x2d_encoder = nn.Sequential(
            nn.Linear(num_2d, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.xtb_encoder = nn.Sequential(
            nn.Linear(num_xtb, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 128, 256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_targets),
        )

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        x_2d: torch.Tensor,
        x_xtb: torch.Tensor,
    ) -> torch.Tensor:
        h_graph = self.backbone(z, pos, batch)
        h_2d = self.x2d_encoder(x_2d.float())
        h_xtb = self.xtb_encoder(x_xtb.float())
        return self.head(torch.cat([h_graph, h_2d, h_xtb], dim=1))


# ==============================================================================
# Data conversion and training utilities
# ==============================================================================
def apply_coordinate_noise(pos: torch.Tensor, batch: torch.Tensor, std: float, prob: float) -> torch.Tensor:
    if std <= 0 or prob <= 0:
        return pos
    if torch.rand((), device=pos.device).item() > prob:
        return pos
    noise = torch.randn_like(pos) * std
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if n_graphs > 0:
        graph_mean_noise = native_scatter_mean(noise, batch, dim_size=n_graphs)
        noise = noise - graph_mean_noise[batch]
    return pos + noise


def make_data_from_examples(
    examples: List[RawExample],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x2d_mean: torch.Tensor,
    x2d_std: torch.Tensor,
) -> List[Data]:
    data_list: List[Data] = []
    for ex in examples:
        y_norm = ((ex.y_raw - y_mean) / y_std).unsqueeze(0)
        x2d_norm = ((ex.x2d_raw - x2d_mean) / x2d_std).unsqueeze(0)
        data_list.append(
            Data(
                z=ex.z,
                pos=ex.pos,
                y=y_norm,
                x_2d=x2d_norm,
                molecule=ex.molecule,
                smiles=ex.smiles,
                example_id=ex.example_id,
                row_index=int(ex.row_index),
            )
        )
    return data_list


def make_xtb_data_from_examples(
    examples: List[RawExample],
    residual_y_mean: torch.Tensor,
    residual_y_std: torch.Tensor,
    x2d_mean: torch.Tensor,
    x2d_std: torch.Tensor,
    xtb_map: Dict[int, np.ndarray],
    xtb_mean: torch.Tensor,
    xtb_std: torch.Tensor,
) -> List[Data]:
    data_list: List[Data] = []
    for ex in examples:
        y_norm = ((ex.y_raw - residual_y_mean) / residual_y_std).unsqueeze(0)
        x2d_norm = ((ex.x2d_raw - x2d_mean) / x2d_std).unsqueeze(0)
        xtb_raw = torch.tensor(xtb_map[int(ex.row_index)], dtype=torch.float)
        xtb_norm = ((xtb_raw - xtb_mean) / xtb_std).unsqueeze(0)
        data_list.append(
            Data(
                z=ex.z,
                pos=ex.pos,
                y=y_norm,
                x_2d=x2d_norm,
                x_xtb=xtb_norm,
                molecule=ex.molecule,
                smiles=ex.smiles,
                example_id=ex.example_id,
                row_index=int(ex.row_index),
            )
        )
    return data_list


def compute_multitask_huber_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = "uniform") -> torch.Tensor:
    per_element = F.huber_loss(pred, target, delta=1.0, reduction="none")
    per_target = per_element.mean(dim=0)

    if mode == "focus_electronic":
        weight_by_target = {
            "Density_calc(g/cm3)": 0.8,
            "Heat_of_Formation(kcal/mol)": 1.0,
            "HOMO_LUMO_Gap(eV)": 1.5,
            "SAscore": 0.7,
            "VS_max": 1.3,
            "Sigma2_tot": 1.5,
            "Nu": 1.5,
            "Trigger_Bond_Rho": 1.6,
            "Molecular_Weight": 0.2,
            "Vertical_BDE(kcal/mol)": 1.4,
        }
        weights = torch.tensor([weight_by_target.get(t, 1.0) for t in TARGET_PROPS], dtype=pred.dtype, device=pred.device)
    elif mode == "no_mw":
        weight_by_target = {
            "Density_calc(g/cm3)": 1.0,
            "Heat_of_Formation(kcal/mol)": 1.0,
            "HOMO_LUMO_Gap(eV)": 1.3,
            "SAscore": 0.8,
            "VS_max": 1.2,
            "Sigma2_tot": 1.4,
            "Nu": 1.4,
            "Trigger_Bond_Rho": 1.5,
            "Molecular_Weight": 0.05,
            "Vertical_BDE(kcal/mol)": 1.2,
        }
        weights = torch.tensor([weight_by_target.get(t, 1.0) for t in TARGET_PROPS], dtype=pred.dtype, device=pred.device)
    else:
        weights = torch.ones_like(per_target)
    if weights.numel() != per_target.numel():
        weights = torch.ones_like(per_target)
    return (per_target * weights).sum() / weights.sum()


def predict_standard_model_raw(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> np.ndarray:
    model.eval()
    blocks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch, batch.x_2d)
            raw = out.cpu() * y_std.cpu() + y_mean.cpu()
            blocks.append(raw.numpy())
    return np.vstack(blocks).astype(np.float32)


def predict_xtb_residual_raw(
    model: XTBResidualEGNN,
    loader: DataLoader,
    device: torch.device,
    resid_mean: torch.Tensor,
    resid_std: torch.Tensor,
) -> np.ndarray:
    model.eval()
    blocks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch, batch.x_2d, batch.x_xtb)
            raw = out.cpu() * resid_std.cpu() + resid_mean.cpu()
            blocks.append(raw.numpy())
    return np.vstack(blocks).astype(np.float32)


# ==============================================================================
# xTB feature processing for final_specialist v2
# ==============================================================================
def load_xtb_feature_table(xtb_feature_path: str) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(xtb_feature_path)
    if not path.exists():
        raise FileNotFoundError(f"xTB feature matrix not found: {path}")
    xtb = pd.read_csv(path)
    if "row_index" not in xtb.columns:
        raise RuntimeError("xTB feature matrix must contain row_index column.")

    exclude = {"row_index", "example_id", "Molecule", "molecule", "xtb_status", "xtb_workdir", "workdir"}
    numeric_cols: List[str] = []
    for c in xtb.columns:
        if c in exclude:
            continue
        if str(c).startswith("xtb_") and pd.api.types.is_numeric_dtype(xtb[c]):
            numeric_cols.append(c)

    if not numeric_cols:
        raise RuntimeError("No numeric xTB feature columns found.")

    ok_mask = xtb.get("xtb_status", "ok").astype(str).str.lower().eq("ok") if "xtb_status" in xtb.columns else np.ones(len(xtb), dtype=bool)
    xtb = xtb.loc[ok_mask].copy()
    xtb[numeric_cols] = xtb[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return xtb, numeric_cols


def _safe_get(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return 0.0
    try:
        val = float(row[col])
    except Exception:
        return 0.0
    if not np.isfinite(val):
        return 0.0
    return val


def load_optional_bde_feature_table(
    bde_feature_path: Optional[str],
    enabled: bool,
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """Load optional weak-bond/BDE descriptor rows keyed by training row_index.

    The current full-library manifest contains selected trigger-bond metadata
    and WBO-like weak-bond descriptors. Completed vertical BDE values are now
    supervised labels, so value columns are intentionally excluded here to
    prevent target leakage.
    """
    if not enabled:
        return {}, []
    if not bde_feature_path:
        print("[WARN] --enable_bde_features was set, but no --bde_feature_path was supplied.")
        return {}, []

    path = Path(bde_feature_path)
    if not path.exists():
        print(f"[WARN] Optional BDE feature table not found: {path}. Continuing without BDE features.")
        return {}, []

    bde = pd.read_csv(path)

    # If the aligned training table itself is used as the BDE feature source,
    # file order is the only safe key. Its legacy Row_Index column may come
    # from the pre-alignment manifest and is not guaranteed to match the reset
    # training-row index after BDE-incomplete rows were dropped.
    looks_like_training_table = "SMILES" in bde.columns and all(c in bde.columns for c in TARGET_PROPS)
    row_col = None
    if looks_like_training_table:
        row_col = "__row_index_from_file_order__"
        bde[row_col] = np.arange(len(bde), dtype=int)
        print(f"[INFO] Optional BDE feature table is an aligned training table; using CSV row order from {path}.")
    else:
        for cand in ["Row_Index", "row_index", "clean_row_index", "Clean_Row_Index"]:
            if cand in bde.columns:
                row_col = cand
                break
        if row_col is None:
            row_col = "__row_index_from_file_order__"
            bde[row_col] = np.arange(len(bde), dtype=int)
            print(f"[INFO] Optional BDE feature table lacks Row_Index; using CSV row order from {path}.")

    numeric_cols = [c for c in BDE_NUMERIC_CANDIDATE_COLS if c in bde.columns]
    value_cols = [
        c
        for c in BDE_VALUE_CANDIDATE_COLS
        if c in bde.columns and pd.api.types.is_numeric_dtype(bde[c])
    ]
    numeric_cols = numeric_cols + [c for c in value_cols if c not in numeric_cols]

    bond_type_col = "Bond_Type" if "Bond_Type" in bde.columns else ("BDE_Bond_Type" if "BDE_Bond_Type" in bde.columns else None)
    if not numeric_cols and bond_type_col is None:
        print(f"[WARN] Optional BDE feature table has no usable numeric/Bond_Type columns: {path}.")
        return {}, []

    feature_names = [f"bde_{c}" for c in numeric_cols]
    feature_names += [f"bde_bond_type_{bt.replace('-', '_')}" for bt in BDE_BOND_TYPES]
    feature_names.append("bde_bond_type_other")

    row_map: Dict[int, np.ndarray] = {}
    for _, row in bde.iterrows():
        try:
            row_index = int(row[row_col])
        except Exception:
            continue

        vals: List[float] = []
        for c in numeric_cols:
            try:
                val = float(row[c])
            except Exception:
                val = np.nan
            vals.append(val if np.isfinite(val) else np.nan)

        bond_type = str(row.get(bond_type_col, "")).strip() if bond_type_col else ""
        matched = False
        for bt in BDE_BOND_TYPES:
            is_match = bond_type == bt
            vals.append(1.0 if is_match else 0.0)
            matched = matched or is_match
        vals.append(0.0 if matched else 1.0)

        arr = np.asarray(vals, dtype=np.float32)
        if row_index in row_map:
            row_map[row_index] = np.nanmean(np.vstack([row_map[row_index], arr]), axis=0).astype(np.float32)
        else:
            row_map[row_index] = arr

    if not row_map:
        print(f"[WARN] Optional BDE feature table produced no matched rows: {path}.")
        return {}, []

    print(
        f"[INFO] Optional BDE feature hook enabled: {len(row_map)} row-index records, "
        f"{len(feature_names)} features from {path}."
    )
    return row_map, feature_names


def engineer_xtb_features(row: pd.Series) -> Tuple[np.ndarray, List[str]]:
    """Small physically motivated interaction features for hard electronic/ESP/QTAIM targets."""
    gap = _safe_get(row, "xtb_gap_eV")
    homo = _safe_get(row, "xtb_homo_eV")
    lumo = _safe_get(row, "xtb_lumo_eV")
    dipole = _safe_get(row, "xtb_dipole_D")
    ch_std = _safe_get(row, "xtb_charge_std")
    ch_abs = _safe_get(row, "xtb_charge_absmax")
    wbo_min = _safe_get(row, "xtb_wbo_min")
    wbo_mean = _safe_get(row, "xtb_wbo_mean")
    trig_wbo = _safe_get(row, "xtb_trigger_wbo_proxy_min")
    energy = _safe_get(row, "xtb_total_energy_Eh")

    eps = 1e-8
    vals = np.array(
        [
            gap * dipole,
            gap * ch_std,
            gap * ch_abs,
            dipole * ch_std,
            dipole * ch_abs,
            ch_std * ch_abs,
            trig_wbo * ch_abs,
            trig_wbo * ch_std,
            wbo_min * ch_abs,
            wbo_mean * ch_std,
            gap / (dipole + 1.0),
            ch_abs / (gap + 1.0),
            ch_std / (gap + 1.0),
            trig_wbo / (gap + 1.0),
            lumo - homo,
            abs(energy) / (abs(energy) + 100.0 + eps),
        ],
        dtype=np.float32,
    )
    names = [
        "eng_gap_x_dipole",
        "eng_gap_x_charge_std",
        "eng_gap_x_charge_absmax",
        "eng_dipole_x_charge_std",
        "eng_dipole_x_charge_absmax",
        "eng_charge_std_x_absmax",
        "eng_trigger_wbo_x_charge_absmax",
        "eng_trigger_wbo_x_charge_std",
        "eng_wbo_min_x_charge_absmax",
        "eng_wbo_mean_x_charge_std",
        "eng_gap_div_dipole_plus1",
        "eng_charge_absmax_div_gap_plus1",
        "eng_charge_std_div_gap_plus1",
        "eng_trigger_wbo_div_gap_plus1",
        "eng_lumo_minus_homo",
        "eng_abs_energy_scaled",
    ]
    return vals, names


def build_final_feature_matrices(
    raw_examples: List[RawExample],
    xtb_feature_path: str,
    fp_size: int = 2048,
    bde_feature_path: Optional[str] = None,
    enable_bde_features: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], List[RawExample], List[str], List[str]]:
    xtb, numeric_cols = load_xtb_feature_table(xtb_feature_path)
    bde_row_to_features, bde_feature_names = load_optional_bde_feature_table(
        bde_feature_path=bde_feature_path,
        enabled=enable_bde_features,
    )

    row_to_xtb: Dict[int, np.ndarray] = {}
    engineered_names: Optional[List[str]] = None
    missing_bde_count = 0

    for _, row in xtb.iterrows():
        row_index = int(row["row_index"])
        base = row[numeric_cols].astype(float).values.astype(np.float32)
        eng, names = engineer_xtb_features(row)
        if engineered_names is None:
            engineered_names = names

        parts = [base, eng]
        if bde_feature_names:
            bde_vec = bde_row_to_features.get(row_index)
            if bde_vec is None:
                missing_bde_count += 1
                bde_vec = np.full((len(bde_feature_names),), np.nan, dtype=np.float32)
            parts.append(bde_vec.astype(np.float32))
        row_to_xtb[row_index] = np.concatenate(parts, axis=0).astype(np.float32)

    if engineered_names is None:
        engineered_names = []

    aligned: List[RawExample] = [ex for ex in raw_examples if int(ex.row_index) in row_to_xtb]
    if not aligned:
        raise RuntimeError("No raw examples matched xTB feature rows.")

    X_teacher: List[np.ndarray] = []
    X_xtb: List[np.ndarray] = []
    for ex in aligned:
        fp = morgan_fp(ex.smiles, fp_size=fp_size)
        compact = ex.x2d_raw.detach().cpu().numpy().astype(np.float32)
        xtb_vec = row_to_xtb[int(ex.row_index)].astype(np.float32)
        X_teacher.append(np.concatenate([fp, compact, xtb_vec], axis=0))
        X_xtb.append(xtb_vec)

    if bde_feature_names and missing_bde_count:
        print(
            f"[INFO] Optional BDE features missing for {missing_bde_count}/{len(xtb)} xTB rows; "
            "training-core medians will impute these values."
        )

    teacher_feature_names = [f"morgan_{i}" for i in range(fp_size)] + list(TWO_D_FEATURE_NAMES) + numeric_cols + engineered_names + bde_feature_names
    xtb_feature_names = numeric_cols + engineered_names + bde_feature_names
    return (
        np.vstack(X_teacher).astype(np.float32),
        np.vstack(X_xtb).astype(np.float32),
        row_to_xtb,
        aligned,
        teacher_feature_names,
        xtb_feature_names,
    )



def impute_fit(X: np.ndarray) -> np.ndarray:
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    return med


def impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(med, np.where(bad)[1])
    return X.astype(np.float32)


def target_transform(name: str):
    eps = 1e-7
    if name == "Sigma2_tot":
        return (
            lambda y: np.log(np.clip(y, eps, None)),
            lambda z: np.exp(z),
            "log",
        )
    if name == "Nu":
        upper = 0.25

        def f(y):
            u = np.clip(y / upper, eps, 1.0 - eps)
            return np.log(u / (1.0 - u))

        def inv(z):
            u = 1.0 / (1.0 + np.exp(-z))
            return upper * u

        return f, inv, "bounded_logit_0p25"
    return lambda y: y, lambda z: z, "identity"


# ==============================================================================
# Teacher model zoo and target-wise OOF ensemble
# ==============================================================================
def make_teacher_zoo(seed: int, n_jobs: int, include_mlp: bool = False) -> List[Tuple[str, Any]]:
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    zoo: List[Tuple[str, Any]] = []
    try:
        from xgboost import XGBRegressor

        zoo.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=650,
                    max_depth=4,
                    learning_rate=0.025,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_lambda=4.0,
                    reg_alpha=0.0,
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=n_jobs,
                    tree_method="hist",
                ),
            )
        )
    except Exception as e:
        print(f"[WARN] XGBoost unavailable, skipped: {e}")

    zoo.extend(
        [
            (
                "ExtraTrees",
                ExtraTreesRegressor(
                    n_estimators=600,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "RandomForest",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=seed,
                    n_jobs=n_jobs,
                ),
            ),
            (
                "HistGBR",
                HistGradientBoostingRegressor(
                    max_iter=600,
                    learning_rate=0.03,
                    l2_regularization=1e-3,
                    random_state=seed,
                ),
            ),
            (
                "Ridge",
                Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=3.0))]),
            ),
        ]
    )

    if include_mlp:
        zoo.append(
            (
                "MLP",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPRegressor(
                                hidden_layer_sizes=(512, 256),
                                activation="relu",
                                alpha=1e-4,
                                learning_rate_init=1e-3,
                                max_iter=500,
                                early_stopping=True,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    return zoo


def fit_predict_oof(proto: Any, X: np.ndarray, y_t: np.ndarray, inv, folds: int, seed: int) -> np.ndarray:
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(X.shape[0], dtype=np.float32)
    for tr, va in kf.split(X):
        model = clone(proto)
        model.fit(X[tr], y_t[tr])
        oof[va] = inv(model.predict(X[va])).astype(np.float32)
    return oof


@dataclass
class TargetTeacherBundle:
    target: str
    transform_name: str
    selected_names: List[str]
    base_models: List[Any]
    meta_model: Optional[Any]
    oof_r2: float
    selection_table: List[Dict[str, Any]]


def fit_targetwise_teacher_ensemble(
    X_core: np.ndarray,
    y_core: np.ndarray,
    X_calib: np.ndarray,
    X_val: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[TargetTeacherBundle], np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Production-clean final_specialist v2 teacher selection.

    This is the retained production path:
    target-wise 2D+xTB OOF teacher selection/stacking.
    No trigger-local v4 branch is used here.
    """
    from sklearn.base import clone
    from sklearn.linear_model import RidgeCV

    zoo = make_teacher_zoo(args.seed, args.teacher_n_jobs, include_mlp=args.include_mlp_teacher)
    if not zoo:
        raise RuntimeError("No teacher models available.")

    n_targets = len(TARGET_PROPS)
    oof_core = np.zeros_like(y_core, dtype=np.float32)
    pred_calib = np.zeros((X_calib.shape[0], n_targets), dtype=np.float32)
    pred_val = np.zeros((X_val.shape[0], n_targets), dtype=np.float32)
    bundles: List[TargetTeacherBundle] = []
    selection_rows: List[Dict[str, Any]] = []

    print("\n" + "=" * 78)
    print("📊 final_specialist v2: target-wise OOF teacher selection/stacking")
    print("=" * 78)

    for j, target in enumerate(TARGET_PROPS):
        fwd, inv, trans_name = target_transform(target)
        y_raw = y_core[:, j].astype(np.float32)
        y_t = fwd(y_raw).astype(np.float32)

        candidate_oof: List[np.ndarray] = []
        candidate_scores: List[float] = []
        candidate_names: List[str] = []
        candidate_protos: List[Any] = []

        for model_idx, (name, proto) in enumerate(zoo):
            try:
                oof = fit_predict_oof(
                    proto,
                    X_core,
                    y_t,
                    inv,
                    folds=args.teacher_cv_folds,
                    seed=args.seed + 101 * j + model_idx,
                )
                score = r2_score_np(y_raw, oof)
                candidate_oof.append(oof)
                candidate_scores.append(score)
                candidate_names.append(name)
                candidate_protos.append(proto)
                selection_rows.append(
                    {
                        "Target": PRETTY_TARGETS.get(target, target),
                        "Candidate_Model": name,
                        "Transform": trans_name,
                        "OOF_R2": score,
                    }
                )
            except Exception as e:
                print(f"[WARN] {target} | {name} failed during OOF selection: {e}")

        if not candidate_oof:
            raise RuntimeError(f"All teacher candidates failed for target {target}")

        order = np.argsort(candidate_scores)[::-1]
        top_k = min(args.teacher_top_k, len(order))
        selected_idx = list(order[:top_k])

        stack_oof = np.column_stack([candidate_oof[k] for k in selected_idx]).astype(np.float32)
        best_single_idx = int(order[0])
        best_single_oof = candidate_oof[best_single_idx]
        best_single_r2 = candidate_scores[best_single_idx]

        meta_model = None
        ensemble_oof = best_single_oof.copy()
        ensemble_r2 = best_single_r2

        if len(selected_idx) >= 2:
            try:
                ridge = RidgeCV(alphas=np.array([0.001, 0.01, 0.1, 1.0, 10.0], dtype=float))
                ridge.fit(stack_oof, y_raw)
                ens = ridge.predict(stack_oof).astype(np.float32)
                ens_r2 = r2_score_np(y_raw, ens)
                if np.isfinite(ens_r2) and ens_r2 >= best_single_r2 - 1e-4:
                    meta_model = ridge
                    ensemble_oof = ens
                    ensemble_r2 = ens_r2
            except Exception as e:
                print(f"[WARN] {target} stacking failed, using best single teacher: {e}")

        if meta_model is None:
            selected_idx = [best_single_idx]

        base_models: List[Any] = []
        calib_base_preds: List[np.ndarray] = []
        val_base_preds: List[np.ndarray] = []

        for k in selected_idx:
            model = clone(candidate_protos[k])
            model.fit(X_core, y_t)
            base_models.append(model)
            calib_base_preds.append(inv(model.predict(X_calib)).astype(np.float32))
            val_base_preds.append(inv(model.predict(X_val)).astype(np.float32))

        calib_stack = np.column_stack(calib_base_preds).astype(np.float32)
        val_stack = np.column_stack(val_base_preds).astype(np.float32)

        if meta_model is None:
            pc = calib_stack[:, 0]
            pv = val_stack[:, 0]
        else:
            pc = meta_model.predict(calib_stack).astype(np.float32)
            pv = meta_model.predict(val_stack).astype(np.float32)

        selected_names = [candidate_names[k] for k in selected_idx]

        oof_core[:, j] = ensemble_oof.astype(np.float32)
        pred_calib[:, j] = pc.astype(np.float32)
        pred_val[:, j] = pv.astype(np.float32)

        bundles.append(
            TargetTeacherBundle(
                target=target,
                transform_name=trans_name,
                selected_names=selected_names,
                base_models=base_models,
                meta_model=meta_model,
                oof_r2=float(ensemble_r2),
                selection_table=[r for r in selection_rows if r["Target"] == PRETTY_TARGETS.get(target, target)],
            )
        )

        print(
            f"{target:30s} | selected={'+'.join(selected_names):48s} | "
            f"transform={trans_name:18s} | OOF_R2={ensemble_r2:.4f}"
        )

    return bundles, oof_core, pred_calib, pred_val, pd.DataFrame(selection_rows)



def split_examples_for_row_level_baseline(args: argparse.Namespace, raw_examples: List[RawExample]) -> Tuple[List[RawExample], List[RawExample], Dict[str, Any]]:
    """
    Return train/validation examples for baseline comparisons.

    Priority:
    1. Reuse train_row_indices / val_row_indices if available.
    2. Reuse final_specialist split as train = train_core + calib, val = val.
    3. Create a new deterministic row-level split and save it.
    """
    if os.path.exists(SPLIT_SAVE_PATH):
        try:
            with open(SPLIT_SAVE_PATH, "r", encoding="utf-8") as f:
                split_ref = json.load(f)

            if "train_row_indices" in split_ref and "val_row_indices" in split_ref:
                train_rows = set(int(x) for x in split_ref.get("train_row_indices", []))
                val_rows = set(int(x) for x in split_ref.get("val_row_indices", []))
            elif "train_core_row_indices" in split_ref and "calib_row_indices" in split_ref and "val_row_indices" in split_ref:
                train_rows = set(int(x) for x in split_ref.get("train_core_row_indices", [])) | set(int(x) for x in split_ref.get("calib_row_indices", []))
                val_rows = set(int(x) for x in split_ref.get("val_row_indices", []))
            else:
                train_rows, val_rows = set(), set()

            if train_rows and val_rows:
                train_examples = [ex for ex in raw_examples if int(ex.row_index) in train_rows]
                val_examples = [ex for ex in raw_examples if int(ex.row_index) in val_rows]
                if train_examples and val_examples:
                    split_info = {
                        "seed": split_ref.get("seed", args.seed),
                        "val_fraction": split_ref.get("val_fraction", args.val_fraction),
                        "train_row_indices": [int(ex.row_index) for ex in train_examples],
                        "val_row_indices": [int(ex.row_index) for ex in val_examples],
                        "train_example_ids": [ex.example_id for ex in train_examples],
                        "val_example_ids": [ex.example_id for ex in val_examples],
                        "targets": TARGET_PROPS,
                        "two_d_features": TWO_D_FEATURE_NAMES,
                        "source": "reused_existing_split",
                    }
                    print(f"[INFO] Reusing existing row-level split: Train={len(train_examples)} | Val={len(val_examples)}")
                    return train_examples, val_examples, split_info
        except Exception as e:
            print(f"[WARN] Could not reuse existing split file, creating a new split: {e}")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(raw_examples))
    rng.shuffle(indices)
    val_size = max(1, int(round(len(raw_examples) * args.val_fraction)))
    val_idx = indices[-val_size:]
    train_idx = indices[:-val_size]
    train_examples = [raw_examples[int(i)] for i in train_idx]
    val_examples = [raw_examples[int(i)] for i in val_idx]

    split_info = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_row_indices": [int(ex.row_index) for ex in train_examples],
        "val_row_indices": [int(ex.row_index) for ex in val_examples],
        "train_example_ids": [ex.example_id for ex in train_examples],
        "val_example_ids": [ex.example_id for ex in val_examples],
        "targets": TARGET_PROPS,
        "two_d_features": TWO_D_FEATURE_NAMES,
        "source": "created_by_03_2d_only_or_standard_baseline",
    }
    with open(SPLIT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Created new row-level split: Train={len(train_examples)} | Val={len(val_examples)}")
    return train_examples, val_examples, split_info


def build_2d_baseline_matrix(examples: List[RawExample], fp_size: int = 2048) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Morgan fingerprint bits + compact energetic descriptors."""
    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    for ex in examples:
        fp = morgan_fp(ex.smiles, fp_size=fp_size).astype(np.float32)
        compact = ex.x2d_raw.detach().cpu().numpy().astype(np.float32)
        X_blocks.append(np.concatenate([fp, compact], axis=0))
        y_blocks.append(ex.y_raw.detach().cpu().numpy().astype(np.float32))
    feature_names = [f"morgan_{i}" for i in range(fp_size)] + list(TWO_D_FEATURE_NAMES)
    return np.vstack(X_blocks).astype(np.float32), np.vstack(y_blocks).astype(np.float32), feature_names


def make_2d_only_models(seed: int, n_jobs: int, include_mlp: bool = True) -> List[Tuple[str, Any]]:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models: List[Tuple[str, Any]] = []
    models.append(
        (
            "RF",
            RandomForestRegressor(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=seed,
                n_jobs=n_jobs,
            ),
        )
    )

    try:
        from xgboost import XGBRegressor
        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=900,
                    max_depth=4,
                    learning_rate=0.025,
                    subsample=0.90,
                    colsample_bytree=0.80,
                    reg_lambda=4.0,
                    reg_alpha=0.0,
                    objective="reg:squarederror",
                    random_state=seed,
                    n_jobs=n_jobs,
                    tree_method="hist",
                ),
            )
        )
    except Exception as e:
        print(f"[WARN] XGBoost unavailable, skipped in 2D-only baseline: {e}")

    if include_mlp:
        models.append(
            (
                "MLP",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "mlp",
                            MLPRegressor(
                                hidden_layer_sizes=(512, 256),
                                activation="relu",
                                alpha=1e-4,
                                learning_rate_init=1e-3,
                                max_iter=500,
                                early_stopping=True,
                                random_state=seed,
                            ),
                        ),
                    ]
                ),
            )
        )
    return models


def update_baseline_summary_table(extra_metrics: Optional[pd.DataFrame] = None) -> None:
    """Collect available baseline metrics into one paper-ready summary table."""
    paths: List[str] = []
    frames: List[pd.DataFrame] = []
    for fp in paths:
        if os.path.exists(fp):
            try:
                df = pd.read_csv(fp)
                if len(df):
                    frames.append(df)
            except Exception as e:
                print(f"[WARN] Could not read metrics file {fp}: {e}")
    if extra_metrics is not None and len(extra_metrics):
        frames.append(extra_metrics)
    if not frames:
        return
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["Model_Group", "Model", "Target"], keep="last")

    target_order = {PRETTY_TARGETS.get(t, t): i for i, t in enumerate(TARGET_PROPS)}
    model_order = {
        "RF": 0,
        "XGBoost": 1,
        "MLP": 2,
        "EGNN": 3,
        "target-wise specialist teacher v2": 4,
        "2D+xTB OOF teacher ensemble + xTB-aware EGNN residual": 5,
    }
    out["_target_order"] = out["Target"].map(target_order).fillna(999)
    out["_model_order"] = out["Model"].map(model_order).fillna(999)
    out = out.sort_values(["_model_order", "Model_Group", "_target_order"]).drop(columns=["_target_order", "_model_order"])

    os.makedirs("../results/baselines", exist_ok=True)
    os.makedirs("../results/tables", exist_ok=True)
    out.to_csv("../results/baselines/baseline_summary_9d.csv", index=False)
    out.to_csv("../results/tables/Table_Model_Baselines.csv", index=False)
    print("[OK] Updated baseline summary: ../results/baselines/baseline_summary_9d.csv")
    print("[OK] Updated paper table: ../results/tables/Table_Model_Baselines.csv")


def run_2d_only_baseline(args: argparse.Namespace, raw_examples: List[RawExample]) -> None:
    """Train RF / XGBoost / MLP 2D-only baselines under the shared row-level split."""
    print("=" * 78)
    print("📊 启动 2D-only baseline: RF / XGBoost / MLP")
    print("=" * 78)

    train_examples, val_examples, split_info = split_examples_for_row_level_baseline(args, raw_examples)
    X_train, y_train, feature_names = build_2d_baseline_matrix(train_examples, fp_size=args.fp_size)
    X_val, y_val, _ = build_2d_baseline_matrix(val_examples, fp_size=args.fp_size)

    print(f"[INFO] Split rows: Train_N={len(train_examples)} | Val_N={len(val_examples)} | Features={X_train.shape[1]}")
    print("[INFO] Input: Morgan fingerprint + compact energetic descriptors")

    all_metrics: List[pd.DataFrame] = []
    model_records: Dict[str, Any] = {}
    models = make_2d_only_models(args.seed, args.teacher_n_jobs, include_mlp=not args.skip_2d_mlp)

    for model_name, proto in models:
        print(f"\nTraining 2D-only {model_name} target-wise...")
        preds = np.zeros_like(y_val, dtype=np.float32)
        target_models: List[Any] = []
        for j, target in enumerate(TARGET_PROPS):
            model = copy.deepcopy(proto)
            model.fit(X_train, y_train[:, j])
            preds[:, j] = model.predict(X_val).astype(np.float32)
            target_models.append(model)
        metrics = metrics_table(
            y_val,
            preds,
            model_group="2D-only",
            model_name=model_name,
            train_n=len(train_examples),
            val_n=len(val_examples),
            split_id="row-level train_val_split_9d",
            input_features="Morgan fingerprint + compact energetic descriptors",
        )
        print(metrics[["Target", "MAE", "RMSE", "R2"]].to_string(index=False))
        all_metrics.append(metrics)
        model_records[model_name] = target_models

    out = pd.concat(all_metrics, ignore_index=True)
    os.makedirs("../results/baselines", exist_ok=True)
    os.makedirs("../results/tables", exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"baseline_2d_only_metrics_{args.run_tag}.csv")
    out.to_csv(metrics_path, index=False)

    # Save reusable model bundle for inspection/reproducibility. It is not used by 04 inference.
    try:
        import joblib
        bundle = {
            "model_type": "2D-only baselines",
            "models": model_records,
            "feature_names": feature_names,
            "fp_size": args.fp_size,
            "targets": TARGET_PROPS,
            "compact_2d_features": TWO_D_FEATURE_NAMES,
            "split_info": split_info,
            "density_label_note": "Density_calc(g/cm3) uses proxy density unless --density_label_mode crystal_preferred finds true/calibrated crystal-density labels.",
        "density_label_mode": getattr(args, "density_label_mode", "proxy_only"),
        "target_profile": getattr(args, "target_profile", "balanced"),
        }
        models_path = os.path.join(args.output_dir, f"baseline_2d_only_models_{args.run_tag}.joblib")
        joblib.dump(bundle, models_path)
        print(f"[OK] Saved 2D-only model bundle: {models_path}")
    except Exception as e:
        print(f"[WARN] Could not save 2D-only model bundle: {e}")

    update_baseline_summary_table(out)
    print("=" * 78)
    print(f"✅ 2D-only baseline metrics saved: {metrics_path}")
    print("=" * 78)

# ==============================================================================
# 3D-only baseline training
# ==============================================================================
def train_standard_egnn(args: argparse.Namespace, raw_examples: List[RawExample], device: torch.device) -> None:
    """Train the retained 3D-only EGNN manuscript baseline.

    2D-only and 3D-only are retained only as manuscript baselines; final_specialist is the production model family.
    """
    if args.model_variant != "3d_only":
        raise ValueError("train_standard_egnn is only used for the retained 3d_only baseline.")

    train_examples, val_examples, _ = split_examples_for_row_level_baseline(args, raw_examples)

    train_y = torch.stack([ex.y_raw for ex in train_examples], dim=0)
    train_x2d = torch.stack([ex.x2d_raw for ex in train_examples], dim=0)
    y_mean, y_std = train_y.mean(dim=0), safe_std(train_y)
    x2d_mean, x2d_std = train_x2d.mean(dim=0), safe_std(train_x2d)

    train_data = make_data_from_examples(train_examples, y_mean, y_std, x2d_mean, x2d_std)
    val_data = make_data_from_examples(val_examples, y_mean, y_std, x2d_mean, x2d_std)

    print(f"[INFO] 有效样本: {len(raw_examples)} | Train: {len(train_data)} | Val: {len(val_data)}")
    print(
        f"[INFO] 训练期坐标扰动: std={0.0 if args.no_coord_aug else args.coord_noise_std:.3f} Å, "
        f"prob={0.0 if args.no_coord_aug else args.coord_noise_prob:.2f}；生产模型建议保持 0，扰动用于 ablation。"
    )

    train_loader = DataLoader(EnergeticDataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EnergeticDataset(val_data), batch_size=args.batch_size, shuffle=False)

    model = True3DOnlyEGNN(hidden_dim=args.hidden_dim, num_targets=len(TARGET_PROPS)).to(device)
    model_save_path = f"../results/baselines/best_3d_only_egnn_seed{args.seed}_{args.loss_weight_mode}.pth"
    model_class_name = "True3DOnlyEGNN_Native"
    metrics_path = "../results/baselines/baseline_3d_egnn_only_metrics.csv"
    input_features = "atom type + coordinates + radius graph"
    model_group = "3D-only"
    model_name = "EGNN"

    print(f"[INFO] Model class: {model_class_name}")
    print("[INFO] 2D features are loaded for normalization compatibility but are not used by the 3D-only baseline.")
    print(f"[INFO] Model checkpoint path: {model_save_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    best_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0
    coord_noise_std = 0.0 if args.no_coord_aug else args.coord_noise_std
    coord_noise_prob = 0.0 if args.no_coord_aug else args.coord_noise_prob

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            noisy_pos = apply_coordinate_noise(batch.pos, batch.batch, coord_noise_std, coord_noise_prob)
            out = model(batch.z, noisy_pos, batch.batch)
            loss = compute_multitask_huber_loss(out, batch.y, args.loss_weight_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_graphs += batch.num_graphs
        train_loss /= max(train_graphs, 1)

        model.eval()
        val_loss = 0.0
        val_graphs = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.z, batch.pos, batch.batch)
                loss = compute_multitask_huber_loss(out, batch.y, args.loss_weight_mode)
                val_loss += loss.item() * batch.num_graphs
                val_graphs += batch.num_graphs
        val_loss /= max(val_graphs, 1)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Train Huber: {train_loss:.4f} | Val Huber: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            print(f"[INFO] Early stopping at epoch {epoch}; best epoch = {best_epoch}, best val Huber = {best_loss:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint state.")

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "means": y_mean.cpu().numpy(),
        "stds": y_std.cpu().numpy(),
        "targets": TARGET_PROPS,
        "model_class": model_class_name,
        "model_variant": args.model_variant,
        "hidden_dim": args.hidden_dim,
        "coord_noise_std": coord_noise_std,
        "coord_noise_prob": coord_noise_prob,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_huber": best_loss,
        "density_label_note": "Density_calc(g/cm3) is molecular-volume-derived proxy density unless crystal/calibrated density is explicitly supplied.",
    }
    torch.save(checkpoint, model_save_path)

    y_val = torch.stack([ex.y_raw for ex in val_examples], dim=0).numpy().astype(np.float32)
    model.eval()
    blocks = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch)
            raw = out.cpu() * y_std.cpu() + y_mean.cpu()
            blocks.append(raw.numpy())
    y_pred = np.vstack(blocks).astype(np.float32)

    metrics = metrics_table(
        y_val,
        y_pred,
        model_group,
        model_name,
        len(train_examples),
        len(val_examples),
        "row-level train_val_split_clean",
        input_features,
    )
    os.makedirs("../results/baselines", exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    split_info = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "train_example_ids": [ex.example_id for ex in train_examples],
        "val_example_ids": [ex.example_id for ex in val_examples],
        "train_row_indices": [int(ex.row_index) for ex in train_examples],
        "val_row_indices": [int(ex.row_index) for ex in val_examples],
        "train_molecules": [ex.molecule for ex in train_examples],
        "val_molecules": [ex.molecule for ex in val_examples],
        "targets": TARGET_PROPS,
    }
    with open(SPLIT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"📊 3D-only baseline metrics saved: {metrics_path}")
    print(metrics[["Target", "MAE", "RMSE", "R2"]].to_string(index=False))
    print("=" * 78)
    print(f"✅ 3D-only baseline training completed. Best epoch = {best_epoch}, Best val Huber = {best_loss:.4f}")
    print(f"📦 Model saved to: {model_save_path}")
    print(f"📄 Split saved to: {SPLIT_SAVE_PATH}")
    print("=" * 78)


# ==============================================================================
# Final-Specialist-Hybrid v2
# ==============================================================================
def make_residual_examples(examples: List[RawExample], residual_raw: np.ndarray) -> List[RawExample]:
    return [replace(ex, y_raw=torch.tensor(r, dtype=torch.float32)) for ex, r in zip(examples, residual_raw)]


def select_split_indices(n: int, seed: int, val_fraction: float, calib_fraction: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(1, int(round(n * val_fraction)))
    val_idx = indices[-val_size:]
    train_all_idx = indices[:-val_size]
    calib_size = max(1, int(round(len(train_all_idx) * calib_fraction)))
    calib_idx = train_all_idx[-calib_size:]
    core_idx = train_all_idx[:-calib_size]
    return core_idx, calib_idx, val_idx


def fit_alpha_grid(
    y_calib: np.ndarray,
    teacher_calib: np.ndarray,
    residual_calib: np.ndarray,
    alpha_max: float,
    alpha_step: float,
    min_improvement: float,
    residual_policy: str = "conservative",
    residual_min_corr: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    alpha_grid = np.arange(0.0, alpha_max + alpha_step * 0.5, alpha_step, dtype=np.float32)
    alphas = np.zeros((y_calib.shape[1],), dtype=np.float32)
    rows: List[Dict[str, Any]] = []

    for j, target in enumerate(TARGET_PROPS):
        base_err = y_calib[:, j] - teacher_calib[:, j]
        base_mse = float(np.mean(base_err ** 2))
        residual_vec = residual_calib[:, j]
        residual_corr = _safe_corr_np(base_err, residual_vec)
        best_a = 0.0
        best_mse = base_mse
        for a in alpha_grid:
            pred = teacher_calib[:, j] + float(a) * residual_vec
            mse = float(np.mean((y_calib[:, j] - pred) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_a = float(a)

        improvement_frac = (base_mse - best_mse) / max(base_mse, 1e-12)
        gate_reason = "accepted"
        if residual_policy == "teacher_only":
            alphas[j] = 0.0
            gate_reason = "teacher_only_policy"
        elif residual_policy == "conservative" and (
            not np.isfinite(residual_corr) or residual_corr < residual_min_corr
        ):
            alphas[j] = 0.0
            gate_reason = f"residual_corr_below_{residual_min_corr:g}"
        elif best_mse < base_mse * (1.0 - min_improvement):
            alphas[j] = best_a
        else:
            alphas[j] = 0.0
            gate_reason = f"calib_improvement_below_{min_improvement:g}"

        rows.append(
            {
                "Target": PRETTY_TARGETS.get(target, target),
                "Alpha": float(alphas[j]),
                "Residual_Policy": residual_policy,
                "Residual_Corr_Calib": float(residual_corr) if np.isfinite(residual_corr) else np.nan,
                "Base_Calib_RMSE": float(np.sqrt(base_mse)),
                "Best_Gated_Calib_RMSE": float(np.sqrt(best_mse)),
                "Calib_MSE_Improvement_Fraction": float(improvement_frac),
                "Gate_Decision": gate_reason,
            }
        )
        print(
            f"[ALPHA] {target:30s} alpha={alphas[j]:.3f} | "
            f"corr={residual_corr:.3f} | base_calib_RMSE={np.sqrt(base_mse):.5f} | "
            f"gated_calib_RMSE={np.sqrt(best_mse):.5f} | gate={gate_reason}"
        )
    return alphas, pd.DataFrame(rows)


def run_final_specialist_v2(args: argparse.Namespace, raw_examples: List[RawExample], device: torch.device) -> None:
    import joblib

    print("=" * 78)
    print("🚀 启动最终模型: final_specialist v2 = 2D+xTB OOF teacher ensemble + xTB-aware EGNN residual")
    print("=" * 78)

    X_teacher_all, X_xtb_all, row_to_xtb, aligned_examples, teacher_feature_names, xtb_feature_names = build_final_feature_matrices(
        raw_examples,
        xtb_feature_path=args.xtb_feature_path,
        fp_size=args.fp_size,
        bde_feature_path=args.bde_feature_path,
        enable_bde_features=args.enable_bde_features,
    )
    y_all = torch.stack([ex.y_raw for ex in aligned_examples], dim=0).numpy().astype(np.float32)

    print(f"[INFO] Aligned examples with xTB features: {len(aligned_examples)} / {len(raw_examples)}")
    print(f"[INFO] Teacher feature dimension: {X_teacher_all.shape[1]}")
    print(f"[INFO] Residual xTB feature dimension: {X_xtb_all.shape[1]}")
    print(f"[INFO] xTB feature path: {args.xtb_feature_path}")


    core_idx, calib_idx, val_idx = select_split_indices(len(aligned_examples), args.seed, args.val_fraction, args.specialist_calib_fraction)

    core_examples = [aligned_examples[i] for i in core_idx]
    calib_examples = [aligned_examples[i] for i in calib_idx]
    val_examples = [aligned_examples[i] for i in val_idx]

    X_core_raw, X_calib_raw, X_val_raw = X_teacher_all[core_idx], X_teacher_all[calib_idx], X_teacher_all[val_idx]
    X_xtb_core_raw, X_xtb_calib_raw, X_xtb_val_raw = X_xtb_all[core_idx], X_xtb_all[calib_idx], X_xtb_all[val_idx]
    y_core, y_calib, y_val = y_all[core_idx], y_all[calib_idx], y_all[val_idx]

    teacher_median = impute_fit(X_core_raw)
    X_core = impute_apply(X_core_raw, teacher_median)
    X_calib = impute_apply(X_calib_raw, teacher_median)
    X_val = impute_apply(X_val_raw, teacher_median)


    xtb_median = impute_fit(X_xtb_core_raw)
    X_xtb_core = impute_apply(X_xtb_core_raw, xtb_median)
    X_xtb_calib = impute_apply(X_xtb_calib_raw, xtb_median)
    X_xtb_val = impute_apply(X_xtb_val_raw, xtb_median)

    xtb_map: Dict[int, np.ndarray] = {}
    for ex, vec in zip(core_examples, X_xtb_core):
        xtb_map[int(ex.row_index)] = vec.astype(np.float32)
    for ex, vec in zip(calib_examples, X_xtb_calib):
        xtb_map[int(ex.row_index)] = vec.astype(np.float32)
    for ex, vec in zip(val_examples, X_xtb_val):
        xtb_map[int(ex.row_index)] = vec.astype(np.float32)

    print(f"[INFO] Final split | Train_core={len(core_idx)} | Calib={len(calib_idx)} | Val={len(val_idx)}")

    teacher_bundles, teacher_oof_core, teacher_calib, teacher_val, selection_table = fit_targetwise_teacher_ensemble(
        X_core,
        y_core,
        X_calib,
        X_val,
        args,
    )

    teacher_metrics = metrics_table(
        y_val,
        teacher_val,
        model_group="2D+xTB-teacher-v2",
        model_name="target-wise OOF specialist ensemble",
        train_n=len(core_examples),
        val_n=len(val_examples),
        split_id="final_specialist_v2 train/calibration/validation split",
        input_features="Morgan fingerprint + compact descriptors + xTB descriptors + engineered xTB interactions",
    )

    print("\n📊 2D+xTB teacher validation metrics:")
    print(teacher_metrics[["Target", "MAE", "RMSE", "R2"]].to_string(index=False))

    # Residual targets based on honest teacher OOF for core.
    resid_core = y_core - teacher_oof_core
    resid_calib_true = y_calib - teacher_calib
    resid_val_true = y_val - teacher_val

    residual_core_examples = make_residual_examples(core_examples, resid_core)
    residual_calib_examples = make_residual_examples(calib_examples, resid_calib_true)
    residual_val_examples = make_residual_examples(val_examples, resid_val_true)

    resid_train_y = torch.stack([ex.y_raw for ex in residual_core_examples], dim=0)
    resid_mean, resid_std = resid_train_y.mean(dim=0), safe_std(resid_train_y)

    train_x2d = torch.stack([ex.x2d_raw for ex in residual_core_examples], dim=0)
    x2d_mean, x2d_std = train_x2d.mean(dim=0), safe_std(train_x2d)

    xtb_train_tensor = torch.tensor(np.vstack([xtb_map[int(ex.row_index)] for ex in residual_core_examples]), dtype=torch.float)
    xtb_mean, xtb_std = xtb_train_tensor.mean(dim=0), safe_std(xtb_train_tensor)

    train_data = make_xtb_data_from_examples(residual_core_examples, resid_mean, resid_std, x2d_mean, x2d_std, xtb_map, xtb_mean, xtb_std)
    calib_data = make_xtb_data_from_examples(residual_calib_examples, resid_mean, resid_std, x2d_mean, x2d_std, xtb_map, xtb_mean, xtb_std)
    val_data = make_xtb_data_from_examples(residual_val_examples, resid_mean, resid_std, x2d_mean, x2d_std, xtb_map, xtb_mean, xtb_std)

    train_loader = DataLoader(EnergeticDataset(train_data), batch_size=args.batch_size, shuffle=True)
    calib_loader = DataLoader(EnergeticDataset(calib_data), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(EnergeticDataset(val_data), batch_size=args.batch_size, shuffle=False)

    residual_model = XTBResidualEGNN(
        hidden_dim=args.hidden_dim,
        num_targets=len(TARGET_PROPS),
        num_2d=len(TWO_D_FEATURE_NAMES),
        num_xtb=len(xtb_feature_names),
    ).to(device)

    optimizer = torch.optim.AdamW(residual_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    coord_noise_std = 0.0 if args.no_coord_aug else args.coord_noise_std
    coord_noise_prob = 0.0 if args.no_coord_aug else args.coord_noise_prob

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print("\n" + "=" * 78)
    print("🧠 Training xTB-aware EGNN residual branch")
    print("=" * 78)

    for epoch in range(1, args.epochs + 1):
        residual_model.train()
        train_loss = 0.0
        train_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            noisy_pos = apply_coordinate_noise(batch.pos, batch.batch, coord_noise_std, coord_noise_prob)
            out = residual_model(batch.z, noisy_pos, batch.batch, batch.x_2d, batch.x_xtb)
            loss = compute_multitask_huber_loss(out, batch.y, args.loss_weight_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            train_graphs += batch.num_graphs
        train_loss /= max(train_graphs, 1)

        residual_model.eval()
        calib_loss = 0.0
        calib_graphs = 0
        with torch.no_grad():
            for batch in calib_loader:
                batch = batch.to(device)
                out = residual_model(batch.z, batch.pos, batch.batch, batch.x_2d, batch.x_xtb)
                loss = compute_multitask_huber_loss(out, batch.y, args.loss_weight_mode)
                calib_loss += loss.item() * batch.num_graphs
                calib_graphs += batch.num_graphs
        calib_loss /= max(calib_graphs, 1)
        scheduler.step(calib_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | Residual Train Huber: {train_loss:.4f} | Residual Calib Huber: {calib_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if calib_loss < best_loss - args.min_delta:
            best_loss = calib_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in residual_model.state_dict().items()}
        else:
            bad_epochs += 1

        if args.early_stopping_patience > 0 and bad_epochs >= args.early_stopping_patience:
            print(f"[INFO] Residual early stopping at epoch {epoch}; best epoch = {best_epoch}, best calib Huber = {best_loss:.4f}")
            break

    if best_state is None:
        raise RuntimeError("Residual EGNN did not produce a valid checkpoint state.")

    residual_model.load_state_dict(best_state)
    r_calib_pred = predict_xtb_residual_raw(residual_model, calib_loader, device, resid_mean, resid_std)
    r_val_pred = predict_xtb_residual_raw(residual_model, val_loader, device, resid_mean, resid_std)

    alpha, alpha_report = fit_alpha_grid(
        y_calib=y_calib,
        teacher_calib=teacher_calib,
        residual_calib=r_calib_pred,
        alpha_max=args.alpha_max,
        alpha_step=args.alpha_step,
        min_improvement=args.alpha_min_improvement,
        residual_policy=args.residual_policy,
        residual_min_corr=args.residual_min_corr,
    )

    final_val = teacher_val + r_val_pred * alpha.reshape(1, -1)

    final_metrics = metrics_table(
        y_val,
        final_val,
        model_group="Final-Specialist-Hybrid-v2",
        model_name="2D+xTB OOF teacher ensemble + xTB-aware EGNN residual",
        train_n=len(core_examples),
        val_n=len(val_examples),
        split_id="final_specialist_v2 train/calibration/validation split",
        input_features="Morgan fingerprint + compact descriptors + xTB descriptors + engineered xTB interactions + 3D EGNN residual correction",
    )

    comparison = pd.concat([teacher_metrics, final_metrics], ignore_index=True)
    os.makedirs("../results/baselines", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    metrics_path = os.path.join(args.output_dir, f"final_specialist_metrics_{args.run_tag}.csv")
    final_metrics_path = os.path.join(args.output_dir, f"final_specialist_final_metrics_{args.run_tag}.csv")
    teacher_selection_path = os.path.join(args.output_dir, f"final_specialist_teacher_selection_{args.run_tag}.csv")
    comparison.to_csv(metrics_path, index=False)
    comparison.to_csv(final_metrics_path, index=False)
    selection_table.to_csv(teacher_selection_path, index=False)

    print("\n📊 Final specialist v2 validation metrics:")
    print(final_metrics[["Target", "MAE", "RMSE", "R2"]].to_string(index=False))

    print("\n📊 Final specialist v2 improvement over 2D+xTB teacher-v2:")
    teacher_r2 = teacher_metrics.set_index("Target")["R2"]
    final_r2 = final_metrics.set_index("Target")["R2"]
    delta = (final_r2 - teacher_r2).rename("Delta_R2").reset_index()
    print(delta.to_string(index=False))

    npj_evidence_paths = export_npj_evidence_bundle(
        args=args,
        y_val=y_val,
        teacher_val=teacher_val,
        final_val=final_val,
        val_examples=val_examples,
        teacher_metrics=teacher_metrics,
        final_metrics=final_metrics,
        teacher_bundles=teacher_bundles,
        alpha_report=alpha_report,
        teacher_feature_names=teacher_feature_names,
        xtb_feature_names=xtb_feature_names,
    )

    run_tag = getattr(args, "run_tag", f"{args.model_variant}_seed{args.seed}")
    if getattr(args, "sensitivity_v3", False):
        sensitivity_classification_auxiliary_report(
            y_core=y_core,
            y_val=y_val,
            y_pred_val=final_val,
            val_examples=val_examples,
            output_dir=args.sensitivity_output_dir,
            run_tag=run_tag,
        )
        sensitivity_active_learning_report(
            y_core=y_core,
            y_val=y_val,
            teacher_val=teacher_val,
            final_val=final_val,
            val_examples=val_examples,
            output_dir=args.sensitivity_output_dir,
            run_tag=run_tag,
            top_k=args.sensitivity_al_top_k,
        )

    split_info = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "specialist_calib_fraction": args.specialist_calib_fraction,
        "train_core_row_indices": [int(ex.row_index) for ex in core_examples],
        "calib_row_indices": [int(ex.row_index) for ex in calib_examples],
        "val_row_indices": [int(ex.row_index) for ex in val_examples],
        "train_core_example_ids": [ex.example_id for ex in core_examples],
        "calib_example_ids": [ex.example_id for ex in calib_examples],
        "val_example_ids": [ex.example_id for ex in val_examples],
        "targets": TARGET_PROPS,
        "compact_2d_features": TWO_D_FEATURE_NAMES,
        "teacher_feature_names": teacher_feature_names,
        "xtb_feature_names": xtb_feature_names,
        "teacher_selected_names": [b.selected_names for b in teacher_bundles],
        "teacher_oof_r2": [b.oof_r2 for b in teacher_bundles],
        "alpha": alpha.tolist(),
        "alpha_report": alpha_report.to_dict(orient="records"),
        "residual_policy": args.residual_policy,
        "residual_min_corr": args.residual_min_corr,
        "xtb_feature_path": args.xtb_feature_path,
        "bde_features_enabled": bool(args.enable_bde_features),
        "bde_feature_path": args.bde_feature_path if args.enable_bde_features else None,
        "npj_evidence_paths": npj_evidence_paths,
    }

    final_bundle = {
        "model_type": "Final-Specialist-Hybrid-v2",
        "teacher_bundles": teacher_bundles,
        "teacher_feature_names": teacher_feature_names,
        "teacher_feature_median": teacher_median,
        "xtb_feature_names": xtb_feature_names,
        "xtb_feature_median": xtb_median,
        "residual_model_class": "XTBResidualEGNN_Native",
        "residual_state_dict": best_state,
        "hidden_dim": args.hidden_dim,
        "resid_mean": resid_mean.cpu().numpy(),
        "resid_std": resid_std.cpu().numpy(),
        "x2d_mean": x2d_mean.cpu().numpy(),
        "x2d_std": x2d_std.cpu().numpy(),
        "xtb_mean": xtb_mean.cpu().numpy(),
        "xtb_std": xtb_std.cpu().numpy(),
        "alpha": alpha,
        "targets": TARGET_PROPS,
        "compact_2d_features": TWO_D_FEATURE_NAMES,
        "split_info": split_info,
        "metrics_path": metrics_path,
        "npj_evidence_paths": npj_evidence_paths,
        "bde_features_enabled": bool(args.enable_bde_features),
        "bde_feature_path": args.bde_feature_path if args.enable_bde_features else None,
        "residual_policy": args.residual_policy,
        "density_label_note": "Density_calc(g/cm3) is molecular-volume-derived proxy density unless crystal density is explicitly supplied.",
    }

    import joblib

    joblib.dump(final_bundle, args.final_model_path)
    with open(SPLIT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print(f"✅ Final specialist v2 model saved: {args.final_model_path}")
    print(f"📄 Final specialist v2 metrics saved: {metrics_path}")
    print(f"📄 Canonical final metrics saved: {final_metrics_path}")
    print(f"📄 Teacher selection table saved: {teacher_selection_path}")
    print(f"📄 Final split saved: {SPLIT_SAVE_PATH}")
    print(f"📌 Residual best epoch = {best_epoch}, best calibration Huber = {best_loss:.4f}")
    print("=" * 78)


# ==============================================================================
# Main
# ==============================================================================
def load_training_dataframe() -> pd.DataFrame:
    if not os.path.exists(OLD_CSV_PATH):
        raise FileNotFoundError(f"找不到训练数据库: {OLD_CSV_PATH}")

    header = pd.read_csv(OLD_CSV_PATH, nrows=1)
    missing_cols = [c for c in TARGET_PROPS + ["SMILES"] if c not in header.columns]
    if missing_cols:
        raise ValueError(f"训练数据库缺失必要列: {missing_cols}")

    df_all = pd.read_csv(OLD_CSV_PATH).dropna(subset=TARGET_PROPS + ["SMILES"]).copy()
    df_all = df_all[
        (df_all["Heat_of_Formation(kcal/mol)"] > -2000)
        & (df_all["Heat_of_Formation(kcal/mol)"] < 3000)
        & (df_all["Density_calc(g/cm3)"] > 0.5)
        & (df_all["Density_calc(g/cm3)"] < 3.5)
        & (df_all["Vertical_BDE(kcal/mol)"] > 0.0)
        & (df_all["Vertical_BDE(kcal/mol)"] < 250.0)
    ].reset_index(drop=True)
    return df_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train 10-target HELS Hybrid Native-EGNN models with vertical BDE labels.")
    parser.add_argument("--model_variant", choices=["2d_only", "3d_only", "final_specialist"], default="final_specialist")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss_weight_mode", choices=["uniform", "focus_electronic", "no_mw"], default="uniform")
    parser.add_argument("--early_stopping_patience", type=int, default=25)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_profile", choices=["balanced", "sensitivity", "density"], default="balanced")
    parser.add_argument("--density_label_mode", choices=["proxy_only", "crystal_preferred"], default="proxy_only")
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--coord_noise_std", type=float, default=0.0)
    parser.add_argument("--coord_noise_prob", type=float, default=0.0)
    parser.add_argument("--no_coord_aug", action="store_true")

    # final_specialist v2 arguments
    parser.add_argument("--xtb_feature_path", default="../data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv")
    parser.add_argument("--specialist_calib_fraction", type=float, default=0.10)
    parser.add_argument("--final_model_path", default=None)
    parser.add_argument("--output_dir", default="../results/clean_training")
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--teacher_n_jobs", type=int, default=16)
    parser.add_argument("--skip_2d_mlp", action="store_true", help="Skip the MLP model in 2d_only baseline mode for faster testing.")
    parser.add_argument("--teacher_cv_folds", type=int, default=5)
    parser.add_argument("--teacher_top_k", type=int, default=3)
    parser.add_argument("--include_mlp_teacher", action="store_true")
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--alpha_max", type=float, default=0.75)
    parser.add_argument("--alpha_step", type=float, default=0.025)
    parser.add_argument("--alpha_min_improvement", type=float, default=0.005)
    parser.add_argument("--residual_policy", choices=["conservative", "calib_gate", "teacher_only"], default="conservative")
    parser.add_argument("--residual_min_corr", type=float, default=0.05)
    parser.add_argument("--enable_bde_features", action="store_true", help="Append optional weak-bond manifest descriptors to xTB features; completed BDE label values are excluded to prevent leakage.")
    parser.add_argument("--bde_feature_path", default="../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
    parser.add_argument("--export_npj_evidence", dest="export_npj_evidence", action="store_true", default=True)
    parser.add_argument("--no_export_npj_evidence", dest="export_npj_evidence", action="store_false")
    parser.add_argument("--npj_evidence_dir", default="../manuscript_npJ/SI/model_diagnostics")
    parser.add_argument("--sensitivity_v3", action="store_true", help="Enable sensitivity_v3 reports: classification auxiliary analysis and AL uncertainty table.")
    parser.add_argument("--sensitivity_al_top_k", type=int, default=80)
    parser.add_argument("--sensitivity_output_dir", default="../results/sensitivity_v3")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args = apply_target_profile_defaults(args)

    # Clean-default runtime output paths
    if args.run_tag is None:
        args.run_tag = f"{args.model_variant}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    global SPLIT_SAVE_PATH, MODEL_SAVE_PATH
    SPLIT_SAVE_PATH = os.path.join(args.output_dir, f"split_{args.run_tag}.json")
    MODEL_SAVE_PATH = os.path.join(args.output_dir, f"model_{args.run_tag}.pth")
    if args.final_model_path is None:
        args.final_model_path = os.path.join(args.output_dir, f"final_specialist_{args.run_tag}.joblib")

    print("=" * 50)
    print("🚀 启动 HELS 10-target training engine")
    print("=" * 50)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model variant: {args.model_variant}")
    print(f"[INFO] Residual policy: {args.residual_policy} | residual_min_corr={args.residual_min_corr}")
    print(f"[INFO] Optional BDE features: {bool(args.enable_bde_features)} | path={args.bde_feature_path}")
    print(f"[INFO] Export npj evidence: {bool(args.export_npj_evidence)} | dir={args.npj_evidence_dir}")

    df_all = load_training_dataframe()
    df_all = apply_density_label_policy(df_all, args)
    raw_examples = build_raw_examples(df_all)
    if len(raw_examples) < 20:
        raise RuntimeError(f"有效 3D 样本过少: {len(raw_examples)}。请检查 old_dataset.csv 与 raw_2100_xyz。")

    if args.model_variant == "2d_only":
        run_2d_only_baseline(args, raw_examples)
    elif args.model_variant == "final_specialist":
        run_final_specialist_v2(args, raw_examples, device)
    elif args.model_variant == "3d_only":
        train_standard_egnn(args, raw_examples, device)
    else:
        raise ValueError(f"Unsupported model_variant: {args.model_variant}")


if __name__ == "__main__":
    main()
