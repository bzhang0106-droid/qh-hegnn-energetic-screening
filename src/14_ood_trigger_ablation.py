#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
14_ood_trigger_ablation.py

Purpose
-------
Model optimization diagnostic before expensive oracle augmentation.

This script evaluates whether trigger-linkage local descriptors improve
target-wise prediction under random, scaffold, and Butina OOD splits.

Feature sets:
  Base:
    RDKit global/sensitivity descriptors + sanitized xTB descriptors

  Base_plus_TriggerLocal:
    Base + rule-based trigger-linkage local descriptors

Outputs:
  results/model_optimization/ood_trigger_ablation_metrics.csv
  results/model_optimization/ood_trigger_ablation_delta.csv
  results/model_optimization/ood_trigger_ablation_summary.csv

Interpretation:
  If Base_plus_TriggerLocal improves OOD R2 for sensitivity-related targets,
  these descriptors should be formally integrated into final_specialist_v4.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "model_optimization"
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = [
    ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv",
    ROOT / "results" / "final_model_release" / "clean_sanitized_v1" / "training_dataset_clean_v1.csv",
]

TARGET_FILES = [
    ROOT / "data" / "baselines" / "target_matrix_10d.csv",
    ROOT / "data" / "curated_molecule_clean_v1" / "target_matrix_10d_molecule_clean.csv",
    ROOT / "data" / "curated_molecule_clean_v1" / "old_dataset_molecule_clean.csv",
    ROOT / "data" / "curated_molecule_clean_v1" / "target_matrix_9targets_molecule_clean.csv",
    ROOT / "results" / "final_model_release" / "clean_sanitized_v1" / "training_dataset_clean_v1.csv",
]

XTB_FILE = ROOT / "data" / "curated_molecule_clean_v1" / "xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv"
SPLIT_FILE = ROOT / "results" / "model_optimization" / "scaffold_ood_split_assignments_10d.csv"
TRIGGER_FILE = ROOT / "results" / "model_optimization" / "trigger_linkage_local_descriptors_10d.csv"

TARGETS = [
    "Density_calc(g/cm3)",
    "Density_calibrated(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "SA_Score",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
    "Vertical_BDE",
]

PREFERRED_TARGETS = [
    # density calibrated preferred if present
    "Density_calibrated(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "SA_Score",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
    "Vertical_BDE",
]

SENSITIVITY_TARGETS = [
    "HOMO_LUMO_Gap(eV)",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Vertical_BDE(kcal/mol)",
    "Vertical_BDE",
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError("No file found from candidates:\n" + "\n".join(map(str, paths)))


def infer_smiles_col(df):
    candidates = [
        "SMILES",
        "smiles",
        "Smiles",
        "Canonical_SMILES",
        "Canonical_SMILES_NoIso",
        "Molecule",
    ]
    for c in candidates:
        if c in df.columns:
            sample = df[c].dropna().astype(str).head(50).tolist()
            ok = sum(Chem.MolFromSmiles(x) is not None for x in sample)
            if ok >= max(3, len(sample) // 3):
                return c

    best_col, best_ok = None, -1
    for c in df.columns:
        if df[c].dtype != object:
            continue
        sample = df[c].dropna().astype(str).head(100).tolist()
        ok = sum(Chem.MolFromSmiles(x) is not None for x in sample)
        if ok > best_ok:
            best_col, best_ok = c, ok

    if best_col is None or best_ok < 3:
        raise ValueError("Cannot infer SMILES column.")
    return best_col


def safe_mol(smi):
    if pd.isna(smi):
        return None
    return Chem.MolFromSmiles(str(smi))


def count_smarts(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None or mol is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def rdkit_global_descriptors(mol):
    if mol is None:
        return {}

    desc = {
        "RDKit_ExactMolWt": Descriptors.ExactMolWt(mol),
        "RDKit_MolLogP": Crippen.MolLogP(mol),
        "RDKit_TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RDKit_NumHeavyAtoms": mol.GetNumHeavyAtoms(),
        "RDKit_NumHeteroatoms": Lipinski.NumHeteroatoms(mol),
        "RDKit_NumRings": rdMolDescriptors.CalcNumRings(mol),
        "RDKit_NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "RDKit_NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "RDKit_NumHBD": Lipinski.NumHDonors(mol),
        "RDKit_NumHBA": Lipinski.NumHAcceptors(mol),
        "RDKit_FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }

    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    n_c = sum(z == 6 for z in atoms)
    n_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    n_n = sum(z == 7 for z in atoms)
    n_o = sum(z == 8 for z in atoms)

    desc.update({
        "Count_C": n_c,
        "Count_H": n_h,
        "Count_N": n_n,
        "Count_O": n_o,
        "N_to_C_Ratio": n_n / max(n_c, 1),
        "O_to_C_Ratio": n_o / max(n_c, 1),
        "N_O_Ratio": n_n / max(n_o, 1),
        "Nitrogen_Oxygen_Total": n_n + n_o,
    })

    # Energetic/sensitivity motif descriptors.
    smarts_dict = {
        "Num_C_NO2": "[#6]-[N+](=O)[O-]",
        "Num_Aromatic_C_NO2": "[c]-[N+](=O)[O-]",
        "Num_N_NO2": "[#7]-[N+](=O)[O-]",
        "Num_O_NO2": "[#8]-[N+](=O)[O-]",
        "Num_N_eq_N": "[#7]=[#7]",
        "Num_Azide": "[N-]=[N+]=N",
        "Num_N_N_single": "[#7]-[#7]",
        "Num_N_O_single": "[#7]-[#8]",
        "Num_C_N_single": "[#6]-[#7]",
        "Num_Furazan_like": "o1nncc1",
        "Num_Tetrazole_like": "n1nnnc1",
        "Num_Nitroso": "[#6,#7]-N=O",
    }

    for name, smarts in smarts_dict.items():
        desc[name] = count_smarts(mol, smarts)

    desc["Explosophore_Count"] = (
        desc["Num_C_NO2"]
        + desc["Num_N_NO2"]
        + desc["Num_O_NO2"]
        + desc["Num_N_eq_N"]
        + desc["Num_Azide"]
    )

    desc["Trigger_Linkage_Count"] = (
        desc["Num_N_N_single"]
        + desc["Num_N_O_single"]
        + desc["Num_C_N_single"]
    )

    heavy = max(desc["RDKit_NumHeavyAtoms"], 1)
    desc["Nitro_Per_HeavyAtom"] = (desc["Num_C_NO2"] + desc["Num_N_NO2"] + desc["Num_O_NO2"]) / heavy
    desc["Explosophore_Per_HeavyAtom"] = desc["Explosophore_Count"] / heavy
    desc["Trigger_Linkage_Per_HeavyAtom"] = desc["Trigger_Linkage_Count"] / heavy

    # Oxygen balance proxy, C/H/N/O only.
    mw = max(desc["RDKit_ExactMolWt"], 1e-8)
    desc["Oxygen_Balance_100"] = -1600.0 * (2 * n_c + n_h / 2.0 - n_o) / mw

    return desc


def target_transform(y, target):
    y = np.asarray(y, dtype=float)
    if target == "Sigma2_tot":
        return np.log(np.clip(y, 1e-12, None)), "log"
    if target == "Nu":
        # Current Nu values are normally around 0.15-0.25.
        upper = 0.25
        z = np.clip(y / upper, 1e-6, 1 - 1e-6)
        return np.log(z / (1 - z)), "bounded_logit_0p25"
    return y, "identity"


def inverse_transform(yhat, transform_name):
    yhat = np.asarray(yhat, dtype=float)
    if transform_name == "log":
        return np.exp(yhat)
    if transform_name == "bounded_logit_0p25":
        return 0.25 / (1 + np.exp(-yhat))
    return yhat


def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
    }


def build_models(seed=42):
    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
        "HistGBR": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(with_mean=False),
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                l2_regularization=0.05,
                random_state=seed,
            ),
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

    return models


def numeric_feature_frame(df, prefix_allow=None, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    num = df.select_dtypes(include=[np.number]).copy()
    drop = [c for c in num.columns if c in exclude_cols]
    if drop:
        num = num.drop(columns=drop)
    if prefix_allow is not None:
        keep = [c for c in num.columns if any(c.startswith(p) for p in prefix_allow)]
        num = num[keep]
    return num


def clean_X(X):
    X = X.replace([np.inf, -np.inf], np.nan)
    # Drop all-NaN columns.
    X = X.dropna(axis=1, how="all")
    # Median impute remaining NaNs manually to keep ExtraTrees simple.
    med = X.median(numeric_only=True)
    return X.fillna(med).fillna(0.0)


def align_xtb_by_row_index(xtb, row_idx):
    row_idx = [int(x) for x in row_idx]
    if "row_index" in xtb.columns:
        keyed = xtb.copy()
        keyed["row_index"] = pd.to_numeric(keyed["row_index"], errors="coerce")
        keyed = keyed.dropna(subset=["row_index"]).copy()
        keyed["row_index"] = keyed["row_index"].astype(int)
        dup = keyed["row_index"].duplicated(keep=False)
        if dup.any():
            examples = sorted(keyed.loc[dup, "row_index"].unique().tolist())[:10]
            raise ValueError(f"Duplicate row_index values in xTB table: {examples}")
        missing = sorted(set(row_idx) - set(keyed["row_index"].tolist()))
        if missing:
            raise ValueError(f"xTB table is missing row_index values, first examples: {missing[:10]}")
        return keyed.set_index("row_index").loc[row_idx].reset_index(drop=False)
    return xtb.iloc[row_idx].reset_index(drop=True).copy()


def main():
    data_path = first_existing(DATA_FILES)
    target_path = first_existing(TARGET_FILES)

    print(f"[INFO] Data: {data_path}")
    print(f"[INFO] Target: {target_path}")
    print(f"[INFO] xTB: {XTB_FILE}")
    print(f"[INFO] Split: {SPLIT_FILE}")
    print(f"[INFO] Trigger descriptors: {TRIGGER_FILE}")
    print(f"[INFO] XGBoost available: {HAS_XGB}")

    df = pd.read_csv(data_path)
    target_df = pd.read_csv(target_path)
    split_df = pd.read_csv(SPLIT_FILE)
    xtb = pd.read_csv(XTB_FILE)
    trigger = pd.read_csv(TRIGGER_FILE)

    smi_col = infer_smiles_col(df)
    print(f"[INFO] SMILES column: {smi_col}")

    # Align by Row_Index from split file.
    row_idx = split_df["Row_Index"].astype(int).values
    work = df.iloc[row_idx].reset_index(drop=True).copy()
    work["Row_Index"] = row_idx

    # Attach target columns by row order from target matrix if missing.
    for t in TARGETS:
        if t not in work.columns and t in target_df.columns:
            work[t] = target_df.iloc[row_idx][t].values

    # Attach splits.
    for c in ["random_split", "scaffold_split", "butina_split"]:
        work[c] = split_df[c].values

    # Build RDKit descriptors.
    rdkit_rows = []
    for smi in work[smi_col].astype(str).tolist():
        mol = safe_mol(smi)
        rdkit_rows.append(rdkit_global_descriptors(mol))
    rdkit_df = pd.DataFrame(rdkit_rows)

    # xTB numeric features, aligned by explicit row_index when available.
    xtb_aligned = align_xtb_by_row_index(xtb, row_idx)
    xtb_num = numeric_feature_frame(xtb_aligned)

    # Remove obvious index-like columns if present.
    for bad in ["row_index", "Row_Index", "example_id", "Example_ID"]:
        if bad in xtb_num.columns:
            xtb_num = xtb_num.drop(columns=[bad])

    xtb_num = xtb_num.add_prefix("xTB__")

    # Trigger local features.
    trigger_map = trigger.set_index("Row_Index")
    trigger_aligned = trigger_map.loc[row_idx].reset_index(drop=False)
    trigger_num = numeric_feature_frame(
        trigger_aligned,
        prefix_allow=[
            "Motif_",
            "Trigger_",
            "Weakest_",
        ],
    )
    trigger_num = trigger_num.add_prefix("TrigLocal__")

    base_X = pd.concat([rdkit_df.add_prefix("RDKit__"), xtb_num], axis=1)
    trig_X = pd.concat([base_X, trigger_num], axis=1)

    base_X = clean_X(base_X)
    trig_X = clean_X(trig_X)

    print(f"[INFO] Base feature dimension: {base_X.shape[1]}")
    print(f"[INFO] Base + trigger-local feature dimension: {trig_X.shape[1]}")
    print(f"[INFO] Samples: {len(work)}")

    # Choose targets. Prefer calibrated density and avoid duplicate SAscore/SA_Score.
    final_targets = []
    if "Density_calibrated(g/cm3)" in work.columns:
        final_targets.append("Density_calibrated(g/cm3)")
    elif "Density_calc(g/cm3)" in work.columns:
        final_targets.append("Density_calc(g/cm3)")

    for t in [
        "Heat_of_Formation(kcal/mol)",
        "HOMO_LUMO_Gap(eV)",
        "SAscore",
        "SA_Score",
        "VS_max",
        "Sigma2_tot",
        "Nu",
        "Trigger_Bond_Rho",
        "Molecular_Weight",
        "Vertical_BDE(kcal/mol)",
        "Vertical_BDE",
    ]:
        if t in work.columns and t not in final_targets:
            # Avoid SA duplicate if both exist.
            if t == "SA_Score" and "SAscore" in final_targets:
                continue
            final_targets.append(t)

    print("[INFO] Targets:")
    for t in final_targets:
        print("  -", t)

    rows = []
    feature_sets = {
        "Base_2D_xTB": base_X,
        "Base_2D_xTB_plus_TriggerLocal": trig_X,
    }

    split_cols = ["random_split", "scaffold_split", "butina_split"]

    for split_col in split_cols:
        print(f"\n===== Split: {split_col} =====")
        train_mask = work[split_col].eq("train").values
        calib_mask = work[split_col].eq("calib").values
        val_mask = work[split_col].eq("val").values

        for target in final_targets:
            y_raw = pd.to_numeric(work[target], errors="coerce")
            valid = y_raw.notna().values

            tr = train_mask & valid
            ca = calib_mask & valid
            va = val_mask & valid

            if tr.sum() < 100 or va.sum() < 20:
                print(f"[WARN] skip {target} in {split_col}; insufficient data.")
                continue

            y_trans, trans_name = target_transform(y_raw.values, target)

            for fs_name, X in feature_sets.items():
                models = build_models(seed=42)
                best = None

                for model_name, model in models.items():
                    try:
                        model.fit(X.loc[tr], y_trans[tr])
                        pred_calib = inverse_transform(model.predict(X.loc[ca]), trans_name)
                        pred_val = inverse_transform(model.predict(X.loc[va]), trans_name)

                        m_cal = metrics(y_raw.values[ca], pred_calib) if ca.sum() >= 20 else {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
                        m_val = metrics(y_raw.values[va], pred_val)

                        rec = {
                            "Split": split_col.replace("_split", ""),
                            "Feature_Set": fs_name,
                            "Target": target,
                            "Transform": trans_name,
                            "Model": model_name,
                            "Train_N": int(tr.sum()),
                            "Calib_N": int(ca.sum()),
                            "Val_N": int(va.sum()),
                            "Calib_R2": m_cal["R2"],
                            "Calib_MAE": m_cal["MAE"],
                            "Calib_RMSE": m_cal["RMSE"],
                            "Val_R2": m_val["R2"],
                            "Val_MAE": m_val["MAE"],
                            "Val_RMSE": m_val["RMSE"],
                        }

                        # Select by calibration R2 if available, otherwise validation R2.
                        score = rec["Calib_R2"] if np.isfinite(rec["Calib_R2"]) else rec["Val_R2"]
                        if best is None or score > best["_score"]:
                            best = rec | {"_score": score}
                    except Exception as e:
                        print(f"[WARN] {split_col} | {fs_name} | {target} | {model_name} failed: {e}")

                if best is not None:
                    best.pop("_score", None)
                    rows.append(best)
                    print(
                        f"{fs_name:34s} | {target:32s} | "
                        f"best={best['Model']:10s} | Val_R2={best['Val_R2']:.4f} | Val_RMSE={best['Val_RMSE']:.4g}"
                    )

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUTDIR / "ood_trigger_ablation_metrics_10d.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Delta: plus trigger - base.
    base = metrics_df[metrics_df["Feature_Set"] == "Base_2D_xTB"].copy()
    plus = metrics_df[metrics_df["Feature_Set"] == "Base_2D_xTB_plus_TriggerLocal"].copy()

    key = ["Split", "Target"]
    merged = base.merge(
        plus,
        on=key,
        suffixes=("_Base", "_PlusTrigger"),
        how="inner",
    )

    delta = pd.DataFrame({
        "Split": merged["Split"],
        "Target": merged["Target"],
        "Base_Model": merged["Model_Base"],
        "PlusTrigger_Model": merged["Model_PlusTrigger"],
        "Base_Val_R2": merged["Val_R2_Base"],
        "PlusTrigger_Val_R2": merged["Val_R2_PlusTrigger"],
        "Delta_Val_R2": merged["Val_R2_PlusTrigger"] - merged["Val_R2_Base"],
        "Base_Val_RMSE": merged["Val_RMSE_Base"],
        "PlusTrigger_Val_RMSE": merged["Val_RMSE_PlusTrigger"],
        "Delta_Val_RMSE": merged["Val_RMSE_PlusTrigger"] - merged["Val_RMSE_Base"],
    })

    delta_path = OUTDIR / "ood_trigger_ablation_delta_10d.csv"
    delta.to_csv(delta_path, index=False)

    # Summary for sensitivity-related targets.
    sens_delta = delta[delta["Target"].isin(SENSITIVITY_TARGETS)].copy()
    summary_rows = []
    for split in sorted(sens_delta["Split"].unique()):
        sub = sens_delta[sens_delta["Split"] == split]
        summary_rows.append({
            "Split": split,
            "Sensitivity_Target_N": len(sub),
            "Mean_Delta_Val_R2": sub["Delta_Val_R2"].mean(),
            "Median_Delta_Val_R2": sub["Delta_Val_R2"].median(),
            "Num_Targets_Improved_R2": int((sub["Delta_Val_R2"] > 0).sum()),
            "Num_Targets_Improved_R2_gt_0p01": int((sub["Delta_Val_R2"] > 0.01).sum()),
            "Mean_Delta_Val_RMSE": sub["Delta_Val_RMSE"].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_path = OUTDIR / "ood_trigger_ablation_summary_10d.csv"
    summary.to_csv(summary_path, index=False)

    print("\n===== Saved =====")
    print(metrics_path)
    print(delta_path)
    print(summary_path)

    print("\n===== Trigger-local ΔR2 by split/target =====")
    show = delta.sort_values(["Split", "Target"])
    print(show.to_string(index=False))

    print("\n===== Sensitivity-target summary =====")
    print(summary.to_string(index=False))

    # Practical decision note.
    print("\n===== Decision guide =====")
    print("If scaffold/butina Mean_Delta_Val_R2 > 0 and at least 2 sensitivity targets improve by >0.01,")
    print("then trigger-local descriptors are worth integrating into final_specialist_v4.")
    print("If Delta_Val_R2 is near zero or negative under scaffold/butina, do not patch 03 yet;")
    print("the next bottleneck is oracle-label quality or target definition, not descriptor engineering.")


if __name__ == "__main__":
    main()
