from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
PRED = ROOT / "manuscript_npJ/SI/model_diagnostics/Supplementary_NPJ_Validation_Predictions_final_specialist_10d_bde_xtbfull_AL08_seed42_20260605.csv"
SIM = ROOT / "results/model_optimization/scaffold_ood_split_assignments_10d.csv"
OUT_DIR = ROOT / "manuscript_npJ/SI/model_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "Density",
    "Heat_of_Formation",
    "HOMO_LUMO_Gap",
    "SA_Score",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE",
]


def main() -> None:
    pred = pd.read_csv(PRED)
    sim = pd.read_csv(
        SIM,
        usecols=[
            "Row_Index",
            "Molecule_ID",
            "scaffold_split",
            "scaffold_split_NearestTrainSim",
            "butina_split_NearestTrainSim",
            "random_split_NearestTrainSim",
        ],
    )
    df = pred.merge(sim, on="Row_Index", how="left")
    df["scaffold_split_NearestTrainSim"] = pd.to_numeric(df["scaffold_split_NearestTrainSim"], errors="coerce")

    rows = []
    for target in TARGETS:
        err_col = f"FinalAbsErr_{target}"
        teacher_col = f"TeacherAbsErr_{target}"
        if err_col not in df:
            continue
        d = df[["Row_Index", "Molecule", "SMILES", "scaffold_split", "scaffold_split_NearestTrainSim", err_col, teacher_col]].copy()
        d = d.rename(columns={err_col: "FinalAbsError", teacher_col: "TeacherAbsError"})
        d["Target"] = target
        d["FinalAbsError"] = pd.to_numeric(d["FinalAbsError"], errors="coerce")
        d["TeacherAbsError"] = pd.to_numeric(d["TeacherAbsError"], errors="coerce")
        d["DeltaAbsError_Final_minus_Teacher"] = d["FinalAbsError"] - d["TeacherAbsError"]
        rows.append(d)
    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["scaffold_split_NearestTrainSim", "FinalAbsError"])

    bins = [0.0, 0.4, 0.5, 0.6, 0.75, 0.9, 1.01]
    labels = ["<0.40", "0.40-0.50", "0.50-0.60", "0.60-0.75", "0.75-0.90", ">=0.90"]
    long["Similarity_Bin"] = pd.cut(long["scaffold_split_NearestTrainSim"], bins=bins, labels=labels, include_lowest=True, right=False)

    audit = []
    for target, g in long.groupby("Target", sort=False):
        corr = g[["scaffold_split_NearestTrainSim", "FinalAbsError"]].corr(method="spearman").iloc[0, 1]
        low = g[g["scaffold_split_NearestTrainSim"] < 0.6]
        high = g[g["scaffold_split_NearestTrainSim"] >= 0.6]
        audit.append(
            {
                "Target": target,
                "N_validation": int(len(g)),
                "Spearman_similarity_vs_final_abs_error": float(corr),
                "Median_abs_error_all": float(g["FinalAbsError"].median()),
                "Median_abs_error_sim_lt_0p60": float(low["FinalAbsError"].median()) if len(low) else np.nan,
                "N_sim_lt_0p60": int(len(low)),
                "Median_abs_error_sim_ge_0p60": float(high["FinalAbsError"].median()) if len(high) else np.nan,
                "N_sim_ge_0p60": int(len(high)),
            }
        )
    audit_df = pd.DataFrame(audit)

    long_out = OUT_DIR / "Table_NPJ_Redundancy_Leakage_ErrorVsSimilarity_Long_10D_AL08_final.csv"
    audit_out = OUT_DIR / "Table_NPJ_Redundancy_Leakage_Audit_10D_AL08_final.csv"
    long.to_csv(long_out, index=False)
    audit_df.to_csv(audit_out, index=False)

    fig, axes = plt.subplots(2, 5, figsize=(12.0, 5.4), sharex=True)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGETS):
        g = long[long["Target"] == target].copy()
        ax.scatter(g["scaffold_split_NearestTrainSim"], g["FinalAbsError"], s=9, alpha=0.35, color="#4477AA", linewidths=0)
        binned = g.groupby("Similarity_Bin", observed=True).agg(
            sim_mid=("scaffold_split_NearestTrainSim", "median"),
            err_med=("FinalAbsError", "median"),
        )
        ax.plot(binned["sim_mid"], binned["err_med"], color="#BB5566", marker="o", ms=3, lw=1.2)
        ax.set_title(target, fontsize=9)
        ax.grid(color="#e5e5e5", lw=0.5)
        ax.set_xlim(0.25, 1.02)
    for ax in axes[5:]:
        ax.set_xlabel("Nearest train similarity")
    for ax in axes[::5]:
        ax.set_ylabel("Final absolute error")
    fig.suptitle("Validation error versus scaffold-nearest training similarity", y=1.01, fontsize=12)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"Figure_NPJ_Error_vs_Similarity_10D_AL08_final.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[P1] {long_out}")
    print(f"[P1] {audit_out}")
    print(audit_df.to_csv(index=False))


if __name__ == "__main__":
    main()
