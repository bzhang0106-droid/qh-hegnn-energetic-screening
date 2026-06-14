from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
MAN = ROOT / "manuscript_npJ"
PKG = MAN / "final_submission_package_AL08_20260605"
OUT_DIR = MAN / "final_tables_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = [
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


def read_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "Target" not in df.columns or "R2" not in df.columns:
        return pd.DataFrame()
    return df


def best_2d_from_available() -> pd.DataFrame:
    candidates = []
    for path in [
        ROOT / "results/baselines/final_frozen_20260606/baseline_2d_only_metrics_final_frozen_2d_only_seed42_20260606.csv",
        ROOT / "results/baselines/baseline_2d_rf_metrics.csv",
        ROOT / "results/baselines/baseline_2d_xgboost_metrics.csv",
        ROOT / "results/baselines/baseline_2d_mlp_metrics.csv",
        ROOT / "results/baselines/baseline_2d_only_metrics.csv",
    ]:
        df = read_metrics(path)
        if not df.empty:
            df = df.copy()
            df["Source_File"] = str(path.relative_to(ROOT))
            candidates.append(df)
    if not candidates:
        return pd.DataFrame()
    all2d = pd.concat(candidates, ignore_index=True)
    all2d = all2d[all2d["Model_Group"].astype(str).str.contains("2D-only", na=False)].copy()
    all2d["Is_Final_Frozen"] = all2d["Source_File"].astype(str).str.contains("final_frozen_20260606")
    all2d = all2d.sort_values(["Target", "Is_Final_Frozen", "R2"], ascending=[True, False, False]).drop_duplicates("Target")
    all2d["Display_Model"] = "2D-only best available"
    all2d["Comparable_Status"] = np.where(all2d["Is_Final_Frozen"], "final_frozen_baseline", "available_but_split_may_differ")
    return all2d


def current_3d() -> pd.DataFrame:
    df = read_metrics(ROOT / "results/baselines/baseline_3d_egnn_only_metrics.csv")
    if df.empty:
        return df
    df = df.copy()
    df["Source_File"] = "results/baselines/baseline_3d_egnn_only_metrics.csv"
    df["Display_Model"] = "3D-only EGNN available"
    df["Comparable_Status"] = "final_frozen_baseline"
    return df


def hybrid_final() -> pd.DataFrame:
    df = read_metrics(ROOT / "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv")
    df = df[df["Model_Group"].astype(str).eq("Final-Specialist-Hybrid-v2")].copy()
    df["Source_File"] = "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv"
    df["Display_Model"] = "Hybrid final"
    df["Comparable_Status"] = "final_frozen_split"
    return df


def teacher_final() -> pd.DataFrame:
    df = read_metrics(ROOT / "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv")
    df = df[df["Model_Group"].astype(str).eq("2D+xTB-teacher-v2")].copy()
    df["Source_File"] = "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv"
    df["Display_Model"] = "2D+xTB teacher final"
    df["Comparable_Status"] = "final_frozen_split"
    return df


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = [best_2d_from_available(), current_3d(), teacher_final(), hybrid_final()]
    rows = []
    for df in pieces:
        if df.empty:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "Target": r["Target"],
                    "Display_Model": r["Display_Model"],
                    "Model_Group": r.get("Model_Group", ""),
                    "Model": r.get("Model", ""),
                    "R2": r.get("R2", np.nan),
                    "MAE": r.get("MAE", np.nan),
                    "RMSE": r.get("RMSE", np.nan),
                    "Train_N": r.get("Train_N", np.nan),
                    "Val_N": r.get("Val_N", np.nan),
                    "Comparable_Status": r.get("Comparable_Status", ""),
                    "Source_File": r.get("Source_File", ""),
                }
            )
    long = pd.DataFrame(rows)
    # Ensure every target/model combination exists.
    display_models = ["2D-only best available", "3D-only EGNN available", "2D+xTB teacher final", "Hybrid final"]
    full_idx = pd.MultiIndex.from_product([TARGET_ORDER, display_models], names=["Target", "Display_Model"])
    long = long.set_index(["Target", "Display_Model"]).reindex(full_idx).reset_index()
    long["Comparable_Status"] = long["Comparable_Status"].fillna("missing")
    long["Caveat"] = np.select(
        [
            long["Display_Model"].isin(["2D-only best available", "3D-only EGNN available"]) & long["Comparable_Status"].str.contains("available", na=False),
            long["Comparable_Status"].eq("missing"),
            long["Comparable_Status"].eq("final_frozen_split"),
            long["Comparable_Status"].eq("final_frozen_baseline"),
        ],
        [
            "Existing baseline result; not guaranteed to use the final frozen split. Use as preliminary baseline until final rerun completes.",
            "Metric not found in current artifacts.",
            "Final frozen 5432-database split.",
            "Final frozen 5432-database baseline rerun.",
        ],
        default="Check source file before manuscript claim.",
    )
    wide = long.pivot(index="Target", columns="Display_Model", values="R2").reset_index()
    wide["Target"] = pd.Categorical(wide["Target"], TARGET_ORDER, ordered=True)
    wide = wide.sort_values("Target")
    return long, wide


def plot_wide(wide: pd.DataFrame) -> tuple[Path, Path]:
    models = ["2D-only best available", "3D-only EGNN available", "2D+xTB teacher final", "Hybrid final"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(wide))
    width = 0.2
    colors = ["#7A7A7A", "#9C6B3C", "#4D78A8", "#2F6B4F"]
    for i, model in enumerate(models):
        vals = pd.to_numeric(wide.get(model), errors="coerce")
        ax.bar(x + (i - 1.5) * width, vals, width=width, label=model, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(wide["Target"].astype(str), rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Validation R2")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    png = OUT_DIR / "Figure_Final_Model_R2_Comparison_2D_3D_Hybrid.png"
    pdf = OUT_DIR / "Figure_Final_Model_R2_Comparison_2D_3D_Hybrid.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_location_index(long: pd.DataFrame, wide: pd.DataFrame, fig_paths: tuple[Path, Path]) -> None:
    fig_table = pd.DataFrame(
        [
            ["Fig. 1", "Closed-loop Workflow2.0 schematic", "not yet drawn as final vector", "main text", "Needs manual/final design redraw"],
            ["Fig. 2", "Final database chemical/property-space map", "manuscript_npJ/final_submission_package_AL08_20260605/main_figures/Figure_3_Density_HOF_Chemical_Space_AL08.png", "main text", "Candidate asset; should be redesigned into multi-panel final frozen version"],
            ["Fig. 3", "10D model parity and robustness", "manuscript_npJ/final_submission_package_AL08_20260605/main_figures/Figure_2_Model_Parity_10D_AL08_seed42.pdf", "main text", "Parity ready; multi-seed/leakage panels in SI assets"],
            ["Fig. 3b", "2D-only / 3D-only / Hybrid R2 comparison", str(fig_paths[1].relative_to(ROOT)), "main/SI", "Current baseline comparison includes caveats for non-final 2D/3D baselines"],
            ["Fig. 4", "Final global Top20 structure grid", "results/final_global_top20/Figure_Final_Global_Top20_Structures_10D.pdf", "main text", "Top20 is based on all 5432 frozen molecules"],
            ["Fig. 5", "Retrosynthesis/route-validation figure", "not yet available", "main/SI after ASKCOS/AiZynthFinder", "Requires installed retrosynthesis tool or API results"],
            ["SI Fig.", "Density calibration uncertainty", "manuscript_npJ/final_submission_package_AL08_20260605/si_figures/Figure_S_Density_Calibration_Uncertainty_AL08.pdf", "SI", "Ready"],
            ["SI Fig.", "Error vs similarity leakage audit", "manuscript_npJ/final_submission_package_AL08_20260605/si_figures/Figure_S_Error_vs_Similarity_AL08.pdf", "SI", "Ready"],
            ["SI Fig.", "Teacher vs residual gain", "manuscript_npJ/final_submission_package_AL08_20260605/si_figures/Figure_S_Teacher_Residual_Gain_AL08.pdf", "SI", "Ready"],
            ["SI Fig.", "P6 strict evidence tiers", "manuscript_npJ/final_submission_package_AL08_20260605/si_figures/Figure_S_P6_Evidence_Tier_Map_AL08.pdf", "SI", "Ready but screening-only"],
        ],
        columns=["Display_Item", "Design", "Current_File", "Placement", "Status_or_Next_Action"],
    )
    table_index = pd.DataFrame(
        [
            ["Table 1", "Final 10D model performance", "manuscript_npJ/final_submission_package_AL08_20260605/main_tables/Table_1_Final_Model_Performance_10D_AL08.csv", "main/SI", "Ready"],
            ["Table 2", "Final global Top20 all-property + synthesizability evidence table", "results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv", "main/SI", "Ready; route-validation columns pending"],
            ["Table 3", "2D-only / 3D-only / teacher / hybrid R2 comparison", "manuscript_npJ/final_tables_figures/Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Wide.csv", "main/SI", "Created; caveats for non-final 2D/3D baseline split"],
            ["SI Table", "Long model comparison with caveats", "manuscript_npJ/final_tables_figures/Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Long.csv", "SI", "Created"],
            ["SI Table", "Multi-seed stability", "manuscript_npJ/final_submission_package_AL08_20260605/si_tables/Table_S_MultiSeed_Stability_AL08.csv", "SI", "Ready"],
            ["SI Table", "Redundancy/leakage audit", "manuscript_npJ/final_submission_package_AL08_20260605/si_tables/Table_S_Redundancy_Leakage_Audit_AL08.csv", "SI", "Ready"],
            ["SI Table", "Retrosynthesis route checklist", "results/final_global_top20/Table_Final_Global_Top20_Retrosynthesis_Checklist_10D.csv", "SI", "Prepared; ASKCOS/AiZynthFinder not yet run"],
        ],
        columns=["Display_Item", "Design", "Current_File", "Placement", "Status_or_Next_Action"],
    )
    fig_table.to_csv(MAN / "Figure_Location_And_Design_Index_FINAL_FROZEN.csv", index=False)
    table_index.to_csv(MAN / "Table_Location_And_Design_Index_FINAL_FROZEN.csv", index=False)
    md = MAN / "FINAL_FIGURE_TABLE_LOCATION_AND_DESIGN_INDEX.md"
    md.write_text(
        "# Final Figure/Table Location and Design Index\n\n"
        "This is the controlling index for manuscript drafting. `AL08` in file names is provenance; manuscript text should use `final frozen database` and `final global Top20`.\n\n"
        "## Figures\n\n"
        + fig_table.to_markdown(index=False)
        + "\n\n## Tables\n\n"
        + table_index.to_markdown(index=False)
        + "\n\n## Model Comparison Caveat\n\n"
        "The 2D-only and 3D-only baselines currently available are marked with split caveats where applicable. They should be used as preliminary manuscript baselines until a final-frozen rerun completes.\n",
        encoding="utf-8",
    )


def sync_to_package(fig_paths: tuple[Path, Path]) -> None:
    if not PKG.exists():
        return
    (PKG / "main_tables").mkdir(exist_ok=True)
    (PKG / "si_tables").mkdir(exist_ok=True)
    (PKG / "si_figures").mkdir(exist_ok=True)
    for src, dst in [
        (OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Wide.csv", PKG / "main_tables/Table_3_Model_R2_Comparison_2D_3D_Hybrid.csv"),
        (OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Long.csv", PKG / "si_tables/Table_S_Model_R2_Comparison_2D_3D_Hybrid_Long.csv"),
        (fig_paths[0], PKG / "si_figures/Figure_S_Model_R2_Comparison_2D_3D_Hybrid.png"),
        (fig_paths[1], PKG / "si_figures/Figure_S_Model_R2_Comparison_2D_3D_Hybrid.pdf"),
        (MAN / "FINAL_FIGURE_TABLE_LOCATION_AND_DESIGN_INDEX.md", PKG / "DISPLAY_ITEM_PLAN_FINAL_FROZEN.md"),
    ]:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())


def main() -> None:
    long, wide = build_table()
    long.to_csv(OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Long.csv", index=False)
    wide.to_csv(OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Wide.csv", index=False)
    fig_paths = plot_wide(wide)
    write_location_index(long, wide, fig_paths)
    sync_to_package(fig_paths)
    print({
        "long": str((OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Long.csv").relative_to(ROOT)),
        "wide": str((OUT_DIR / "Table_Final_Model_R2_Comparison_2D_3D_Hybrid_Wide.csv").relative_to(ROOT)),
        "figure_pdf": str(fig_paths[1].relative_to(ROOT)),
        "index": str((MAN / "FINAL_FIGURE_TABLE_LOCATION_AND_DESIGN_INDEX.md").relative_to(ROOT)),
    })


if __name__ == "__main__":
    main()
