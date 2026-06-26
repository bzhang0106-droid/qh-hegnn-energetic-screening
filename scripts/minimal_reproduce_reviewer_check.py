from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight reviewer check for the QH-HEGNN code/data release."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing processed source/supplementary CSV files. Defaults to data/processed.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for regenerated reviewer-check files. Defaults to reviewer_minimal_check.",
    )
    args = parser.parse_args()

    repo_root = default_repo_root()
    data_root = args.data_root.resolve() if args.data_root else repo_root / "data" / "processed"
    outdir = args.outdir.resolve() if args.outdir else repo_root / "reviewer_minimal_check"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "database": data_root / "Supplementary_Data_1_Curated_5432_Molecule_Database.csv",
        "top20": data_root / "Table_2_QTAIM_Aware_Stability_Constrained_Top20_20260607.csv",
        "parity": data_root / "Table_S_Figure3_HELS_10Target_Parity_Metrics_20260608.csv",
        "fig1_source": data_root / "Figure_1a_generation_screening_stream_refined_NPJstyle_20260626_source.csv",
        "fig4_source": data_root / "Figure_4_truephys_hgs_standard_3split_narrative_atlas_source_20260626.csv",
        "fig4_r2": data_root / "truephys_hgs_standard_3split_qh_hegnn_r2_summary_20260626.csv",
        "fig4_testrows": data_root / "standard_3split_hgs_testrows_20260626.csv",
        "fig4_predictions": data_root / "truephys_hgs_standard_3split_prediction_long_20260626.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required processed data files: " + "; ".join(missing))

    database = pd.read_csv(paths["database"])
    top20 = pd.read_csv(paths["top20"])
    parity = pd.read_csv(paths["parity"])
    fig1_source = pd.read_csv(paths["fig1_source"])
    fig4_source = pd.read_csv(paths["fig4_source"])
    fig4_r2 = pd.read_csv(paths["fig4_r2"])
    fig4_testrows = pd.read_csv(paths["fig4_testrows"], usecols=["case", "split_name", "row_index", "NearestTrainSim"])
    fig4_predictions = pd.read_csv(paths["fig4_predictions"], usecols=["Split", "Target", "NearestTrainSim", "NormAbsErr"])

    checks = {
        "database_rows": len(database),
        "database_columns": len(database.columns),
        "top20_rows": len(top20),
        "top20_columns": len(top20.columns),
        "parity_metric_rows": len(parity),
        "parity_metric_columns": len(parity.columns),
        "fig1_source_rows": len(fig1_source),
        "fig4_source_rows": len(fig4_source),
        "fig4_r2_rows": len(fig4_r2),
        "fig4_split_families": fig4_r2["split_family"].nunique(),
        "fig4_targets": fig4_r2["Target"].nunique(),
        "fig4_testrows": len(fig4_testrows),
        "fig4_prediction_rows": len(fig4_predictions),
    }
    if checks["database_rows"] != 5432:
        raise AssertionError(f"Expected 5432 curated molecules, found {checks['database_rows']}.")
    if checks["top20_rows"] < 20:
        raise AssertionError(f"Expected at least 20 QTAIM-aware candidate rows, found {checks['top20_rows']}.")
    if checks["parity_metric_rows"] == 0:
        raise AssertionError("Parity/source metric table is empty.")
    if checks["fig1_source_rows"] != 5:
        raise AssertionError(f"Expected 5 Fig. 1a source rows, found {checks['fig1_source_rows']}.")
    if checks["fig4_r2_rows"] != 27 or checks["fig4_split_families"] != 3 or checks["fig4_targets"] != 9:
        raise AssertionError("Fig. 4 R2 summary should contain 9 targets across 3 validation split families.")
    if checks["fig4_testrows"] != 10171:
        raise AssertionError(f"Expected 10171 Fig. 4 validation rows, found {checks['fig4_testrows']}.")
    if checks["fig4_prediction_rows"] != 91539:
        raise AssertionError(f"Expected 91539 Fig. 4 prediction rows, found {checks['fig4_prediction_rows']}.")

    pd.DataFrame([checks]).to_csv(outdir / "minimal_reviewer_check_summary.csv", index=False)

    preferred_cols = [
        "Molecule",
        "SMILES",
        "Final_Detonation_D(km/s)",
        "Final_Detonation_P(GPa)",
        "Vertical_BDE(kcal/mol)",
        "Trigger_Bond_Rho",
        "SAscore",
        "Final_Synthesis_Readiness_Tier",
    ]
    top20_cols = [col for col in preferred_cols if col in top20.columns]
    if not top20_cols:
        top20_cols = list(top20.columns[: min(8, len(top20.columns))])
    top20[top20_cols].head(20).to_csv(outdir / "minimal_top20_preview.csv", index=False)
    fig4_r2[["split_family", "Target", "R2_mean", "MAE_mean", "RMSE_mean"]].to_csv(
        outdir / "minimal_fig4_truephys_hgs_r2_summary.csv", index=False
    )

    try:
        import matplotlib.pyplot as plt

        numeric_cols = list(parity.select_dtypes("number").columns)
        fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
        if numeric_cols:
            parity[numeric_cols[0]].plot(kind="bar", ax=ax, color="#4C78A8")
            ax.set_ylabel(numeric_cols[0])
            ax.set_xlabel("Target index")
        else:
            ax.text(0.5, 0.5, "No numeric parity column found", ha="center", va="center")
            ax.set_axis_off()
        ax.set_title("Reviewer minimal check: source metric preview")
        fig.tight_layout()
        fig.savefig(outdir / "minimal_parity_metric_preview.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        (outdir / "minimal_plot_warning.txt").write_text(str(exc), encoding="utf-8")

    print(f"Reviewer minimal check completed: {outdir}")


if __name__ == "__main__":
    main()
