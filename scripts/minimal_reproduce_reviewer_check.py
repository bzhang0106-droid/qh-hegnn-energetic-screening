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
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required processed data files: " + "; ".join(missing))

    database = pd.read_csv(paths["database"])
    top20 = pd.read_csv(paths["top20"])
    parity = pd.read_csv(paths["parity"])

    checks = {
        "database_rows": len(database),
        "database_columns": len(database.columns),
        "top20_rows": len(top20),
        "top20_columns": len(top20.columns),
        "parity_metric_rows": len(parity),
        "parity_metric_columns": len(parity.columns),
    }
    if checks["database_rows"] != 5432:
        raise AssertionError(f"Expected 5432 curated molecules, found {checks['database_rows']}.")
    if checks["top20_rows"] < 20:
        raise AssertionError(f"Expected at least 20 QTAIM-aware candidate rows, found {checks['top20_rows']}.")
    if checks["parity_metric_rows"] == 0:
        raise AssertionError("Parity/source metric table is empty.")

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
