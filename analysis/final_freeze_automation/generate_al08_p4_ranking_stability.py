from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
TRUTH = ROOT / "results/True_vs_Pred_Detonation.csv"
BOOT = ROOT / "results/density_calibration/Table_AL08_Density_Calibration_Bootstrap_10D.csv"
MULTISEED = ROOT / "manuscript_npJ/SI/model_diagnostics/Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv"
OUT = ROOT / "results/ranking_stability"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    truth = pd.read_csv(TRUTH).copy()
    boot = pd.read_csv(BOOT).copy()
    if not MULTISEED.exists():
        raise SystemExit(f"Missing multi-seed summary required for P4 audit: {MULTISEED}")
    multi = pd.read_csv(MULTISEED)

    truth = truth.sort_values("Oracle_D(km/s)", ascending=False).reset_index(drop=True)
    truth["Final_Rank"] = np.arange(1, len(truth) + 1)
    top_mols = truth.head(20)["Molecule"].tolist()

    rho_proxy = pd.to_numeric(truth["Density_Proxy(g/cm3)"], errors="coerce").to_numpy()
    d_proxy = pd.to_numeric(truth["D_proxy(km/s)"], errors="coerce").to_numpy()
    k_d = d_proxy / (1.0 + 1.3 * rho_proxy)
    base_rank = truth["Final_Rank"].to_numpy()

    rank_records = {m: [] for m in top_mols}
    rank_corr_rows = []
    for _, b in boot.iterrows():
        rho = np.clip(float(b["slope"]) * rho_proxy + float(b["intercept"]), 0.8, 3.2)
        d_sample = k_d * (1.0 + 1.3 * rho)
        order = np.argsort(-d_sample)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        sample_rank = pd.Series(ranks, index=truth["Molecule"])
        for m in top_mols:
            rank_records[m].append(int(sample_rank.loc[m]))
        rank_corr_rows.append(
            {
                "bootstrap_id": int(b["bootstrap_id"]),
                "spearman_rank_corr_vs_reported": float(pd.Series(base_rank).corr(pd.Series(ranks), method="spearman")),
                "top20_overlap_fraction": float(np.isin(truth.loc[order[:20], "Molecule"], top_mols).mean()),
            }
        )

    rows = []
    for _, row in truth.head(20).iterrows():
        ranks = np.array(rank_records[row["Molecule"]])
        rows.append(
            {
                "Molecule": row["Molecule"],
                "SMILES": row["SMILES"],
                "Final_Rank": int(row["Final_Rank"]),
                "Oracle_D(km/s)": float(row["Oracle_D(km/s)"]),
                "Oracle_P(GPa)": float(row["Oracle_P(GPa)"]),
                "Rank_boot_median": float(np.median(ranks)),
                "Rank_boot_p025": float(np.quantile(ranks, 0.025)),
                "Rank_boot_p975": float(np.quantile(ranks, 0.975)),
                "Top20_frequency_under_density_bootstrap": float(np.mean(ranks <= 20)),
                "Top10_frequency_under_density_bootstrap": float(np.mean(ranks <= 10)),
            }
        )
    stability = pd.DataFrame(rows)
    corr = pd.DataFrame(rank_corr_rows)

    seed_note = pd.DataFrame(
        {
            "Metric": ["mean_R2_std_across_targets", "max_R2_std_across_targets", "targets_in_multiseed_summary"],
            "Value": [
                float(pd.to_numeric(multi["R2_std"], errors="coerce").mean()),
                float(pd.to_numeric(multi["R2_std"], errors="coerce").max()),
                int(len(multi)),
            ],
            "Interpretation": [
                "Model-seed stability summary; ranking perturbation here uses density calibration bootstrap.",
                "Largest per-target R2 standard deviation across seeds.",
                "Number of 10D targets summarized across seeds 7/42/123.",
            ],
        }
    )

    stability_out = OUT / "Table_AL08_Final_Top20_Ranking_Stability_10D.csv"
    corr_out = OUT / "Table_AL08_Final_Rank_Correlation_10D.csv"
    seed_out = OUT / "Table_AL08_Final_Multiseed_Context_For_Ranking_10D.csv"
    stability.to_csv(stability_out, index=False)
    corr.to_csv(corr_out, index=False)
    seed_note.to_csv(seed_out, index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = stability["Final_Rank"]
    ax.errorbar(
        stability["Final_Rank"],
        stability["Rank_boot_median"],
        yerr=[
            stability["Rank_boot_median"] - stability["Rank_boot_p025"],
            stability["Rank_boot_p975"] - stability["Rank_boot_median"],
        ],
        fmt="o",
        color="#4477AA",
        ecolor="#99BBCC",
        capsize=2.5,
        ms=4,
    )
    ax.plot([1, 20], [1, 20], color="black", lw=0.8, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel("Reported final Top20 rank")
    ax.set_ylabel("Bootstrap rank under density calibration")
    ax.set_title("Top20 ranking stability under density calibration uncertainty")
    ax.grid(color="#dddddd", lw=0.6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"Figure_AL08_Final_Top20_Rank_Stability_10D.{ext}", dpi=300)
    plt.close(fig)

    print(f"[P4] {stability_out}")
    print(f"[P4] {corr_out}")
    print(f"[P4] {seed_out}")
    print(stability[["Molecule", "Final_Rank", "Rank_boot_p025", "Rank_boot_p975", "Top20_frequency_under_density_bootstrap"]].head(10).to_csv(index=False))


if __name__ == "__main__":
    main()
