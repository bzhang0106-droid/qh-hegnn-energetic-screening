from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
METRICS = ROOT / "results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv"
CAL = ROOT / "data/density_calibration/density_calibration_reference_8benchmarks_raw.csv"
TRUTH = ROOT / "results/True_vs_Pred_Detonation.csv"
SI = ROOT / "manuscript_npJ/SI/model_diagnostics"
DC = ROOT / "results/density_calibration"
SI.mkdir(parents=True, exist_ok=True)
DC.mkdir(parents=True, exist_ok=True)


def p3_teacher_vs_residual() -> None:
    df = pd.read_csv(METRICS)
    keep = df[df["Model_Group"].isin(["2D+xTB-teacher-v2", "Final-Specialist-Hybrid-v2"])].copy()
    wide = keep.pivot(index="Target", columns="Model_Group", values=["R2", "MAE", "RMSE"])
    out = pd.DataFrame({"Target": wide.index})
    for metric in ["R2", "MAE", "RMSE"]:
        out[f"Teacher_{metric}"] = wide[(metric, "2D+xTB-teacher-v2")].values
        out[f"Final_{metric}"] = wide[(metric, "Final-Specialist-Hybrid-v2")].values
    out["Delta_R2_Final_minus_Teacher"] = out["Final_R2"] - out["Teacher_R2"]
    out["Delta_MAE_Final_minus_Teacher"] = out["Final_MAE"] - out["Teacher_MAE"]
    out["Delta_RMSE_Final_minus_Teacher"] = out["Final_RMSE"] - out["Teacher_RMSE"]
    out["Interpretation"] = np.where(
        out["Delta_R2_Final_minus_Teacher"] > 0.002,
        "residual improves",
        np.where(out["Delta_R2_Final_minus_Teacher"] < -0.002, "teacher retained/safer", "statistically similar"),
    )
    table = SI / "Table_NPJ_Teacher_vs_Residual_10D_AL08_final.csv"
    out.to_csv(table, index=False)

    plot = out.sort_values("Delta_R2_Final_minus_Teacher")
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = ["#4477AA" if v >= 0 else "#BB5566" for v in plot["Delta_R2_Final_minus_Teacher"]]
    ax.barh(plot["Target"], plot["Delta_R2_Final_minus_Teacher"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Delta R2 (final residual model - teacher)")
    ax.set_ylabel("")
    ax.set_title("Residual correction gain by target")
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(SI / f"Figure_NPJ_Teacher_Residual_Gain_10D_AL08_final.{ext}", dpi=300)
    plt.close(fig)
    print(f"[P3] {table}")


def p5_density_bootstrap(seed: int = 42, n_boot: int = 5000) -> None:
    cal = pd.read_csv(CAL)
    x = pd.to_numeric(cal["Density_proxy"], errors="coerce").to_numpy()
    y = pd.to_numeric(cal["Crystal_Density(g/cm3)"], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    rng = np.random.default_rng(seed)

    rows = []
    coefs = []
    for i in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb, yb = x[idx], y[idx]
        try:
            a, b = np.polyfit(xb, yb, deg=1)
        except Exception:
            continue
        pred = a * x + b
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        coefs.append((a, b, rmse))
        rows.append({"bootstrap_id": i, "slope": a, "intercept": b, "rmse_on_reference": rmse})
    coef = pd.DataFrame(rows)
    boot_table = DC / "Table_AL08_Density_Calibration_Bootstrap_10D.csv"
    coef.to_csv(boot_table, index=False)

    truth = pd.read_csv(TRUTH).sort_values("Oracle_D(km/s)", ascending=False).head(20).copy()
    rho_proxy = pd.to_numeric(truth["Density_Proxy(g/cm3)"], errors="coerce").to_numpy()
    d_proxy = pd.to_numeric(truth["D_proxy(km/s)"], errors="coerce").to_numpy()
    p_proxy = pd.to_numeric(truth["P_proxy(GPa)"], errors="coerce").to_numpy()
    k_d = d_proxy / (1.0 + 1.3 * rho_proxy)
    k_p = p_proxy / np.square(rho_proxy)

    samples = []
    coef_arr = np.array(coefs)
    for rank, (_, row) in enumerate(truth.iterrows(), 1):
        rp = float(row["Density_Proxy(g/cm3)"])
        rho_s = coef_arr[:, 0] * rp + coef_arr[:, 1]
        rho_s = np.clip(rho_s, 0.8, 3.2)
        d_s = k_d[rank - 1] * (1.0 + 1.3 * rho_s)
        p_s = k_p[rank - 1] * np.square(rho_s)
        samples.append(
            {
                "Rank_by_Oracle_D": rank,
                "Molecule": row["Molecule"],
                "SMILES": row["SMILES"],
                "Density_proxy": rp,
                "Density_reported": row["Oracle_Density"],
                "Density_boot_mean": float(np.mean(rho_s)),
                "Density_boot_p025": float(np.quantile(rho_s, 0.025)),
                "Density_boot_p975": float(np.quantile(rho_s, 0.975)),
                "Oracle_D(km/s)": row["Oracle_D(km/s)"],
                "D_boot_mean(km/s)": float(np.mean(d_s)),
                "D_boot_p025(km/s)": float(np.quantile(d_s, 0.025)),
                "D_boot_p975(km/s)": float(np.quantile(d_s, 0.975)),
                "Oracle_P(GPa)": row["Oracle_P(GPa)"],
                "P_boot_mean(GPa)": float(np.mean(p_s)),
                "P_boot_p025(GPa)": float(np.quantile(p_s, 0.025)),
                "P_boot_p975(GPa)": float(np.quantile(p_s, 0.975)),
            }
        )
    top = pd.DataFrame(samples)
    top_table = DC / "Table_AL08_Top20_Density_Detonation_Uncertainty_10D.csv"
    top.to_csv(top_table, index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(
        top["Rank_by_Oracle_D"],
        top["Oracle_D(km/s)"],
        yerr=[
            top["Oracle_D(km/s)"] - top["D_boot_p025(km/s)"],
            top["D_boot_p975(km/s)"] - top["Oracle_D(km/s)"],
        ],
        fmt="o",
        color="#4477AA",
        ecolor="#99BBCC",
        capsize=2.5,
        ms=4,
    )
    ax.set_xlabel("Top-candidate rank by reported detonation velocity")
    ax.set_ylabel("D (km/s), density-calibration interval")
    ax.set_title("Density calibration uncertainty propagated to Top20")
    ax.grid(color="#dddddd", lw=0.6)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(DC / f"Figure_AL08_Density_Calibration_Uncertainty_10D.{ext}", dpi=300)
    plt.close(fig)
    print(f"[P5] {boot_table}")
    print(f"[P5] {top_table}")


def main() -> None:
    p3_teacher_vs_residual()
    p5_density_bootstrap()


if __name__ == "__main__":
    main()
