from pathlib import Path

import pandas as pd

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
PROD = ROOT / "results/final_model_release/production_v4_10d"
SEED_DIR = PROD / "seed_stability_AL08"
OUT_DIR = ROOT / "manuscript_npJ/SI/model_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    42: PROD / "final_specialist_final_metrics_final_specialist_10d_bde_xtbfull_AL08_seed42_20260605.csv",
    7: SEED_DIR / "final_specialist_final_metrics_final_specialist_10d_bde_xtbfull_AL08_seed7_20260605.csv",
    123: SEED_DIR / "final_specialist_final_metrics_final_specialist_10d_bde_xtbfull_AL08_seed123_20260605.csv",
}

frames = []
missing = []
for seed, path in RUNS.items():
    if not path.exists():
        missing.append(str(path))
        continue
    df = pd.read_csv(path)
    df = df[df["Model_Group"].astype(str).eq("Final-Specialist-Hybrid-v2")].copy()
    df["Seed"] = seed
    frames.append(df)

if missing:
    raise SystemExit("Missing seed metric files: " + "; ".join(missing))
if not frames:
    raise SystemExit("No seed metric files loaded")

long = pd.concat(frames, ignore_index=True)
metric_cols = [c for c in ["MAE", "RMSE", "R2"] if c in long.columns]
for c in metric_cols:
    long[c] = pd.to_numeric(long[c], errors="coerce")

summary = (
    long.groupby("Target", as_index=False)
    .agg(
        R2_mean=("R2", "mean"),
        R2_std=("R2", "std"),
        R2_min=("R2", "min"),
        R2_max=("R2", "max"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        Seeds=("Seed", lambda s: ",".join(map(str, sorted(s.astype(int).unique())))),
    )
    .sort_values("Target")
)
summary["R2_mean_pm_std"] = summary.apply(lambda r: f"{r.R2_mean:.4f} +/- {r.R2_std:.4f}", axis=1)

long_out = OUT_DIR / "Table_NPJ_MultiSeed_Stability_10D_AL08_final_long.csv"
sum_out = OUT_DIR / "Table_NPJ_MultiSeed_Stability_10D_AL08_final.csv"
long.to_csv(long_out, index=False)
summary.to_csv(sum_out, index=False)
print(f"[SAVE] {long_out}")
print(f"[SAVE] {sum_out}")
print(summary[["Target", "R2_mean", "R2_std", "R2_min", "R2_max", "Seeds"]].to_csv(index=False))
