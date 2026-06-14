#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database-wide density calibration for curated_molecule_clean_v1.

Purpose:
  Add calibrated density columns to the active clean training database.

Important:
  Density_calibrated(g/cm3) is a high-throughput bias-corrected density estimate.
  It is not a full crystal-structure-predicted density from CSP/periodic DFT.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

DEFAULT_A = 0.748450
DEFAULT_B = 0.518061

DATASET = Path("data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
TARGET_MATRIX = Path("data/curated_molecule_clean_v1/target_matrix_9targets_molecule_clean.csv")
RELEASE_DATASET = Path("results/final_model_release/clean_sanitized_v1/training_dataset_clean_v1.csv")

REF_CANDIDATES = [
    Path("data/density_calibration/density_calibration_reference_8benchmarks_raw.csv"),
    Path("data/density_calibration_reference.csv"),
    Path("results/density_calibration/density_calibration_reference_8benchmarks_raw.csv"),
    Path("manuscript_npJ/SI/tables/Table_S6_Density_Calibration_Reference.csv"),
    Path("manuscript_npJ/SI/tables/Table_S6_Industrial_Benchmark_Density_Calibration_Raw.csv"),
    Path("manuscript_npJ/SI/supplementary_data/Supplementary_Data_5_Density_Calibration_Reference.csv"),
    Path("manuscript_npJ/SI/supplementary_data/Supplementary_Data_5_Industrial_Benchmark_Density_Calibration_Raw.csv"),
]


def find_density_columns(df: pd.DataFrame):
    cols = list(df.columns)

    proxy_keys = ["proxy", "calc", "pred"]
    crystal_keys = ["crystal", "ref", "exp", "experimental"]

    proxy_cols = []
    crystal_cols = []

    for c in cols:
        lc = c.lower()
        density_like = ("density" in lc) or ("rho" in lc) or ("ρ" in lc)
        if not density_like:
            continue

        if any(k in lc for k in proxy_keys):
            proxy_cols.append(c)

        if any(k in lc for k in crystal_keys):
            crystal_cols.append(c)

    # Prefer explicit names if present.
    preferred_proxy = [
        "rho_proxy",
        "Density_proxy",
        "Density_calc(g/cm3)",
        "Density_calc",
        "rho_calc",
    ]
    preferred_crystal = [
        "rho_crystal",
        "rho_crystal_ref",
        "Density_crystal",
        "Density_crystal_ref",
        "Density_reference",
        "Density_exp",
    ]

    for c in preferred_proxy:
        if c in df.columns:
            proxy_cols = [c] + [x for x in proxy_cols if x != c]
            break

    for c in preferred_crystal:
        if c in df.columns:
            crystal_cols = [c] + [x for x in crystal_cols if x != c]
            break

    if proxy_cols and crystal_cols:
        return proxy_cols[0], crystal_cols[0]

    return None, None


def load_or_fit_calibration():
    for p in REF_CANDIDATES:
        if not p.exists():
            continue

        try:
            ref = pd.read_csv(p)
        except Exception:
            continue

        proxy_col, crystal_col = find_density_columns(ref)
        if proxy_col is None or crystal_col is None:
            continue

        x = pd.to_numeric(ref[proxy_col], errors="coerce")
        y = pd.to_numeric(ref[crystal_col], errors="coerce")
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

        x = x[mask].to_numpy(dtype=float)
        y = y[mask].to_numpy(dtype=float)

        if len(x) < 4:
            continue

        a, b = np.polyfit(x, y, 1)
        pred = a * x + b
        mae = float(np.mean(np.abs(pred - y)))
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        ss_res = float(np.sum((pred - y) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else np.nan

        return {
            "a": float(a),
            "b": float(b),
            "source": str(p),
            "source_mode": "fit_from_reference_table",
            "proxy_col": proxy_col,
            "crystal_col": crystal_col,
            "n_ref": int(len(x)),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    return {
        "a": DEFAULT_A,
        "b": DEFAULT_B,
        "source": "stored_default_8benchmark_linear_calibration",
        "source_mode": "default_coefficients",
        "proxy_col": "Density_calc(g/cm3)",
        "crystal_col": "reference_crystal_density",
        "n_ref": 8,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
    }


def apply_calibration_to_table(path: Path, calib: dict, update_release: bool = False):
    if not path.exists():
        print(f"[SKIP] missing table: {path}")
        return None

    df = pd.read_csv(path)

    if "Density_calc(g/cm3)" not in df.columns:
        print(f"[SKIP] no Density_calc(g/cm3): {path}")
        return None

    rho_proxy = pd.to_numeric(df["Density_calc(g/cm3)"], errors="coerce")

    if "Density_proxy_before_calibration(g/cm3)" not in df.columns:
        df["Density_proxy_before_calibration(g/cm3)"] = rho_proxy

    rho_cal = calib["a"] * rho_proxy + calib["b"]

    df["Density_calibrated(g/cm3)"] = rho_cal
    df["Density_calibration_delta(g/cm3)"] = df["Density_calibrated(g/cm3)"] - rho_proxy
    df["Density_Label_Source"] = "proxy_linear_calibrated_to_reference_crystal_density"
    df["Density_Calibration_Formula"] = (
        f"rho_calibrated = {calib['a']:.6f} * Density_calc(g/cm3) + {calib['b']:.6f}"
    )
    df["Density_Calibration_Source"] = calib["source"]
    df["Density_Label_Is_True_Crystal"] = False

    # Conservative sanity flag only. Do not clip.
    df["Density_Calibration_Flag"] = "ok"
    df.loc[df["Density_calibrated(g/cm3)"].isna(), "Density_Calibration_Flag"] = "missing_proxy_density"
    df.loc[df["Density_calibrated(g/cm3)"] < 1.0, "Density_Calibration_Flag"] = "low_calibrated_density_check"
    df.loc[df["Density_calibrated(g/cm3)"] > 2.4, "Density_Calibration_Flag"] = "high_calibrated_density_check"

    df.to_csv(path, index=False)
    print(f"[OK] updated: {path} | rows={len(df)}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update_release", action="store_true", help="Also update final_model_release training_dataset_clean_v1.csv")
    args = parser.parse_args()

    outdir = Path("results/density_calibration_clean_v1")
    outdir.mkdir(parents=True, exist_ok=True)

    calib = load_or_fit_calibration()

    print("===== density calibration =====")
    for k, v in calib.items():
        print(f"{k}: {v}")

    main_df = apply_calibration_to_table(DATASET, calib)
    target_df = apply_calibration_to_table(TARGET_MATRIX, calib)

    if args.update_release:
        apply_calibration_to_table(RELEASE_DATASET, calib, update_release=True)

    if main_df is None:
        raise SystemExit("[ERROR] active clean dataset was not updated.")

    stats = {
        "dataset": str(DATASET),
        "n_rows": int(len(main_df)),
        "calibration_a": calib["a"],
        "calibration_b": calib["b"],
        "calibration_source": calib["source"],
        "source_mode": calib["source_mode"],
        "rho_proxy_mean": float(pd.to_numeric(main_df["Density_proxy_before_calibration(g/cm3)"], errors="coerce").mean()),
        "rho_proxy_median": float(pd.to_numeric(main_df["Density_proxy_before_calibration(g/cm3)"], errors="coerce").median()),
        "rho_calibrated_mean": float(pd.to_numeric(main_df["Density_calibrated(g/cm3)"], errors="coerce").mean()),
        "rho_calibrated_median": float(pd.to_numeric(main_df["Density_calibrated(g/cm3)"], errors="coerce").median()),
        "rho_calibrated_min": float(pd.to_numeric(main_df["Density_calibrated(g/cm3)"], errors="coerce").min()),
        "rho_calibrated_max": float(pd.to_numeric(main_df["Density_calibrated(g/cm3)"], errors="coerce").max()),
        "delta_mean": float(pd.to_numeric(main_df["Density_calibration_delta(g/cm3)"], errors="coerce").mean()),
        "delta_median": float(pd.to_numeric(main_df["Density_calibration_delta(g/cm3)"], errors="coerce").median()),
        "n_low_flag": int((main_df["Density_Calibration_Flag"] == "low_calibrated_density_check").sum()),
        "n_high_flag": int((main_df["Density_Calibration_Flag"] == "high_calibrated_density_check").sum()),
        "note": "Density_calibrated(g/cm3) is bias-corrected proxy density, not full CSP/periodic-DFT crystal density.",
    }

    pd.DataFrame([stats]).to_csv(outdir / "density_calibration_clean_v1_summary.csv", index=False)

    cols = [
        c for c in [
            "Molecule",
            "SMILES",
            "Density_calc(g/cm3)",
            "Density_proxy_before_calibration(g/cm3)",
            "Density_calibrated(g/cm3)",
            "Density_calibration_delta(g/cm3)",
            "Density_Calibration_Flag",
            "Density_Label_Source",
        ]
        if c in main_df.columns
    ]

    main_df[cols].to_csv(outdir / "density_calibrated_database_clean_v1.csv", index=False)

    print()
    print("===== summary =====")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print()
    print("[OK] wrote:", outdir / "density_calibration_clean_v1_summary.csv")
    print("[OK] wrote:", outdir / "density_calibrated_database_clean_v1.csv")


if __name__ == "__main__":
    main()
