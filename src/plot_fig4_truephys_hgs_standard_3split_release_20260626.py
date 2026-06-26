#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize


SCRIPT_DIR = Path(__file__).resolve().parent


def first_existing_dir(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the candidate data directories exists: " + "; ".join(str(p) for p in candidates))


def preferred_output_dir() -> Path:
    if (SCRIPT_DIR.parent / "main_figures").exists():
        return SCRIPT_DIR.parent / "main_figures"
    out = SCRIPT_DIR.parent / "figures" / "main"
    out.mkdir(parents=True, exist_ok=True)
    return out


DATA_DIR = first_existing_dir(
    [
        SCRIPT_DIR,
        SCRIPT_DIR.parent / "supplementary_information" / "data",
        SCRIPT_DIR.parent / "data" / "processed",
    ]
)
OUT_DIR = preferred_output_dir()
OUT_STEM = OUT_DIR / "Figure_4_Robustness_applicability_stress_maps"
SOURCE_OUT_PATH = OUT_DIR / "Figure_4_truephys_hgs_standard_3split_narrative_atlas_source_20260626.csv"
SUMMARY_PATH = DATA_DIR / "truephys_hgs_standard_3split_qh_hegnn_r2_summary_20260626.csv"
TESTROWS_PATH = DATA_DIR / "standard_3split_hgs_testrows_20260626.csv"
PRED_LONG_PATH = DATA_DIR / "truephys_hgs_standard_3split_prediction_long_20260626.csv"

TOKENS = {
    "surface": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "neutral_xlight": "#F4F5F7",
    "neutral_light": "#E2E5EA",
    "neutral_base": "#C5CAD3",
    "neutral_mid": "#7A828F",
    "neutral_dark": "#464C55",
    "blue_xlight": "#EAF1FE",
    "blue_light": "#CEDFFE",
    "blue_base": "#A3BEFA",
    "blue_mid": "#5477C4",
    "blue_dark": "#2E4780",
    "orange_xlight": "#FFEDDE",
    "orange_light": "#FFBDA1",
    "orange_base": "#F0986E",
    "orange_mid": "#CC6F47",
    "orange_dark": "#804126",
    "olive_xlight": "#D8ECBD",
    "olive_light": "#BEEB96",
    "olive_base": "#A3D576",
    "olive_mid": "#71B436",
    "olive_dark": "#386411",
    "gold_xlight": "#FFF4C2",
    "gold_base": "#FFE15B",
    "gold_dark": "#736422",
}

SPLIT_ORDER = ["random_80_20", "scaffold_80_20", "butina_80_20"]
SPLIT_LABELS = {"random_80_20": "Random", "scaffold_80_20": "Scaffold", "butina_80_20": "Butina"}
SPLIT_COLORS = {"random_80_20": TOKENS["blue_base"], "scaffold_80_20": TOKENS["orange_base"], "butina_80_20": TOKENS["olive_base"]}
SPLIT_DARK = {"random_80_20": TOKENS["blue_dark"], "scaffold_80_20": TOKENS["orange_dark"], "butina_80_20": TOKENS["olive_dark"]}

TARGET_ORDER = [
    "Density",
    "Heat_of_Formation",
    "HOMO_LUMO_Gap",
    "SA_Score",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Vertical_BDE",
]
TARGET_LABELS = {
    "Density": "Density",
    "Heat_of_Formation": r"$\Delta H_f$",
    "HOMO_LUMO_Gap": r"$E_{\mathrm{gap}}$",
    "SA_Score": "SA",
    "VS_max": r"$V_{S,\max}$",
    "Sigma2_tot": r"$\sigma^2_{\mathrm{tot}}$",
    "Nu": r"$\nu$",
    "Trigger_Bond_Rho": r"$\rho_{\mathrm{BCP}}$",
    "Vertical_BDE": "BDE",
}


def setup_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Segoe UI"],
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": TOKENS["surface"],
            "savefig.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["ink"],
            "ytick.color": TOKENS["ink"],
            "axes.titlesize": 8.1,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
        }
    )


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.2, fontweight="bold", color="black")


def style_axes(ax: plt.Axes, grid_axis: str = "none") -> None:
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(TOKENS["axis"])
        ax.spines[side].set_linewidth(0.7)
    if grid_axis == "x":
        ax.grid(True, axis="x", color=TOKENS["grid"], lw=0.52)
    elif grid_axis == "y":
        ax.grid(True, axis="y", color=TOKENS["grid"], lw=0.52)
    elif grid_axis == "both":
        ax.grid(True, color=TOKENS["grid"], lw=0.52)
    else:
        ax.grid(False)
    ax.tick_params(width=0.7, length=2.1, pad=1.7)
    ax.set_axisbelow(True)


def smooth_hist(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, density=True)
    kernel_x = np.linspace(-2.3, 2.3, 11)
    kernel = np.exp(-0.5 * kernel_x**2)
    kernel = kernel / kernel.sum()
    hist = np.convolve(hist, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, hist


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SUMMARY_PATH).rename(columns={"split_family": "Split"})
    summary = summary[summary["Split"].isin(SPLIT_ORDER) & summary["Target"].isin(TARGET_ORDER)].copy()
    testrows = pd.read_csv(TESTROWS_PATH).rename(columns={"split_name": "Split", "row_index": "Row_Index"})
    testrows = testrows[testrows["Split"].isin(SPLIT_ORDER)].copy()
    pred = pd.read_csv(PRED_LONG_PATH)
    pred = pred[pred["Split"].isin(SPLIT_ORDER) & pred["Target"].isin(TARGET_ORDER)].copy()
    return summary, testrows, pred


def panel_a(ax: plt.Axes, split: str, testrows: pd.DataFrame, first: bool) -> pd.DataFrame:
    part = testrows[testrows["Split"].eq(split)].copy()
    vals = part["NearestTrainSim"].dropna().to_numpy()
    xs, ys = smooth_hist(vals, np.linspace(0.20, 1.00, 48))
    color = SPLIT_COLORS[split]
    dark = SPLIT_DARK[split]
    ax.fill_between(xs, ys, color=color, alpha=0.48, lw=0)
    ax.plot(xs, ys, color=dark, lw=0.9)
    med = float(np.median(vals))
    ax.axvline(med, color=dark, lw=0.85)
    ax.text(med + 0.012, max(ys) * 0.78, f"{med:.2f}", fontsize=6.1, color=TOKENS["ink"])
    ax.text(0.985, 0.92, f"n={len(part)}", transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=TOKENS["muted"])
    ax.set_title(SPLIT_LABELS[split], loc="left", pad=4.5)
    ax.set_xlim(0.20, 1.0)
    ax.set_ylim(0, max(ys) * 1.22)
    ax.set_xlabel("Nearest-train Tanimoto")
    if first:
        ax.set_ylabel("Density")
        add_panel_label(ax, "a", x=-0.16, y=1.10)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    style_axes(ax, "x")

    status = part["true_phys_status"].fillna("unknown").value_counts()
    ok = int(status.get("ok", 0))
    partial = int(status.get("partial", 0))
    other = int(len(part) - ok - partial)

    return pd.DataFrame(
        {
            "panel": ["a_validation_profiles"],
            "Split": [split],
            "n": [len(part)],
            "median_nearest_train_tanimoto": [med],
            "true_phys_ok": [ok],
            "true_phys_partial": [partial],
            "true_phys_other": [other],
        }
    )


def panel_b(ax: plt.Axes, summary: pd.DataFrame) -> pd.DataFrame:
    data = summary.copy()
    data["Target"] = pd.Categorical(data["Target"], TARGET_ORDER, ordered=True)
    data["Split"] = pd.Categorical(data["Split"], SPLIT_ORDER, ordered=True)
    data = data.sort_values(["Target", "Split"])
    cmap = LinearSegmentedColormap.from_list("r2_bubbles", [TOKENS["neutral_xlight"], TOKENS["blue_light"], TOKENS["blue_mid"]])
    norm = Normalize(vmin=0.50, vmax=1.0)
    ax.set_xlim(-0.78, len(SPLIT_ORDER) - 0.22)
    ax.set_ylim(len(TARGET_ORDER) - 0.05, -0.95)
    for y in range(len(TARGET_ORDER)):
        ax.axhline(y, color=TOKENS["grid"], lw=0.48, zorder=0)
    for x in range(len(SPLIT_ORDER)):
        ax.axvline(x, color=TOKENS["grid"], lw=0.48, zorder=0)
    for _, row in data.iterrows():
        x = SPLIT_ORDER.index(str(row["Split"]))
        y = TARGET_ORDER.index(str(row["Target"]))
        r2 = float(row["R2_mean"])
        size = 82 + 360 * max(r2, 0) ** 2
        ax.scatter(x, y, s=size, color=cmap(norm(r2)), edgecolor=TOKENS["neutral_dark"], linewidth=0.7, zorder=2, clip_on=False)
        ax.text(x, y, f"{r2:.2f}", ha="center", va="center", fontsize=5.5, color="white" if r2 >= 0.86 else TOKENS["ink"], zorder=3)
    ax.set_xticks(range(len(SPLIT_ORDER)), [SPLIT_LABELS[s] for s in SPLIT_ORDER])
    ax.set_yticks(range(len(TARGET_ORDER)), [TARGET_LABELS[t] for t in TARGET_ORDER])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(TOKENS["axis"])
        spine.set_linewidth(0.7)
    ax.set_title("Target-by-split performance", loc="left", pad=7)
    add_panel_label(ax, "b", x=-0.14, y=1.08)
    return data.assign(panel="b_r2_bubble_matrix")


def panel_c(ax: plt.Axes, summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot(index="Target", columns="Split", values="R2_mean").reindex(TARGET_ORDER)
    out = wide.copy()
    out["floor"] = out[SPLIT_ORDER].min(axis=1)
    out["ceiling"] = out[SPLIT_ORDER].max(axis=1)
    out["loss"] = out["random_80_20"] - out["floor"]
    out = out.sort_values("floor", ascending=True)
    y = np.arange(len(out))
    ax.barh(y, out["floor"], color=TOKENS["blue_base"], edgecolor=TOKENS["blue_dark"], linewidth=0.7, height=0.55)
    for yy, (_, row) in zip(y, out.iterrows()):
        ax.plot([row["floor"], row["ceiling"]], [yy, yy], color=TOKENS["neutral_mid"], lw=0.9, zorder=3)
        ax.scatter(row["ceiling"], yy, s=17, facecolor=TOKENS["panel"], edgecolor=TOKENS["neutral_dark"], linewidth=0.7, zorder=4)
        label_x = max(0.468, row["floor"] - 0.030)
        ax.text(label_x, yy, f"{row['floor']:.2f}", va="center", ha="right", fontsize=5.7, color=TOKENS["ink"])
    ax.set_yticks(y, [TARGET_LABELS[str(t)] for t in out.index])
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("Minimum mean R$^2$")
    ax.set_title("Robustness floor", loc="left", pad=7)
    style_axes(ax, "x")
    add_panel_label(ax, "c", x=-0.14, y=1.08)
    return out.reset_index().assign(panel="c_robustness_floor")


def sensitivity_targets(pred: pd.DataFrame) -> list[str]:
    subset = pred[pred["NearestTrainSim"].le(0.50) | pred["NearestTrainSim"].ge(0.65)].copy()
    subset["band"] = np.where(subset["NearestTrainSim"].le(0.50), "low", "high")
    med = subset.groupby(["Target", "band"], observed=False)["NormAbsErr"].median().unstack()
    med["delta"] = med["low"] - med["high"]
    return [str(x) for x in med.sort_values("delta", ascending=False).head(4).index]


def panel_d(ax: plt.Axes, pred: pd.DataFrame, target: str, first: bool = False) -> pd.DataFrame:
    bins = np.array([0.20, 0.35, 0.50, 0.65, 0.80, 1.00])
    labels = pd.IntervalIndex.from_breaks(bins, closed="left")
    part = pred[pred["Target"].eq(target)].copy()
    part["sim_bin"] = pd.cut(part["NearestTrainSim"], bins=bins, right=False, include_lowest=True)
    agg = (
        part.groupby("sim_bin", observed=False)
        .agg(median_norm_abs_error=("NormAbsErr", "median"), q25=("NormAbsErr", lambda x: np.nanpercentile(x, 25)), q75=("NormAbsErr", lambda x: np.nanpercentile(x, 75)), n=("NormAbsErr", "size"))
        .reindex(labels)
        .reset_index()
    )
    centers = np.array([(iv.left + iv.right) / 2 for iv in labels])
    y = agg["median_norm_abs_error"].to_numpy(dtype=float)
    q25 = agg["q25"].to_numpy(dtype=float)
    q75 = agg["q75"].to_numpy(dtype=float)
    ax.fill_between(centers, q25, q75, color=TOKENS["orange_light"], alpha=0.30, linewidth=0)
    ax.plot(centers, y, color=TOKENS["orange_dark"], lw=0.98)
    ax.scatter(centers, y, s=20, facecolor=TOKENS["orange_base"], edgecolor=TOKENS["orange_dark"], linewidth=0.68, zorder=3)
    ax.set_xlim(0.22, 0.98)
    ax.set_ylim(0, max(0.42, float(np.nanmax(q75)) * 1.16))
    ax.set_title(TARGET_LABELS[target], loc="left", pad=4.5)
    ax.set_xlabel("Nearest-train Tanimoto")
    if first:
        ax.set_ylabel("|error| / target IQR")
        add_panel_label(ax, "d", x=-0.20, y=1.12)
    else:
        ax.set_ylabel("")
    style_axes(ax, "both")
    return agg.assign(Target=target, panel="d_error_curves")


def main() -> None:
    setup_theme()
    summary, testrows, pred = load_data()
    fig = plt.figure(figsize=(7.35, 7.45), constrained_layout=False)
    gs = fig.add_gridspec(4, 12, height_ratios=[1.05, 1.42, 1.16, 1.28], hspace=0.62, wspace=0.58)
    sources: list[pd.DataFrame] = []

    for i, split in enumerate(SPLIT_ORDER):
        sources.append(panel_a(fig.add_subplot(gs[0, i * 4 : (i + 1) * 4]), split, testrows, first=(i == 0)))
    sources.append(panel_b(fig.add_subplot(gs[1:3, 0:6]), summary))
    sources.append(panel_c(fig.add_subplot(gs[1:3, 7:12]), summary))

    target_list = sensitivity_targets(pred)
    curve_gs = gs[3, :].subgridspec(1, 4, wspace=0.34)
    for i, target in enumerate(target_list):
        sources.append(panel_d(fig.add_subplot(curve_gs[0, i]), pred, target, first=(i == 0)))

    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(OUT_STEM.with_suffix(suffix), dpi=430, bbox_inches="tight")
    plt.close(fig)

    pd.concat(sources, ignore_index=True, sort=False).to_csv(SOURCE_OUT_PATH, index=False)
    print(f"wrote={OUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
