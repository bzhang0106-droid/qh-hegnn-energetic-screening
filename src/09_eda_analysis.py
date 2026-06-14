"""
09_eda_analysis.py

Manual HELS workflow model evaluation + EDA script.

Purpose:
1. Replace the old standalone evaluate_3d.py. This script evaluates the same
   10-target Native-EGNN architecture used by 03/04.
2. Reuse the fixed validation split saved by 03_egnn_painn_train.py.
3. Run coordinate-perturbation ablation as a robustness test, not as a default
   training requirement.
4. Produce EDA plots for the 10-target database.

Recommended usage:
    python -u 09_eda_analysis.py --mode all
    python -u 09_eda_analysis.py --mode eval
    python -u 09_eda_analysis.py --mode eda

Important:
- This script does not run ORCA and does not change the ORCA verification logic.
- Density_calc(g/cm3) is treated as a density proxy unless a true crystal-density
  label is explicitly supplied elsewhere.
"""

import argparse
import glob
import json
import os
import re
from typing import List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, radius_graph
from tqdm import tqdm

try:
    import seaborn as sns
except Exception:  # seaborn is convenient but not essential
    sns = None

RDLogger.DisableLog("rdApp.*")


def apply_npj_figure_style() -> None:
    """Use a clean Nature/npj-like style for manuscript-ready figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 8,
        "savefig.dpi": 600,
    })


apply_npj_figure_style()

OLD_CSV_PATH = "../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
XYZ_DIR = "../data/raw_2100_xyz"
MODEL_SAVE_PATH = "../results/final_model_release/production_v4_10d/final_specialist_10d_bde_production.joblib"
SPLIT_SAVE_PATH = "../results/final_model_release/production_v4_10d/split_final_specialist_10d_bde_production.json"
OUT_DIR = "../results/eda_plots"
EVAL_METRICS_CSV = "../results/Model_Evaluation_10Target.csv"
ABLATION_CSV = "../results/Coordinate_Noise_Ablation_10Target.csv"
PARITY_PLOT_PATH = "../results/Parity_Plot_10D.png"
PARITY_PLOT_DIR = "../results/model_parity_plots"
VALIDATION_PRED_GLOB = "../manuscript_npJ/SI/model_diagnostics/Supplementary_NPJ_Validation_Predictions_*.csv"
EVAL_TEXT_PATH = "../results/latest_model_eval_10Target.txt"

DEFAULT_TARGET_PROPS = [
    "Density_calc(g/cm3)",
    "Heat_of_Formation(kcal/mol)",
    "HOMO_LUMO_Gap(eV)",
    "SAscore",
    "VS_max",
    "Sigma2_tot",
    "Nu",
    "Trigger_Bond_Rho",
    "Molecular_Weight",
    "Vertical_BDE(kcal/mol)",
]

TWO_D_FEATURE_NAMES = [
    "ExactMolWt",
    "NumHeteroatoms",
    "NumRings",
    "NumNitro",
    "NumNitrogen",
    "NumC_NO2",
    "NumN_NO2",
    "NumN_eq_N",
    "NumAzide",
]

UNITS = {
    "Density_calc(g/cm3)": "g/cm³",
    "Heat_of_Formation(kcal/mol)": "kcal/mol",
    "HOMO_LUMO_Gap(eV)": "eV",
    "SAscore": "score",
    "VS_max": "kcal/mol",
    "Sigma2_tot": "variance",
    "Nu": "ratio",
    "Trigger_Bond_Rho": "e/Bohr³",
    "Molecular_Weight": "Da",
}

DISPLAY_NAMES = {
    "Density_calc(g/cm3)": "Density",
    "Heat_of_Formation(kcal/mol)": "Heat of Formation",
    "HOMO_LUMO_Gap(eV)": "HOMO-LUMO Gap",
    "SAscore": "SA Score",
    "VS_max": "VS Max (ESP)",
    "Sigma2_tot": "Sigma2 (Variance)",
    "Nu": "Nu (Balance)",
    "Trigger_Bond_Rho": "Trigger Bond (Rho)",
    "Molecular_Weight": "Molecular Weight",
}

ATOMIC_NUMS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17}


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def make_example_id(row_index: int, molecule: str, smiles: str) -> str:
    return f"row{int(row_index):08d}||{molecule}||{canonicalize_smiles(smiles)}"


# ==============================================================================
# Native EGNN, identical to 03/04.
# ==============================================================================
def native_scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    count = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    count.scatter_add_(0, index_expanded, torch.ones_like(src))
    return out / count.clamp(min=1)


class NativeEGNNLayer(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(emb_dim * 2 + 1, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.node_mlp = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.coord_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, 1, bias=False))

    def forward(self, h: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        coord_diff = pos[row] - pos[col]
        radial = torch.sum(coord_diff ** 2, dim=1).unsqueeze(1)
        m_ij = self.edge_mlp(torch.cat([h[row], h[col], radial], dim=-1))
        coord_msg = coord_diff * self.coord_mlp(m_ij)
        pos_aggr = native_scatter_mean(coord_msg, row, dim_size=pos.size(0))
        pos_out = pos + pos_aggr
        m_aggr = native_scatter_mean(m_ij, row, dim_size=h.size(0))
        h_out = h + self.node_mlp(torch.cat([h, m_aggr], dim=-1))
        return h_out, pos_out




# ==============================================================================
# Featurization
# ==============================================================================
def _count_substructure(mol: Chem.Mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def extract_2d_features(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [0.0] * len(TWO_D_FEATURE_NAMES)
    nitro = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
    return [
        float(Descriptors.ExactMolWt(mol)),
        float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        float(rdMolDescriptors.CalcNumRings(mol)),
        float(_count_substructure(mol, nitro)),
        float(sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)),
        float(_count_substructure(mol, f"[#6]-{nitro}")),
        float(_count_substructure(mol, f"[#7]-{nitro}")),
        float(_count_substructure(mol, "[#7]=[#7]")),
        float(_count_substructure(mol, "[N]=[N+]=[N-]")),
    ]


def load_xyz_coordinates(mol_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    xyz_path = os.path.join(XYZ_DIR, f"{mol_name}.xyz")
    if not os.path.exists(xyz_path):
        return None, None
    z, pos = [], []
    try:
        with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) < 3:
            return None, None
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] in ATOMIC_NUMS:
                z.append(ATOMIC_NUMS[parts[0]])
                pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None, None
    if not z:
        return None, None
    return torch.tensor(z, dtype=torch.long), torch.tensor(pos, dtype=torch.float)


class EnergeticDataset(Dataset):
    def __init__(self, data_list: List[Data]):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


def add_coordinate_noise(pos: torch.Tensor, batch: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return pos
    noise = torch.randn_like(pos) * std
    n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
    if n_graphs > 0:
        graph_mean_noise = native_scatter_mean(noise, batch, dim_size=n_graphs)
        noise = noise - graph_mean_noise[batch]
    return pos + noise


def build_eval_dataset(
    df: pd.DataFrame,
    target_props: List[str],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    x2d_mean: torch.Tensor,
    x2d_std: torch.Tensor,
    val_example_ids: Optional[set] = None,
    val_row_indices: Optional[set] = None,
    val_molecules: Optional[set] = None,
) -> EnergeticDataset:
    mol_col = "Molecule" if "Molecule" in df.columns else "Moleule"
    data_list: List[Data] = []
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="构建验证集 3D 图"):
        mol_name = str(row[mol_col]).replace(".xyz", "").replace(".out", "").strip()
        smiles = str(row["SMILES"])
        example_id = make_example_id(int(row_idx), mol_name, smiles)

        # Priority: exact v2.1 row-level identifiers > row indices > legacy molecule names.
        if val_example_ids is not None:
            if example_id not in val_example_ids:
                continue
        elif val_row_indices is not None:
            if int(row_idx) not in val_row_indices:
                continue
        elif val_molecules is not None:
            # Legacy fallback only. Molecule names are not unique across AL rounds.
            if mol_name not in val_molecules:
                continue

        z, pos = load_xyz_coordinates(mol_name)
        if z is None:
            continue
        y_real = torch.tensor(row[target_props].astype(float).values, dtype=torch.float)
        if torch.isnan(y_real).any():
            continue
        y_norm = ((y_real - y_mean.cpu()) / y_std.cpu()).unsqueeze(0)
        x2d_raw = torch.tensor(extract_2d_features(str(row["SMILES"])), dtype=torch.float)
        x2d_norm = ((x2d_raw - x2d_mean.cpu()) / x2d_std.cpu()).unsqueeze(0)
        data_list.append(
            Data(
                z=z,
                pos=pos,
                y=y_norm,
                y_real=y_real.unsqueeze(0),
                x_2d=x2d_norm,
                molecule=mol_name,
                smiles=smiles,
                example_id=example_id,
                row_index=int(row_idx),
            )
        )
    return EnergeticDataset(data_list)


def predict_dataset(model, loader, device, y_mean, y_std, noise_std: float = 0.0):
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pos = add_coordinate_noise(batch.pos, batch.batch, noise_std)
            out_norm = model(batch.z, pos, batch.batch, batch.x_2d)
            out_real = out_norm * y_std + y_mean
            preds.append(out_real.cpu().numpy())
            targets.append(batch.y_real.cpu().numpy())
    return np.vstack(preds), np.vstack(targets)


def parse_noise_grid(text: str) -> List[float]:
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return vals or [0.0]




def _safe_plot_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_") or "target"


def _latest_validation_predictions_path() -> Optional[str]:
    candidates = [p for p in glob.glob(VALIDATION_PRED_GLOB) if os.path.getsize(p) > 0]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def generate_validation_parity_plots() -> Optional[str]:
    """Create a manuscript-ready 2 x 5 true-vs-predicted parity figure."""
    pred_path = _latest_validation_predictions_path()
    if pred_path is None:
        print(f"[WARN] No validation prediction CSV found for parity plots: {VALIDATION_PRED_GLOB}")
        return None

    df = pd.read_csv(pred_path)
    targets = []
    for col in df.columns:
        if not col.startswith("True_"):
            continue
        target = col[len("True_"):]
        pred_col = f"FinalPred_{target}"
        if pred_col in df.columns:
            targets.append(target)
    if not targets:
        print(f"[WARN] Validation prediction CSV has no True_*/FinalPred_* pairs: {pred_path}")
        return None

    target_order = [
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
    order_index = {name: i for i, name in enumerate(target_order)}
    targets = sorted(targets, key=lambda x: order_index.get(x, 999))[:10]

    os.makedirs(PARITY_PLOT_DIR, exist_ok=True)
    run_label = os.path.basename(pred_path).replace("Supplementary_NPJ_Validation_Predictions_", "").replace(".csv", "")
    run_label_safe = _safe_plot_label(run_label)
    metrics_rows = []

    pretty_names = {
        "Density": "Density",
        "Heat_of_Formation": "Heat of formation",
        "HOMO_LUMO_Gap": "HOMO-LUMO gap",
        "SA_Score": "SA score",
        "VS_max": r"$V_{S,max}$",
        "Sigma2_tot": r"$\sigma^2_{tot}$",
        "Nu": "Nu",
        "Trigger_Bond_Rho": r"Trigger-bond $\rho$",
        "Molecular_Weight": "Molecular weight",
        "Vertical_BDE": "Vertical BDE",
    }

    # Double-column manuscript width with square panels: 5 columns x 2 rows.
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 3.25))
    axes = np.asarray(axes).reshape(-1)
    panel_letters = list("abcdefghij")
    scatter_color = "#2F6C8E"
    line_color = "#222222"

    for idx, target in enumerate(targets):
        true_col = f"True_{target}"
        pred_col = f"FinalPred_{target}"
        pair = df[[true_col, pred_col]].apply(pd.to_numeric, errors="coerce").dropna()
        ax = axes[idx]
        label = pretty_names.get(target, target.replace("_", " "))
        if pair.empty:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center", fontsize=7)
            ax.set_title(label, pad=2)
            ax.set_box_aspect(1)
            continue

        y_true = pair[true_col].to_numpy(dtype=float)
        y_pred = pair[pred_col].to_numpy(dtype=float)
        r2 = r2_score(y_true, y_pred) if len(pair) > 1 else np.nan
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics_rows.append({
            "Target": target,
            "N_validation": len(pair),
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse,
            "Validation_Predictions": pred_path,
        })

        lo = float(np.nanmin([np.min(y_true), np.min(y_pred)]))
        hi = float(np.nanmax([np.max(y_true), np.max(y_pred)]))
        pad = (hi - lo) * 0.055 if hi > lo else 1.0
        lim = (lo - pad, hi + pad)

        ax.scatter(y_true, y_pred, s=7, alpha=0.62, color=scatter_color, linewidths=0, rasterized=True)
        ax.plot(lim, lim, linestyle="--", linewidth=0.7, color=line_color, alpha=0.75)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_title(label, pad=2.0, fontsize=7)
        ax.text(
            0.04,
            0.94,
            panel_letters[idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            fontweight="bold",
        )
        ax.text(
            0.96,
            0.06,
            f"$R^2$={r2:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.8,
        )
        ax.tick_params(length=2.0, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)

    for ax in axes[len(targets):]:
        ax.axis("off")
        ax.set_box_aspect(1)

    fig.supxlabel("True value", fontsize=7, y=0.018)
    fig.supylabel("Predicted value", fontsize=7, x=0.012)
    fig.tight_layout(rect=(0.035, 0.04, 1.0, 0.995), w_pad=0.45, h_pad=0.55)

    combined_png = os.path.join(PARITY_PLOT_DIR, f"Figure_Model_Parity_10D_{run_label_safe}.png")
    combined_pdf = combined_png.replace(".png", ".pdf")
    latest_png = os.path.join(PARITY_PLOT_DIR, "Figure_Model_Parity_10D_latest.png")
    latest_pdf = latest_png.replace(".png", ".pdf")
    fig.savefig(combined_png, dpi=600, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    fig.savefig(latest_png, dpi=600, bbox_inches="tight")
    fig.savefig(latest_pdf, bbox_inches="tight")
    fig.savefig(PARITY_PLOT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    metrics_path = os.path.join(PARITY_PLOT_DIR, f"Figure_Model_Parity_10D_{run_label_safe}_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"Manuscript parity figure saved: {combined_png}")
    print(f"Manuscript parity figure saved: {combined_pdf}")
    print(f"Validation parity metrics saved: {metrics_path}")
    return combined_png

def run_evaluation(args) -> str:
    print("==================================================")
    print("🔍 启动 clean-sanitized final-specialist 模型评估汇总")
    print("==================================================")

    production_metrics = "../results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv"
    seed_summary = "../results/final_model_release/production_v4_10d/final_specialist_10d_bde_sensitivity_classification.csv"

    if not os.path.exists(production_metrics):
        msg = f"❌ 找不到 production metrics: {production_metrics}"
        print(msg)
        return msg

    df = pd.read_csv(production_metrics)
    if "Model_Group" in df.columns:
        df_final = df[df["Model_Group"] == "Final-Specialist-Hybrid-v2"].copy()
        if df_final.empty:
            df_final = df.copy()
    else:
        df_final = df.copy()

    os.makedirs("../results", exist_ok=True)
    out_metrics = "../results/Model_Evaluation_10Target_clean_sanitized.csv"
    df_final.to_csv(out_metrics, index=False)

    report_lines = []
    print()
    print("📊 Production seed7 validation metrics:")
    for _, row in df_final.iterrows():
        target = str(row.get("Target", "Unknown"))
        mae = float(row.get("MAE", float("nan")))
        rmse = float(row.get("RMSE", float("nan")))
        r2 = float(row.get("R2", float("nan")))
        line = f"  {target:<22} | MAE: {mae:<10.6g} | RMSE: {rmse:<10.6g} | R2: {r2:.4f}"
        print(line)
        report_lines.append(line)

    if os.path.exists(seed_summary):
        ssum = pd.read_csv(seed_summary)
        ssum.to_csv("../results/Model_Evaluation_10Target_clean_sanitized_seed_stability_summary.csv", index=False)
        print()
        print("📊 Seed stability summary:")
        print(ssum.to_string(index=False))

    with open("../results/latest_model_eval_10Target_clean_sanitized.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    generate_validation_parity_plots()

    print()
    print(f"📄 模型评估指标已保存: {out_metrics}")
    return "\n".join(report_lines)


def run_eda(args) -> None:
    print("==================================================")
    print("📊 启动高能分子数据集 EDA 分析（10D/10-target final-specialist 版）")
    print("==================================================")

    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(OLD_CSV_PATH):
        print(f"❌ 找不到核心数据集: {OLD_CSV_PATH}")
        return

    df = pd.read_csv(OLD_CSV_PATH)
    existing_cols = [col for col in DEFAULT_TARGET_PROPS if col in df.columns]
    if not existing_cols:
        print("❌ 没有匹配到 10D/10-target 物理特征列。")
        return

    print(f"\n📦 数据集规模: {df.shape[0]} 个分子, {df.shape[1]} 个特征维度")
    print(f"✅ 成功匹配到 {len(existing_cols)} 个有效物理特征。")
    print("\n🔍 核心特征统计学描述:")
    print(df[existing_cols].describe().round(4))

    # 1. Feature distributions.
    print("\n🎨 正在绘制 10D/10-target 特征分布直方图...")
    fig, axes = plt.subplots(2, 5, figsize=(7.1, 3.8))
    axes = axes.flatten()
    for i, col in enumerate(existing_cols[:10]):
        if sns is not None:
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        else:
            axes[i].hist(df[col].dropna(), bins=30)
        axes[i].set_title(col, pad=2, fontsize=6.5)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count", fontsize=6.5)
    for j in range(len(existing_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    dist_path = os.path.join(OUT_DIR, "01_feature_distributions_10D.png")
    plt.savefig(dist_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> 已保存: {dist_path}")

    # 2. Density-HOF map.
    if "Density_calc(g/cm3)" in existing_cols and "Heat_of_Formation(kcal/mol)" in existing_cols:
        print("🎯 正在绘制 Density proxy vs HOF 散点图...")
        plt.figure(figsize=(10, 8))
        cvals = df["SAscore"] if "SAscore" in df.columns else None
        scatter = plt.scatter(
            df["Density_calc(g/cm3)"],
            df["Heat_of_Formation(kcal/mol)"],
            c=cvals,
            alpha=0.70,
            edgecolors="w",
            s=50,
        )
        if cvals is not None:
            plt.colorbar(scatter, label="SAscore (lower is easier)")
        plt.xlabel("Density proxy (g/cm³)")
        plt.ylabel("Heat of Formation (kcal/mol)")
        plt.title("HELS Candidate Map: Density Proxy vs HOF")
        plt.axvline(x=1.8, linestyle="--", alpha=0.5)
        plt.axhline(y=100, linestyle="--", alpha=0.5)
        pareto_path = os.path.join(OUT_DIR, "02_density_hof_map.png")
        plt.savefig(pareto_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  -> 已保存: {pareto_path}")

    # 3. Correlation matrix.
    print("🔗 正在绘制 10D/10-target 特征相关性热力图...")
    plt.figure(figsize=(12, 10))
    corr = df[existing_cols].corr(numeric_only=True)
    if sns is not None:
        sns.heatmap(corr, annot=True, vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
    else:
        plt.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(label="Pearson r")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Feature correlation matrix (10D final-specialist)", pad=15)
    corr_path = os.path.join(OUT_DIR, "03_correlation_matrix_10D.png")
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> 已保存: {corr_path}")

    print("\n🎉 EDA 分析完成。")



# ============================================================
# 2D-only baseline module: RF / XGBoost / MLP
# Added for SCI-review baseline comparison.
# Inputs:
#   ../data/baselines/2d_feature_matrix.csv
#   ../data/baselines/target_matrix_10d.csv
# Outputs:
#   ../results/baselines/baseline_2d_*_metrics.csv
#   ../results/baselines/baseline_summary_10d.csv
# ============================================================

def _baseline_rmse(y_true, y_pred):
    import numpy as _np
    from sklearn.metrics import mean_squared_error as _mse
    return float(_np.sqrt(_mse(y_true, y_pred)))


def _baseline_metrics(y_true, y_pred, target_names, model_group, model_name, train_n, val_n, feature_desc):
    import numpy as _np
    import pandas as _pd
    from sklearn.metrics import mean_absolute_error, r2_score

    rows = []
    for i, target in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "Model_Group": model_group,
            "Model": model_name,
            "Target": target,
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": _baseline_rmse(yt, yp),
            "R2": float(r2_score(yt, yp)),
            "Train_N": int(train_n),
            "Val_N": int(val_n),
            "Split_ID": "row-level train_val_split_10d",
            "Input_Features": feature_desc,
            "Status": "Done",
        })
    return _pd.DataFrame(rows)


def _load_2d_baseline_data():
    import os
    import json
    import numpy as np
    import pandas as pd

    feature_path = "../data/baselines/2d_feature_matrix.csv"
    target_path = "../data/baselines/target_matrix_10d.csv"

    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing 2D feature matrix: {feature_path}")
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Missing target matrix: {target_path}")

    feat = pd.read_csv(feature_path)
    targ = pd.read_csv(target_path)

    meta_cols = {
        "Example_ID", "Molecule", "SMILES", "Split",
        "Row_Index", "Graph_Input_Mode", "Uses_2D_Descriptor_Branch",
    }

    target_names = [
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

    missing = [c for c in target_names if c not in targ.columns]
    if missing:
        raise RuntimeError(f"target_matrix_10d.csv is missing targets: {missing}")

    feature_cols = [c for c in feat.columns if c not in meta_cols]

    # 防止任何目标列被意外放入 2D features
    leak_like = set(target_names) | {
        "Density_calc(g/cm3)",
        "Heat_of_Formation(kcal/mol)",
        "HOMO_LUMO_Gap(eV)",
        "HOMO-LUMO_Gap(eV)",
        "SAscore",
        "SA_Score",
        "VS_max",
        "Sigma2_tot",
        "Trigger_Bond_Rho",
        "Molecular_Weight",
        "Vertical_BDE",
        "Vertical_BDE(kcal/mol)",
    }
    feature_cols = [c for c in feature_cols if c not in leak_like]

    df = feat[["Example_ID", "Molecule", "SMILES", "Split"] + feature_cols].merge(
        targ[["Example_ID"] + target_names],
        on="Example_ID",
        how="inner"
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=target_names)

    train = df[df["Split"].eq("train")].copy()
    val = df[df["Split"].eq("val")].copy()

    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(
            f"Invalid split for baselines: train={len(train)}, val={len(val)}. "
            "Check ../data/baselines/* Split column."
        )

    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_val = val[feature_cols].apply(pd.to_numeric, errors="coerce")

    # Median imputation fitted on train only
    med = X_train.median(axis=0)
    X_train = X_train.fillna(med).fillna(0.0)
    X_val = X_val.fillna(med).fillna(0.0)

    y_train = train[target_names].astype(float).values
    y_val = val[target_names].astype(float).values

    os.makedirs("../results/baselines", exist_ok=True)
    with open("../results/baselines/selected_feature_columns_2d.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "targets": target_names,
            "train_n": int(len(train)),
            "val_n": int(len(val)),
            "note": "Features are Morgan fingerprints plus SMILES-derived descriptors. Quantum target labels are excluded from input features.",
        }, f, indent=2)

    return X_train, X_val, y_train, y_val, target_names, len(train), len(val), len(feature_cols)


def run_2d_baselines(args=None):
    import os
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    print("=" * 70)
    print("📊 启动 2D-only baseline 对照组: RF / XGBoost / MLP")
    print("    target-wise training on shared row-level split")
    print("=" * 70)

    X_train_all, X_val_all, y_train_all, y_val_all, target_names, train_n_all, val_n_all, n_features = _load_2d_baseline_data_no_complete_case()

    print(f"[INFO] Split rows: Train_N = {train_n_all} | Val_N = {val_n_all} | Features = {n_features}")
    print("[INFO] Shared targets:", ", ".join(target_names))

    os.makedirs("../results/baselines", exist_ok=True)

    feature_desc = "Morgan fingerprint + energetic SMILES descriptors"
    all_metrics = []

    def eval_one_model(model_name, build_model):
        rows = []

        print(f"\nTraining 2D-only {model_name} target-wise...")
        for i, target in enumerate(target_names):
            yt_train = y_train_all[:, i]
            yt_val = y_val_all[:, i]

            tr_mask = np.isfinite(yt_train)
            va_mask = np.isfinite(yt_val)

            train_n = int(tr_mask.sum())
            val_n = int(va_mask.sum())

            if train_n < 20 or val_n < 5:
                rows.append({
                    "Model_Group": "2D-only",
                    "Model": model_name,
                    "Target": target,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "R2": np.nan,
                    "Train_N": train_n,
                    "Val_N": val_n,
                    "Split_ID": "row-level train_val_split_10d",
                    "Input_Features": feature_desc,
                    "Status": f"Skipped: insufficient labels for {target}",
                })
                continue

            model = build_model()
            model.fit(X_train_all.iloc[tr_mask], yt_train[tr_mask])
            pred = model.predict(X_val_all.iloc[va_mask])

            mae = float(mean_absolute_error(yt_val[va_mask], pred))
            rmse = float(np.sqrt(mean_squared_error(yt_val[va_mask], pred)))
            r2 = float(r2_score(yt_val[va_mask], pred))

            rows.append({
                "Model_Group": "2D-only",
                "Model": model_name,
                "Target": target,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "Train_N": train_n,
                "Val_N": val_n,
                "Split_ID": "row-level train_val_split_10d",
                "Input_Features": feature_desc,
                "Status": "Done",
            })

        out = pd.DataFrame(rows)
        print(out[["Target", "Train_N", "Val_N", "MAE", "RMSE", "R2"]].to_string(index=False))
        return out

    # RF
    def build_rf():
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    m_rf = eval_one_model("RF", build_rf)
    m_rf.to_csv("../results/baselines/baseline_2d_rf_metrics.csv", index=False)
    all_metrics.append(m_rf)

    # XGBoost
    try:
        from xgboost import XGBRegressor

        def build_xgb():
            return XGBRegressor(
                n_estimators=450,
                max_depth=5,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                n_jobs=16,
                reg_lambda=1.0,
            )

        m_xgb = eval_one_model("XGBoost", build_xgb)
        m_xgb.to_csv("../results/baselines/baseline_2d_xgboost_metrics.csv", index=False)
        all_metrics.append(m_xgb)

    except Exception as e:
        print(f"[WARN] XGBoost baseline skipped: {e}")
        m_xgb = pd.DataFrame([{
            "Model_Group": "2D-only",
            "Model": "XGBoost",
            "Target": t,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Train_N": int(np.isfinite(y_train_all[:, i]).sum()),
            "Val_N": int(np.isfinite(y_val_all[:, i]).sum()),
            "Split_ID": "row-level train_val_split_10d",
            "Input_Features": feature_desc,
            "Status": f"Skipped: {type(e).__name__}: {e}",
        } for i, t in enumerate(target_names)])
        m_xgb.to_csv("../results/baselines/baseline_2d_xgboost_metrics.csv", index=False)
        all_metrics.append(m_xgb)

    # MLP
    def build_mlp():
        pipe = Pipeline([
            ("x_scaler", StandardScaler(with_mean=False)),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=800,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                random_state=42,
                verbose=False,
            )),
        ])
        return TransformedTargetRegressor(
            regressor=pipe,
            transformer=StandardScaler()
        )

    m_mlp = eval_one_model("MLP", build_mlp)
    m_mlp.to_csv("../results/baselines/baseline_2d_mlp_metrics.csv", index=False)
    all_metrics.append(m_mlp)

    summary = pd.concat(all_metrics, ignore_index=True)

    # Hybrid current model
    if os.path.exists(hybrid_path):
        try:
            h = pd.read_csv(hybrid_path)
            if "Target" in h.columns and "R2" in h.columns:
                h2 = h.copy()
                h2["Model_Group"] = "Hybrid"
                h2["Model"] = "3D/2D EGNN"
                if "Train_N" not in h2.columns:
                    h2["Train_N"] = train_n_all
                if "Val_N" not in h2.columns:
                    h2["Val_N"] = val_n_all
                if "Split_ID" not in h2.columns:
                    h2["Split_ID"] = "row-level train_val_split_10d"
                if "Input_Features" not in h2.columns:
                    h2["Input_Features"] = "Atom type + coordinates + graph + 2D descriptors"
                if "Status" not in h2.columns:
                    h2["Status"] = "Current main model"
                h2 = h2.reindex(columns=summary.columns)
                summary = pd.concat([summary, h2], ignore_index=True)
                print(f"\n[INFO] Hybrid metrics appended from {hybrid_path}")
        except Exception as e:
            print(f"[WARN] Failed to append hybrid metrics: {e}")

    # 3D-only placeholder or existing result
    eg3d_path = "../results/baselines/baseline_3d_egnn_only_metrics.csv"
    if os.path.exists(eg3d_path):
        try:
            e3 = pd.read_csv(eg3d_path)
            e3 = e3.reindex(columns=summary.columns)
            summary = pd.concat([summary, e3], ignore_index=True)
        except Exception as e:
            print(f"[WARN] Failed to append 3D-only metrics: {e}")
    else:
        placeholder = pd.DataFrame([{
            "Model_Group": "3D-only",
            "Model": "EGNN",
            "Target": t,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Train_N": train_n_all,
            "Val_N": val_n_all,
            "Split_ID": "row-level train_val_split_10d",
            "Input_Features": "Atom type + coordinates + graph",
            "Status": "Pending: train with 03_egnn_painn_train.py --model_variant 3d_only",
        } for t in target_names])
        summary = pd.concat([summary, placeholder], ignore_index=True)

    summary_path = "../results/baselines/baseline_summary_10d.csv"
    summary.to_csv(summary_path, index=False)

    os.makedirs("../results/tables", exist_ok=True)
    summary.to_csv("../results/tables/Table_Model_Baselines.csv", index=False)

    print("\n" + "=" * 70)
    print(f"✅ Baseline summary saved: {summary_path}")
    print("✅ Paper table saved: ../results/tables/Table_Model_Baselines.csv")
    print("=" * 70)

    try:
        plot_df = summary.dropna(subset=["R2"]).copy()
        if len(plot_df) > 0:
            pivot = plot_df.pivot_table(index="Target", columns="Model", values="R2", aggfunc="first")
            pivot = pivot.reindex(target_names)
            ax = pivot.plot(kind="bar", figsize=(14, 6), width=0.82)
            ax.set_ylabel("Validation R²", fontweight="bold")
            ax.set_xlabel("Target", fontweight="bold")
            ax.set_title("Baseline comparison on the shared row-level validation split", fontweight="bold")
            ax.axhline(0.0, linestyle="--", linewidth=1)
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            out_png = "../results/baselines/baseline_r2_comparison_2d.png"
            out_pdf = "../results/baselines/baseline_r2_comparison_2d.pdf"
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Baseline R² figure saved: {out_png}")
    except Exception as e:
        print(f"[WARN] Baseline plot failed: {e}")

    return summary


def _load_2d_baseline_data_no_complete_case():
    import os
    import json
    import numpy as np
    import pandas as pd

    feature_path = "../data/baselines/2d_feature_matrix.csv"
    target_path = "../data/baselines/target_matrix_10d.csv"

    feat = pd.read_csv(feature_path)
    targ = pd.read_csv(target_path)

    meta_cols = {
        "Example_ID", "Molecule", "SMILES", "Split",
        "Row_Index", "Graph_Input_Mode", "Uses_2D_Descriptor_Branch",
    }

    target_names = [
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

    feature_cols = [c for c in feat.columns if c not in meta_cols]

    leak_like = set(target_names) | {
        "Density_calc(g/cm3)",
        "Heat_of_Formation(kcal/mol)",
        "HOMO_LUMO_Gap(eV)",
        "HOMO-LUMO_Gap(eV)",
        "SAscore",
        "SA_Score",
        "VS_max",
        "Sigma2_tot",
        "Trigger_Bond_Rho",
        "Molecular_Weight",
        "Vertical_BDE",
        "Vertical_BDE(kcal/mol)",
    }
    feature_cols = [c for c in feature_cols if c not in leak_like]

    df = feat[["Example_ID", "Molecule", "SMILES", "Split"] + feature_cols].merge(
        targ[["Example_ID"] + target_names],
        on="Example_ID",
        how="inner"
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    train = df[df["Split"].eq("train")].copy()
    val = df[df["Split"].eq("val")].copy()

    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(f"Invalid split: train={len(train)}, val={len(val)}")

    X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_val = val[feature_cols].apply(pd.to_numeric, errors="coerce")

    med = X_train.median(axis=0)
    X_train = X_train.fillna(med).fillna(0.0)
    X_val = X_val.fillna(med).fillna(0.0)

    y_train = train[target_names].apply(pd.to_numeric, errors="coerce").values
    y_val = val[target_names].apply(pd.to_numeric, errors="coerce").values

    os.makedirs("../results/baselines", exist_ok=True)
    with open("../results/baselines/selected_feature_columns_2d.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "targets": target_names,
            "train_rows": int(len(train)),
            "val_rows": int(len(val)),
            "note": "Target-wise baselines use the same row-level split and train/evaluate each target on non-missing labels only.",
        }, f, indent=2)

    return X_train, X_val, y_train, y_val, target_names, len(train), len(val), len(feature_cols)



# ============================================================
# Synthesizability analysis module
# Outputs:
#   Table_Top20_Chemical_Validity.csv
#   Table_Top20_Synthesizability.csv
#   Figure_SA_Distribution_Top20.png/pdf
#   Table_Top20_ScaffoldSimilarity.csv
#   Figure_Top20_BenchmarkSimilarity_Heatmap.png/pdf
#   Top10_Retrosynthesis_Input.smi/csv
#   Table_Top10_Retrosynthesis_Assessment.csv
# ============================================================

def run_synthesizability_analysis(args=None):
    import os
    import math
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdMolDescriptors, Descriptors

    RDLogger.DisableLog("rdApp.*")

    print("=" * 78)
    print("🧪 启动 Top20 可合成性与逆合成可行性分析")
    print("=" * 78)

    out_dir = "../results/synthesizability"
    os.makedirs(out_dir, exist_ok=True)

    top20_path = "../results/density_calibration/Top20_Density_Calibrated_Detonation_8benchmarks.csv"
    old_path = "../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv"
    benchmark_path = "../data/benchmarks/industrial_benchmarks_8_raw.csv"

    for fp in [top20_path, old_path]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing required file: {fp}")

    top20 = pd.read_csv(top20_path)
    old = pd.read_csv(old_path)

    if "Candidate" not in top20.columns and "Molecule" in top20.columns:
        top20 = top20.rename(columns={"Molecule": "Candidate"})
    if "Candidate" not in top20.columns:
        raise RuntimeError("Top20 table must contain Candidate or Molecule column.")
    if "SMILES" not in top20.columns:
        raise RuntimeError("Top20 table must contain SMILES column.")

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------
    def safe_mol(smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def count_substruct(mol, smarts):
        try:
            patt = Chem.MolFromSmarts(smarts)
            if mol is None or patt is None:
                return 0
            return len(mol.GetSubstructMatches(patt))
        except Exception:
            return 0

    def has_substruct(mol, smarts):
        return count_substruct(mol, smarts) > 0

    def atom_counts(mol):
        counts = {"C": 0, "H": 0, "N": 0, "O": 0}
        if mol is None:
            return counts
        molh = Chem.AddHs(mol)
        for a in molh.GetAtoms():
            sym = a.GetSymbol()
            if sym in counts:
                counts[sym] += 1
        return counts

    def oxygen_balance_1600(C, H, O, MW):
        # OB% = 1600 * (O - 2C - H/2) / MW
        try:
            if MW <= 0:
                return np.nan
            return 1600.0 * (O - 2.0 * C - H / 2.0) / MW
        except Exception:
            return np.nan

    def sa_class(score):
        try:
            s = float(score)
        except Exception:
            return "unknown"
        if s <= 3.0:
            return "relatively accessible"
        if s <= 5.0:
            return "moderate synthetic accessibility"
        if s <= 6.0:
            return "challenging but not excluded"
        return "low synthetic accessibility"

    def get_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # --------------------------------------------------------
    # Merge SA score from old_dataset
    # --------------------------------------------------------
    sa_col = "SAscore" if "SAscore" in old.columns else ("SA_Score" if "SA_Score" in old.columns else None)
    if sa_col is None:
        raise RuntimeError("old_dataset.csv lacks SAscore / SA_Score column.")

    merge_cols = []
    if "Molecule" in old.columns:
        merge_cols.append("Molecule")
    if "SMILES" in old.columns:
        merge_cols.append("SMILES")

    top20["SA_Score"] = np.nan

    if "Molecule" in old.columns:
        lookup = old[["Molecule", sa_col]].dropna().drop_duplicates("Molecule")
        tmp = top20.merge(lookup, left_on="Candidate", right_on="Molecule", how="left")
        if sa_col in tmp.columns:
            top20["SA_Score"] = pd.to_numeric(tmp[sa_col], errors="coerce")

    if top20["SA_Score"].isna().any() and "SMILES" in old.columns:
        lookup = old[["SMILES", sa_col]].dropna().drop_duplicates("SMILES")
        tmp = top20[["SMILES"]].merge(lookup, on="SMILES", how="left")
        top20.loc[top20["SA_Score"].isna(), "SA_Score"] = pd.to_numeric(
            tmp.loc[top20["SA_Score"].isna(), sa_col],
            errors="coerce"
        ).values

    # --------------------------------------------------------
    # Chemical validity and functional-group flags
    # --------------------------------------------------------
    validity_rows = []

    for _, row in top20.iterrows():
        name = str(row["Candidate"])
        smiles = str(row["SMILES"])
        mol = safe_mol(smiles)

        valid = mol is not None
        counts = atom_counts(mol)
        C, H, N, O = counts["C"], counts["H"], counts["N"], counts["O"]
        MW = float(Descriptors.ExactMolWt(mol)) if mol is not None else np.nan
        formal_charge = int(Chem.GetFormalCharge(mol)) if mol is not None else np.nan

        n_c_ratio = float(N / C) if C > 0 else np.inf
        ob = oxygen_balance_1600(C, H, O, MW) if np.isfinite(MW) else np.nan

        # Core energetic functional groups
        num_nitro = count_substruct(mol, "[N+](=O)[O-]")
        num_c_no2 = count_substruct(mol, "[#6]-[N+](=O)[O-]")
        num_n_no2 = count_substruct(mol, "[#7]-[N+](=O)[O-]")
        num_n_eq_n = count_substruct(mol, "[#7]=[#7]")

        # Risk flags
        has_peroxide = has_substruct(mol, "[O]-[O]")
        has_azide = has_substruct(mol, "[N-]=[N+]=N") or has_substruct(mol, "N=[N+]=[N-]")
        has_diazo = has_substruct(mol, "[#6]=[N+]=[N-]") or has_substruct(mol, "[#6-]-[N+]#N")
        has_long_nn_chain = has_substruct(mol, "[#7]-[#7]-[#7]-[#7]")

        forbidden = bool(has_peroxide or has_azide or has_diazo or has_long_nn_chain or (formal_charge != 0) or (n_c_ratio > 3.0))

        status = "pass"
        if not valid:
            status = "invalid_RDKit"
        elif forbidden:
            status = "flagged"
        elif n_c_ratio > 2.5:
            status = "caution_high_NC"
        else:
            status = "pass"

        validity_rows.append({
            "Candidate": name,
            "SMILES": smiles,
            "RDKit_Valid": bool(valid),
            "Formal_Charge": formal_charge,
            "C": C,
            "H": H,
            "N": N,
            "O": O,
            "MW": MW,
            "N_C_Ratio": n_c_ratio,
            "O_Balance_percent": ob,
            "Num_Nitro": num_nitro,
            "Num_C_NO2": num_c_no2,
            "Num_N_NO2": num_n_no2,
            "Num_N_eq_N": num_n_eq_n,
            "Has_Peroxide": bool(has_peroxide),
            "Has_Azide": bool(has_azide),
            "Has_Diazo": bool(has_diazo),
            "Has_Long_NN_Chain": bool(has_long_nn_chain),
            "Forbidden_Group_Flag": bool(forbidden),
            "Chemical_Validity_Status": status,
        })

    validity = pd.DataFrame(validity_rows)

    # --------------------------------------------------------
    # Synthesizability table
    # --------------------------------------------------------
    synth = top20.copy()
    synth = synth.merge(validity.drop(columns=["SMILES"]), on="Candidate", how="left")
    synth["SA_Score"] = pd.to_numeric(synth["SA_Score"], errors="coerce")
    synth["SA_Class"] = synth["SA_Score"].apply(sa_class)

    # Normalize common calibrated columns
    rename_map = {
        "ρ_proxy(g/cm3)": "rho_proxy(g/cm3)",
        "ρ_calibrated(g/cm3)": "rho_calibrated(g/cm3)",
        "D_proxy(km/s)": "D_proxy(km/s)",
        "D_calibrated(km/s)": "D_calibrated(km/s)",
        "P_proxy(GPa)": "P_proxy(GPa)",
        "P_calibrated(GPa)": "P_calibrated(GPa)",
    }
    synth = synth.rename(columns={k: v for k, v in rename_map.items() if k in synth.columns})

    validity_out = os.path.join(out_dir, "Table_Top20_Chemical_Validity.csv")
    synth_out = os.path.join(out_dir, "Table_Top20_Synthesizability.csv")

    validity.to_csv(validity_out, index=False)

    synth_cols = [
        "Calibrated_Rank",
        "Candidate",
        "SMILES",
        "SA_Score",
        "SA_Class",
        "rho_proxy(g/cm3)",
        "rho_calibrated(g/cm3)",
        "D_proxy(km/s)",
        "D_calibrated(km/s)",
        "P_proxy(GPa)",
        "P_calibrated(GPa)",
        "Chemical_Validity_Status",
        "Forbidden_Group_Flag",
        "N_C_Ratio",
        "O_Balance_percent",
        "Num_Nitro",
        "Num_C_NO2",
        "Num_N_NO2",
        "Num_N_eq_N",
    ]
    synth_cols = [c for c in synth_cols if c in synth.columns]
    synth[synth_cols].to_csv(synth_out, index=False)

    print(f"[OK] Saved chemical validity table: {validity_out}")
    print(f"[OK] Saved synthesizability table: {synth_out}")

    # --------------------------------------------------------
    # SA score distribution figure
    # --------------------------------------------------------
    all_sa = pd.to_numeric(old[sa_col], errors="coerce").dropna()
    top_sa = pd.to_numeric(synth["SA_Score"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.hist(all_sa, bins=35, alpha=0.65, density=True, label="Training/screening set")
    ax.hist(top_sa, bins=12, alpha=0.70, density=True, label="Top20 calibrated candidates")

    if len(top_sa) > 0:
        ax.axvline(top_sa.median(), linestyle="--", linewidth=1.4, label=f"Top20 median = {top_sa.median():.2f}")

    ax.set_xlabel("Synthetic accessibility score")
    ax.set_ylabel("Density")
    ax.set_title("SA-score distribution of top calibrated candidates")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    fig_png = os.path.join(out_dir, "Figure_SA_Distribution_Top20.png")
    fig_pdf = os.path.join(out_dir, "Figure_SA_Distribution_Top20.pdf")
    plt.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Saved SA distribution figure: {fig_png}")
    print(f"[OK] Saved SA distribution figure: {fig_pdf}")

    # --------------------------------------------------------
    # Benchmark scaffold similarity
    # --------------------------------------------------------
    benchmark_smiles = {
        "TNT": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
        "RDX": "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
        "HMX": "C1N([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]",
        "epsilon-CL-20": "C12N3CN4CN1CN(C2)CN(C3)C4",
        "PETN": "C(CO[N+](=O)[O-])(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
        "TATB": "Nc1c([N+](=O)[O-])c(N)c([N+](=O)[O-])c(N)c1[N+](=O)[O-]",
        "FOX-7": "NC(=C(N)[N+](=O)[O-])[N+](=O)[O-]",
        "NTO": "O=c1[nH]nc([N+](=O)[O-])n1",
    }

    scaffold_class = {
        "TNT": "nitroaromatic",
        "RDX": "nitramine",
        "HMX": "nitramine",
        "epsilon-CL-20": "caged nitramine",
        "PETN": "nitrate ester",
        "TATB": "insensitive nitroaromatic",
        "FOX-7": "push-pull nitroalkene",
        "NTO": "nitro heterocycle",
    }

    def fp_from_smiles(smiles):
        mol = safe_mol(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

    bench_fps = {}
    for name, smi in benchmark_smiles.items():
        fp = fp_from_smiles(smi)
        if fp is not None:
            bench_fps[name] = fp

    sim_rows = []
    heatmap_rows = []

    for _, row in synth.iterrows():
        cand = str(row["Candidate"])
        smiles = str(row["SMILES"])
        fp = fp_from_smiles(smiles)

        sim_dict = {}
        if fp is not None:
            for bname, bfp in bench_fps.items():
                sim_dict[bname] = float(DataStructs.TanimotoSimilarity(fp, bfp))
        else:
            sim_dict = {bname: np.nan for bname in bench_fps}

        sorted_hits = sorted(sim_dict.items(), key=lambda x: (-999 if pd.isna(x[1]) else x[1]), reverse=True)
        nearest_name, nearest_sim = sorted_hits[0] if sorted_hits else ("NA", np.nan)

        if pd.isna(nearest_sim):
            plaus = "unresolved"
        elif nearest_sim >= 0.50:
            plaus = "close to known energetic scaffold"
        elif nearest_sim >= 0.30:
            plaus = "moderately related scaffold"
        else:
            plaus = "novel scaffold; requires retrosynthetic caution"

        top3 = "; ".join([f"{k}:{v:.3f}" for k, v in sorted_hits[:3]])

        sim_rows.append({
            "Candidate": cand,
            "SMILES": smiles,
            "Nearest_Benchmark": nearest_name,
            "Nearest_Benchmark_Tanimoto": nearest_sim,
            "Nearest_Scaffold_Class": scaffold_class.get(nearest_name, "unknown"),
            "Top3_Benchmark_Matches": top3,
            "Scaffold_Plausibility_Status": plaus,
        })

        hm = {"Candidate": cand}
        hm.update(sim_dict)
        heatmap_rows.append(hm)

    sim_df = pd.DataFrame(sim_rows)
    heatmap_df = pd.DataFrame(heatmap_rows)

    sim_out = os.path.join(out_dir, "Table_Top20_ScaffoldSimilarity.csv")
    sim_df.to_csv(sim_out, index=False)
    print(f"[OK] Saved scaffold similarity table: {sim_out}")

    # Heatmap
    if len(heatmap_df) > 0:
        bench_cols = [c for c in heatmap_df.columns if c != "Candidate"]
        matrix = heatmap_df[bench_cols].astype(float).values

        fig, ax = plt.subplots(figsize=(8.8, 7.0))
        im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=np.nanmax(matrix) if np.isfinite(matrix).any() else 1.0)

        ax.set_xticks(np.arange(len(bench_cols)))
        ax.set_xticklabels(bench_cols, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(heatmap_df)))
        ax.set_yticklabels(heatmap_df["Candidate"].tolist(), fontsize=8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Morgan fingerprint Tanimoto similarity")

        ax.set_title("Top20 similarity to industrial energetic benchmarks")
        fig.tight_layout()

        heat_png = os.path.join(out_dir, "Figure_Top20_BenchmarkSimilarity_Heatmap.png")
        heat_pdf = os.path.join(out_dir, "Figure_Top20_BenchmarkSimilarity_Heatmap.pdf")
        plt.savefig(heat_png, dpi=300, bbox_inches="tight")
        plt.savefig(heat_pdf, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[OK] Saved scaffold similarity heatmap: {heat_png}")
        print(f"[OK] Saved scaffold similarity heatmap: {heat_pdf}")

    # --------------------------------------------------------
    # Retrosynthesis input and rule-based assessment
    # --------------------------------------------------------
    top10 = synth.sort_values("Calibrated_Rank").head(10).copy()

    retrosmi = os.path.join(out_dir, "Top10_Retrosynthesis_Input.smi")
    retrocsv = os.path.join(out_dir, "Top10_Retrosynthesis_Input.csv")

    with open(retrosmi, "w", encoding="utf-8") as f:
        for _, r in top10.iterrows():
            f.write(f"{r['SMILES']} {r['Candidate']}\\n")

    top10[["Candidate", "SMILES"]].to_csv(retrocsv, index=False)

    retro_rows = []
    sim_lookup = sim_df.set_index("Candidate").to_dict(orient="index")

    def infer_core_scaffold(row, sim_info):
        nearest = sim_info.get("Nearest_Benchmark", "NA") if sim_info else "NA"
        cls = sim_info.get("Nearest_Scaffold_Class", "unknown") if sim_info else "unknown"

        if int(row.get("Num_N_NO2", 0)) > 0:
            return "nitramine/nitroamino-rich heterocycle", cls
        if int(row.get("Num_C_NO2", 0)) > 0:
            return "C-nitro heterocycle or nitroaromatic-like motif", cls
        if int(row.get("Num_N_eq_N", 0)) > 0:
            return "azo/hydrazone-containing N-rich motif", cls
        return "N/O-rich heterocyclic motif", cls

    for _, row in top10.iterrows():
        cand = row["Candidate"]
        sim_info = sim_lookup.get(cand, {})
        nearest_sim = sim_info.get("Nearest_Benchmark_Tanimoto", np.nan)

        core, known_cls = infer_core_scaffold(row, sim_info)
        sa = row.get("SA_Score", np.nan)
        validity_status = row.get("Chemical_Validity_Status", "unknown")

        risk_items = []
        if bool(row.get("Forbidden_Group_Flag", False)):
            risk_items.append("hard structural flag")
        if pd.notna(sa) and float(sa) > 5.0:
            risk_items.append("high SA score")
        if pd.notna(nearest_sim) and float(nearest_sim) < 0.30:
            risk_items.append("low similarity to benchmark scaffolds")
        if float(row.get("N_C_Ratio", 0)) > 2.5:
            risk_items.append("high N/C ratio")

        if not risk_items:
            risk = "no major rule-based risk flag"
        else:
            risk = "; ".join(risk_items)

        if validity_status == "pass" and pd.notna(sa) and float(sa) <= 5.0:
            if pd.notna(nearest_sim) and float(nearest_sim) >= 0.30:
                assessment = "prioritize for retrosynthesis query"
            else:
                assessment = "chemically valid but scaffold novelty requires route validation"
        elif validity_status == "pass":
            assessment = "valid structure; synthetic difficulty should be checked by retrosynthesis"
        else:
            assessment = "requires caution before route planning"

        if int(row.get("Num_N_NO2", 0)) > 0:
            disconnection = "late-stage nitramination/nitroamino functionalization of a preformed heterocycle"
            precursor = "aminated or hydrazino N-rich heterocyclic precursor"
        elif int(row.get("Num_C_NO2", 0)) > 0:
            disconnection = "late-stage nitration or C-nitro functionalization"
            precursor = "activated heteroaromatic or N/O-rich ring precursor"
        else:
            disconnection = "construction of N/O-rich heterocyclic core followed by oxidation/nitration"
            precursor = "preformed N/O-rich heterocycle"

        retro_rows.append({
            "Candidate": cand,
            "SMILES": row["SMILES"],
            "Calibrated_Rank": row.get("Calibrated_Rank", np.nan),
            "SA_Score": sa,
            "Core_Scaffold": core,
            "Nearest_Benchmark": sim_info.get("Nearest_Benchmark", "NA"),
            "Nearest_Benchmark_Tanimoto": nearest_sim,
            "Known_Motif": known_cls,
            "Suggested_Disconnection": disconnection,
            "Plausible_Precursor_Class": precursor,
            "Main_Synthetic_Risk": risk,
            "Assessment": assessment,
            "Manual_Comment": "Rule-based retrosynthetic plausibility only; detailed route planning with ASKCOS/AiZynthFinder or experimental validation is still required.",
        })

    retro_df = pd.DataFrame(retro_rows)
    retro_out = os.path.join(out_dir, "Table_Top10_Retrosynthesis_Assessment.csv")
    retro_df.to_csv(retro_out, index=False)

    print(f"[OK] Saved retrosynthesis input: {retrosmi}")
    print(f"[OK] Saved retrosynthesis input CSV: {retrocsv}")
    print(f"[OK] Saved rule-based retrosynthesis assessment: {retro_out}")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\\n" + "=" * 78)
    print("📌 Top20 synthesizability summary")
    print("=" * 78)
    print("Top20 SA_Score median:", float(top_sa.median()) if len(top_sa) else "NA")
    print("Top20 SA_Score min/max:", (float(top_sa.min()), float(top_sa.max())) if len(top_sa) else "NA")
    print("Chemical validity counts:")
    print(validity["Chemical_Validity_Status"].value_counts(dropna=False).to_string())
    print("\\nNearest benchmark counts:")
    print(sim_df["Nearest_Benchmark"].value_counts(dropna=False).to_string())
    print("=" * 78)

def main() -> None:
    parser = argparse.ArgumentParser(description="10-target final-specialist evaluation and EDA for the manual HELS workflow.")
    parser.add_argument("--mode", choices=["eval", "eda", "all", "baselines", "synth"], default="all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--noise_grid", type=str, default="0,0.005,0.01,0.02", help="Comma-separated coordinate-noise std values in Angstrom for ablation.")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_plots", action="store_true")
    args = parser.parse_args()

    if getattr(args, "mode", None) == "synth":
        run_synthesizability_analysis(args)
        return

    if getattr(args, "mode", None) == "baselines":
        run_2d_baselines(args)
        return

    if args.mode in ("eval", "all"):
        run_evaluation(args)
    if args.mode in ("eda", "all"):
        run_eda(args)


if __name__ == "__main__":
    main()

