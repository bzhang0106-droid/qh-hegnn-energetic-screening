from __future__ import annotations

from pathlib import Path
import json
import math
import re
import shutil
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
TARGET_MATRIX = ROOT / "data/curated_molecule_clean_v1/target_matrix_10d_molecule_clean.csv"
XTB = ROOT / "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv"
OUT = ROOT / "results/model_optimization/final_leakage_masking_20260607"
PKG = ROOT / "manuscript_npJ/final_submission_package_AL08_20260605"
SI_TABLES = PKG / "si_tables"
SI_FIGURES = PKG / "si_figures"
INTERNAL = PKG / "internal_audit"
CODE_RELEASE = PKG / "code_release"
MAJOR = ROOT / "manuscript_npJ/major_revision_20260607"

for path in [OUT, SI_TABLES, SI_FIGURES, INTERNAL, CODE_RELEASE, MAJOR]:
    path.mkdir(parents=True, exist_ok=True)


TARGETS = {
    "Density": "Density",
    "Heat_of_Formation": "Heat_of_Formation",
    "HOMO_LUMO_Gap": "HOMO_LUMO_Gap",
    "SA_Score": "SA_Score",
    "VS_max": "VS_max",
    "Sigma2_tot": "Sigma2_tot",
    "Nu": "Nu",
    "Trigger_Bond_Rho": "Trigger_Bond_Rho",
    "Molecular_Weight": "Molecular_Weight",
    "Vertical_BDE": "Vertical_BDE",
}

TARGET_ALIASES = {
    "Density": ["density", "rho_cal", "rho_proxy", "volume"],
    "Heat_of_Formation": ["heat_of_formation", "hof", "formation", "enthalpy", "total_energy"],
    "HOMO_LUMO_Gap": ["homo", "lumo", "gap"],
    "SA_Score": ["sa_score", "sascore", "sasc", "scscore", "syba", "synth", "pubchem", "route", "askcos", "aizynth"],
    "VS_max": ["vs_max", "esp", "surface_potential", "v_s"],
    "Sigma2_tot": ["sigma2", "sigma", "esp", "surface_potential"],
    "Nu": ["nu", "balance", "esp", "surface_potential"],
    "Trigger_Bond_Rho": ["trigger_bond_rho", "rho", "bcp", "critic", "trigger"],
    "Molecular_Weight": ["molecular_weight", "molwt", "exactmolwt", "mass", "mw", "n_atoms", "numheavy", "count_", "_count"],
    "Vertical_BDE": ["vertical_bde", "bde", "bond", "wbo", "trigger"],
}

ALL_LABEL_TERMS = [
    "density_calc",
    "density_calibrated",
    "density",
    "heat_of_formation",
    "homo_lumo_gap",
    "sascore",
    "sa_score",
    "vs_max",
    "sigma2_tot",
    "nu",
    "trigger_bond_rho",
    "molecular_weight",
    "vertical_bde",
    "final_detonation",
    "oracle",
]

SEEDS = [42, 7, 123]


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def count_smarts(mol: Chem.Mol | None, smarts: str) -> int:
    if mol is None:
        return 0
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def mol_from_smiles(smi: object) -> Chem.Mol | None:
    if pd.isna(smi):
        return None
    try:
        return Chem.MolFromSmiles(str(smi))
    except Exception:
        return None


def rdkit_features(smiles: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    scaffolds = []
    for smi in smiles.astype(str).fillna(""):
        mol = mol_from_smiles(smi)
        if mol is None:
            rows.append({})
            scaffolds.append("")
            continue
        atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
        c = sum(z == 6 for z in atoms)
        n = sum(z == 7 for z in atoms)
        o = sum(z == 8 for z in atoms)
        h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
        heavy = max(mol.GetNumHeavyAtoms(), 1)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""
        scaffolds.append(scaffold)
        desc = {
            "rdkit_exact_mol_wt": Descriptors.ExactMolWt(mol),
            "rdkit_mol_logp": Crippen.MolLogP(mol),
            "rdkit_tpsa": rdMolDescriptors.CalcTPSA(mol),
            "rdkit_num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "rdkit_num_heteroatoms": Lipinski.NumHeteroatoms(mol),
            "rdkit_num_rings": rdMolDescriptors.CalcNumRings(mol),
            "rdkit_num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "rdkit_num_aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
            "rdkit_fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "rdkit_count_c": c,
            "rdkit_count_h": h,
            "rdkit_count_n": n,
            "rdkit_count_o": o,
            "rdkit_n_to_c_ratio": n / max(c, 1),
            "rdkit_o_to_c_ratio": o / max(c, 1),
            "rdkit_num_c_no2": count_smarts(mol, "[#6]-[N+](=O)[O-]"),
            "rdkit_num_n_no2": count_smarts(mol, "[#7]-[N+](=O)[O-]"),
            "rdkit_num_o_no2": count_smarts(mol, "[#8]-[N+](=O)[O-]"),
            "rdkit_num_n_n_single": count_smarts(mol, "[#7]-[#7]"),
            "rdkit_num_n_o_single": count_smarts(mol, "[#7]-[#8]"),
            "rdkit_num_azide": count_smarts(mol, "[N-]=[N+]=N"),
            "rdkit_num_furazan_like": count_smarts(mol, "o1nncc1"),
            "rdkit_num_tetrazole_like": count_smarts(mol, "n1nnnc1"),
        }
        desc["rdkit_explosophore_count"] = desc["rdkit_num_c_no2"] + desc["rdkit_num_n_no2"] + desc["rdkit_num_o_no2"] + desc["rdkit_num_azide"]
        desc["rdkit_trigger_linkage_count"] = desc["rdkit_num_n_n_single"] + desc["rdkit_num_n_o_single"]
        desc["rdkit_explosophore_per_heavy_atom"] = desc["rdkit_explosophore_count"] / heavy
        rows.append(desc)
    return pd.DataFrame(rows), pd.Series(scaffolds, name="murcko_scaffold")


def scaffold_split(scaffolds: pd.Series, seed: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = pd.Series(scaffolds.fillna("").astype(str)).copy()
    # Empty scaffolds are grouped by row index to avoid placing all acyclic molecules in one giant group.
    empty = groups.eq("")
    groups.loc[empty] = ["acyclic_" + str(i) for i in np.where(empty)[0]]
    unique = groups.drop_duplicates().to_numpy()
    rng.shuffle(unique)
    test_target = int(math.ceil(len(groups) * test_frac))
    test_groups = set()
    count = 0
    for g in unique:
        test_groups.add(g)
        count += int((groups == g).sum())
        if count >= test_target:
            break
    test = groups.isin(test_groups).to_numpy()
    train = ~test
    return train, test


def random_split(n: int, seed: int, test_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(math.ceil(n * test_frac))
    test_idx = idx[:n_test]
    test = np.zeros(n, dtype=bool)
    test[test_idx] = True
    return ~test, test


def select_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric = []
    for col in frame.columns:
        if col.startswith("__"):
            continue
        c_norm = norm(col)
        if any(term in c_norm for term in ["row_index", "example_id", "smiles", "workdir", "status", "source", "dir"]):
            continue
        if any(term == c_norm or c_norm.endswith("_" + term) for term in [norm(t) for t in TARGETS.values()]):
            continue
        if any(term in c_norm for term in ALL_LABEL_TERMS):
            # Exact labels and directly derived final detonation labels are never used
            # in the base feature set.
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            numeric.append(col)
    return numeric


def masked_columns_for_target(columns: Iterable[str], target: str) -> list[str]:
    terms = [norm(x) for x in TARGET_ALIASES[target]]
    masked = []
    for col in columns:
        c = norm(col)
        if any(term in c for term in terms):
            masked.append(col)
    return masked


def fit_eval(X: pd.DataFrame, y: pd.Series, train: np.ndarray, test: np.ndarray, seed: int) -> dict[str, float]:
    valid = y.notna().to_numpy() & np.isfinite(pd.to_numeric(y, errors="coerce").to_numpy())
    train = train & valid
    test = test & valid
    if train.sum() < 100 or test.sum() < 30:
        return {"Train_N": int(train.sum()), "Test_N": int(test.sum()), "R2": np.nan, "MAE": np.nan, "RMSE": np.nan}
    Xn = X.replace([np.inf, -np.inf], np.nan)
    y_num = pd.to_numeric(y, errors="coerce")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=350,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    )
    model.fit(Xn.loc[train], y_num.loc[train])
    pred = model.predict(Xn.loc[test])
    true = y_num.loc[test].to_numpy()
    return {
        "Train_N": int(train.sum()),
        "Test_N": int(test.sum()),
        "R2": float(r2_score(true, pred)),
        "MAE": float(mean_absolute_error(true, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
    }


def main() -> None:
    tm = pd.read_csv(TARGET_MATRIX)
    xtb = pd.read_csv(XTB)
    if len(tm) != len(xtb):
        raise RuntimeError(f"Target matrix and xTB row counts differ: {len(tm)} vs {len(xtb)}")
    aligned_molecule_match = float((tm["Molecule"].astype(str).to_numpy() == xtb["Molecule"].astype(str).to_numpy()).mean())
    aligned_smiles_match = float((tm["_canonical_smiles_noiso"].astype(str).to_numpy() == xtb["_training_canonical_smiles_noiso"].astype(str).to_numpy()).mean()) if "_training_canonical_smiles_noiso" in xtb else np.nan

    rdkit, scaffolds = rdkit_features(tm["SMILES"])
    feature_frame = pd.concat(
        [
            tm.add_prefix("tm__"),
            xtb.add_prefix("xtb__"),
            rdkit.add_prefix("rdkit__"),
        ],
        axis=1,
    )
    feature_cols = select_feature_columns(feature_frame)

    rows = []
    mask_rows = []
    for target, label in TARGETS.items():
        if label not in tm.columns:
            continue
        y = pd.to_numeric(tm[label], errors="coerce")
        full_cols = list(feature_cols)
        mask_cols = masked_columns_for_target(full_cols, target)
        masked_cols = [c for c in full_cols if c not in set(mask_cols)]
        mask_rows.append({
            "Target": target,
            "Label_Column": label,
            "Homologous_Mask_Terms": "; ".join(TARGET_ALIASES[target]),
            "N_Full_Features": len(full_cols),
            "N_Masked_Features": len(masked_cols),
            "N_Removed_Features": len(mask_cols),
            "Removed_Features": "; ".join(mask_cols[:120]),
            "Removed_Features_Truncated": len(mask_cols) > 120,
        })
        for seed in SEEDS:
            split_defs = {
                "random_80_20": random_split(len(tm), seed),
                "murcko_scaffold_80_20": scaffold_split(scaffolds, seed),
            }
            for split_name, (train, test) in split_defs.items():
                for feature_set, cols in [("full_nonlabel_feature_pool", full_cols), ("homologous_masked_feature_pool", masked_cols)]:
                    metrics = fit_eval(feature_frame[cols], y, train, test, seed)
                    rows.append({
                        "Target": target,
                        "Label_Column": label,
                        "Split": split_name,
                        "Seed": seed,
                        "Feature_Set": feature_set,
                        "N_Features": len(cols),
                        **metrics,
                    })

    raw = pd.DataFrame(rows)
    masks = pd.DataFrame(mask_rows)
    summary = (
        raw.groupby(["Target", "Split", "Feature_Set"], as_index=False)
        .agg(
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MAE_mean=("MAE", "mean"),
            RMSE_mean=("RMSE", "mean"),
            Train_N_mean=("Train_N", "mean"),
            Test_N_mean=("Test_N", "mean"),
            N_Features=("N_Features", "mean"),
        )
    )
    full = summary[summary["Feature_Set"].eq("full_nonlabel_feature_pool")].rename(columns={"R2_mean": "Full_R2_mean", "MAE_mean": "Full_MAE_mean", "RMSE_mean": "Full_RMSE_mean"})
    masked = summary[summary["Feature_Set"].eq("homologous_masked_feature_pool")].rename(columns={"R2_mean": "Masked_R2_mean", "MAE_mean": "Masked_MAE_mean", "RMSE_mean": "Masked_RMSE_mean"})
    delta = full.merge(masked, on=["Target", "Split"], suffixes=("_full", "_masked"))
    delta["Delta_R2_Masked_minus_Full"] = delta["Masked_R2_mean"] - delta["Full_R2_mean"]
    delta["Delta_MAE_Masked_minus_Full"] = delta["Masked_MAE_mean"] - delta["Full_MAE_mean"]
    delta["Leakage_Risk_Interpretation"] = np.where(
        delta["Delta_R2_Masked_minus_Full"] < -0.15,
        "large performance drop after homologous masking",
        np.where(delta["Delta_R2_Masked_minus_Full"] < -0.05, "moderate drop after homologous masking", "robust to homologous masking in this surrogate audit"),
    )

    raw_path = OUT / "Table_NPJ_TargetWise_Leakage_Masking_Ablation_Raw_20260607.csv"
    masks_path = OUT / "Table_NPJ_TargetWise_Feature_Mask_Rules_20260607.csv"
    summary_path = OUT / "Table_NPJ_TargetWise_Leakage_Masking_Ablation_Summary_20260607.csv"
    delta_path = OUT / "Table_NPJ_TargetWise_Leakage_Masking_Ablation_Delta_20260607.csv"
    raw.to_csv(raw_path, index=False)
    masks.to_csv(masks_path, index=False)
    summary.to_csv(summary_path, index=False)
    delta.to_csv(delta_path, index=False)

    plot = delta[delta["Split"].eq("murcko_scaffold_80_20")].copy()
    order = list(TARGETS.keys())
    plot["Target"] = pd.Categorical(plot["Target"], categories=order, ordered=True)
    plot = plot.sort_values("Target")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = ["#BB5566" if v < -0.05 else "#4477AA" for v in plot["Delta_R2_Masked_minus_Full"]]
    ax.bar(plot["Target"].astype(str), plot["Delta_R2_Masked_minus_Full"], color=colors)
    ax.axhline(0, color="black", lw=0.7)
    ax.axhline(-0.05, color="#BB5566", lw=0.8, ls="--")
    ax.set_ylabel("Delta R2, masked - full", fontsize=8)
    ax.set_title("Target-wise homologous feature masking under scaffold split", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_png = OUT / "Figure_NPJ_TargetWise_Leakage_Masking_Delta_20260607.png"
    fig_pdf = OUT / "Figure_NPJ_TargetWise_Leakage_Masking_Delta_20260607.pdf"
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    status = {
        "generated_at": "2026-06-07",
        "target_matrix": str(TARGET_MATRIX.relative_to(ROOT)),
        "xtb_aligned": str(XTB.relative_to(ROOT)),
        "n_rows": int(len(tm)),
        "n_full_features": int(len(feature_cols)),
        "aligned_molecule_match_fraction": aligned_molecule_match,
        "aligned_canonical_smiles_match_fraction": aligned_smiles_match,
        "seeds": SEEDS,
        "splits": ["random_80_20", "murcko_scaffold_80_20"],
        "model": "HistGradientBoostingRegressor tabular surrogate",
        "claim_boundary": "This is a target-wise homologous feature masking audit for leakage risk. It is not a full QH-HEGNN retraining replacement.",
    }
    status_path = OUT / "NPJ_TargetWise_Leakage_Masking_Status_20260607.md"
    lines = [
        "# Target-Wise Homologous Feature Masking Ablation",
        "",
        "Generated: 2026-06-07",
        "",
        "## Scope",
        "",
        "This audit retrains tabular surrogate regressors on the final frozen 5432-row target matrix and aligned xTB descriptors after removing target-homologous input features. It is designed to address leakage-risk controls and should be reported as a leakage diagnostic, not as a replacement for the production QH-HEGNN model.",
        "",
        "## Inputs",
        "",
        f"- Target matrix: {TARGET_MATRIX.relative_to(ROOT)}",
        f"- xTB aligned features: {XTB.relative_to(ROOT)}",
        f"- Row count: {len(tm)}",
        f"- Molecule row-order match fraction: {aligned_molecule_match:.4f}",
        f"- Canonical SMILES row-order match fraction: {aligned_smiles_match:.4f}",
        "",
        "## Outputs",
        "",
        f"- Raw metrics: {raw_path.relative_to(ROOT)}",
        f"- Mask rules: {masks_path.relative_to(ROOT)}",
        f"- Summary: {summary_path.relative_to(ROOT)}",
        f"- Delta table: {delta_path.relative_to(ROOT)}",
        f"- Figure: {fig_png.relative_to(ROOT)}",
        "",
        "## Key Scaffold-Split Delta R2",
        "",
        delta[delta["Split"].eq("murcko_scaffold_80_20")][["Target", "Full_R2_mean", "Masked_R2_mean", "Delta_R2_Masked_minus_Full", "Leakage_Risk_Interpretation"]].to_markdown(index=False),
        "",
    ]
    status_path.write_text("\n".join(lines), encoding="utf-8")

    for p in [raw_path, masks_path, summary_path, delta_path]:
        shutil.copy2(p, SI_TABLES / p.name)
        shutil.copy2(p, MAJOR / p.name)
    for p in [fig_png, fig_pdf]:
        shutil.copy2(p, SI_FIGURES / p.name)
        shutil.copy2(p, MAJOR / p.name)
    shutil.copy2(status_path, INTERNAL / status_path.name)
    shutil.copy2(status_path, MAJOR / status_path.name)
    script_path = ROOT / "scripts/run_npj_targetwise_leakage_masking_ablation_20260607.py"
    if script_path.exists():
        shutil.copy2(script_path, CODE_RELEASE / script_path.name)

    print(json.dumps(status, indent=2))
    print(delta[["Target", "Split", "Full_R2_mean", "Masked_R2_mean", "Delta_R2_Masked_minus_Full", "Leakage_Risk_Interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
