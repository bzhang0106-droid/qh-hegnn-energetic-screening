#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_ultimate_discovery.py

Clean-sanitized HELS candidate inference.

This version reads the production 10-target final-specialist joblib model from:
  ../results/final_model_release/production_v4_10d/

Model:
  Final-Specialist-Hybrid v2
  = target-wise 2D+xTB teacher ensemble + xTB-aware EGNN residual correction

Notes:
- The production model was trained on curated_molecule_clean_v1.
- Candidate tables usually contain SMILES only. If candidate-level xTB descriptor
  columns are absent, the model uses training-set median xTB descriptors stored in
  the joblib bundle and records this explicitly in the output.
- xTB/MMFF 3D geometry is still generated for the residual EGNN branch.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import copy
import multiprocessing as mp
import subprocess
import tempfile
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, radius_graph

torch.set_num_threads(1)
RDLogger.DisableLog("rdApp.*")

INPUT_CSV = "../data/GPT_Generated_Candidates.csv"
OUTPUT_CSV = "../results/Surrogate_10Target_Predictions.csv"
COMPAT_OUTPUT_CSV = "../results/Surrogate_10D_Predictions.csv"
MODEL_PATH = "../results/final_model_release/production_v4_10d/final_specialist_10d_bde_production.joblib"

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

    "Num_Nitramine_NNO2",
    "Num_Nitrate_Ester_ONO2",
    "Num_Nitroaromatic_CNO2",
    "Num_Gem_Dinitro",
    "Num_Furazan",
    "Num_Tetrazole",
    "Num_Triazole",
    "Num_Nitroso",
    "Num_N_Oxide",
    "Num_NitroAdjacentPairs",
    "Max_Nitro_Adjacency",
    "Nitro_Nitrogen_Ratio",
    "Nitro_Per_HeavyAtom",
    "Oxygen_Balance_100",
    "DBE",
    "Aromatic_Ring_Count",
    "Heteroaromatic_Ring_Count",
    "Nitrogen_Oxygen_Ratio",
    "Explosophore_Count",
    "Trigger_Linkage_Count",
]

ATOMIC_NUMS = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "Cl": 17}


@dataclass
class TargetTeacherBundle:
    target: str
    transform_name: str
    selected_names: List[str]
    base_models: List[Any]
    meta_model: Optional[Any]
    oof_r2: float
    selection_table: List[Dict[str, Any]]


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
        self.edge_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, 1, bias=False),
        )

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


class EGNNBackbone(nn.Module):
    def __init__(self, hidden_dim: int = 128, n_layers: int = 4, radius: float = 4.0, max_neighbors: int = 32):
        super().__init__()
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.node_emb = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([NativeEGNNLayer(hidden_dim) for _ in range(n_layers)])

    def forward(self, z: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = self.node_emb(z)
        edge_index = radius_graph(pos, r=self.radius, batch=batch, max_num_neighbors=self.max_neighbors)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)
        return global_mean_pool(h, batch)


class XTBResidualEGNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_targets: int = 10, num_2d: int = 9, num_xtb: int = 1):
        super().__init__()
        self.backbone = EGNNBackbone(hidden_dim=hidden_dim)
        self.x2d_encoder = nn.Sequential(
            nn.Linear(num_2d, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.xtb_encoder = nn.Sequential(
            nn.Linear(num_xtb, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 64 + 128, 256),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, num_targets),
        )

    def forward(self, z, pos, batch, x_2d, x_xtb):
        h_graph = self.backbone(z, pos, batch)
        h_2d = self.x2d_encoder(x_2d.float())
        h_xtb = self.xtb_encoder(x_xtb.float())
        return self.head(torch.cat([h_graph, h_2d, h_xtb], dim=1))


class InferenceDataset(Dataset):
    def __init__(self, data_list: List[Data]):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _safe_smarts_count(mol: Chem.Mol, smarts: str) -> int:
    try:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            return 0
        return len(mol.GetSubstructMatches(patt, uniquify=True))
    except Exception:
        return 0


def _calc_dbe(mol: Chem.Mol) -> float:
    try:
        formula = rdMolDescriptors.CalcMolFormula(mol)
        counts = {el: 0 for el in ["C", "H", "N", "F", "Cl", "Br", "I"]}
        for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
            if el in counts:
                counts[el] += int(num) if num else 1
        c = counts["C"]
        h = counts["H"]
        n = counts["N"]
        x = counts["F"] + counts["Cl"] + counts["Br"] + counts["I"]
        return float((2 * c + 2 + n - h - x) / 2.0)
    except Exception:
        return 0.0


def _nitro_adjacency_features(mol: Chem.Mol) -> Tuple[int, int]:
    nitro = Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")
    if nitro is None:
        return 0, 0

    nitro_matches = mol.GetSubstructMatches(nitro, uniquify=True)
    nitro_n_atoms = set()
    for m in nitro_matches:
        if len(m):
            nitro_n_atoms.add(int(m[0]))

    attachment_counts = {}
    for n_idx in nitro_n_atoms:
        atom = mol.GetAtomWithIdx(n_idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() not in (7, 8):
                aidx = int(nbr.GetIdx())
                attachment_counts[aidx] = attachment_counts.get(aidx, 0) + 1
            elif nbr.GetAtomicNum() == 7:
                aidx = int(nbr.GetIdx())
                attachment_counts[aidx] = attachment_counts.get(aidx, 0) + 1

    max_adj = max(attachment_counts.values()) if attachment_counts else 0
    total_pairs = sum(v * (v - 1) // 2 for v in attachment_counts.values() if v >= 2)
    return int(total_pairs), int(max_adj)


def extract_2d_features(smiles: str) -> List[float]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return [0.0] * len(TWO_D_FEATURE_NAMES)

    nitro = "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"
    num_nitro = _safe_smarts_count(mol, nitro)
    num_n = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    num_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    heavy = max(1, mol.GetNumHeavyAtoms())

    exact_mw = float(Descriptors.ExactMolWt(mol))
    num_hetero = float(rdMolDescriptors.CalcNumHeteroatoms(mol))
    num_rings = float(rdMolDescriptors.CalcNumRings(mol))
    num_c_no2 = float(_safe_smarts_count(mol, f"[#6]-{nitro}"))
    num_n_no2 = float(_safe_smarts_count(mol, f"[#7]-{nitro}"))
    num_n_eq_n = float(_safe_smarts_count(mol, "[#7]=[#7]"))
    num_azide = float(_safe_smarts_count(mol, "[N]=[N+]=[N-]"))

    num_nitramine = float(_safe_smarts_count(mol, f"[NX3,NX4]-{nitro}"))
    num_nitrate_ester = float(_safe_smarts_count(mol, "[OX2]-[N+](=O)[O-]"))
    num_nitroaromatic = float(_safe_smarts_count(mol, f"[a]-{nitro}"))
    num_gem_dinitro = float(_safe_smarts_count(mol, f"[#6]({nitro})({nitro})"))
    num_furazan = float(_safe_smarts_count(mol, "c1nonc1") + _safe_smarts_count(mol, "C1=NON=C1"))
    num_tetrazole = float(_safe_smarts_count(mol, "c1nnnn1") + _safe_smarts_count(mol, "C1=NN=NN1"))
    num_triazole = float(_safe_smarts_count(mol, "c1nncn1") + _safe_smarts_count(mol, "C1=NNC=N1"))
    num_nitroso = float(_safe_smarts_count(mol, "[N!+]=O"))
    num_n_oxide = float(_safe_smarts_count(mol, "[n+][O-]") + _safe_smarts_count(mol, "[N+][O-]"))

    nitro_adj_pairs, max_nitro_adj = _nitro_adjacency_features(mol)
    nitro_n_ratio = float(num_nitro / max(num_n, 1))
    nitro_heavy_ratio = float(num_nitro / heavy)

    c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    h = sum(1 for atom in Chem.AddHs(mol).GetAtoms() if atom.GetAtomicNum() == 1)
    oxygen_balance_100 = float(1600.0 * (num_o - 2.0 * c - 0.5 * h) / max(exact_mw, 1e-6))

    dbe = float(_calc_dbe(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    heteroaromatic_rings = float(_safe_smarts_count(mol, "[a]1[a][a][a][a]1"))
    n_o_ratio = float(num_n / max(num_o, 1))

    explosophore_count = float(
        num_nitro + num_azide + num_nitramine + num_nitrate_ester + num_nitroso + num_n_oxide
    )
    trigger_linkage_count = float(
        num_c_no2 + num_n_no2 + num_nitramine + num_nitrate_ester + num_n_eq_n + num_azide + num_gem_dinitro
    )

    vals = [
        exact_mw,
        num_hetero,
        num_rings,
        float(num_nitro),
        float(num_n),
        num_c_no2,
        num_n_no2,
        num_n_eq_n,
        num_azide,

        num_nitramine,
        num_nitrate_ester,
        num_nitroaromatic,
        num_gem_dinitro,
        num_furazan,
        num_tetrazole,
        num_triazole,
        num_nitroso,
        num_n_oxide,
        float(nitro_adj_pairs),
        float(max_nitro_adj),
        nitro_n_ratio,
        nitro_heavy_ratio,
        oxygen_balance_100,
        dbe,
        aromatic_rings,
        heteroaromatic_rings,
        n_o_ratio,
        explosophore_count,
        trigger_linkage_count,
    ]

    if len(vals) != len(TWO_D_FEATURE_NAMES):
        raise RuntimeError(f"2D feature length mismatch: {len(vals)} vs {len(TWO_D_FEATURE_NAMES)}")
    return [float(x) if np.isfinite(x) else 0.0 for x in vals]


def morgan_fp(smiles: str, fp_size: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    arr = np.zeros((fp_size,), dtype=np.float32)
    if mol is None:
        return arr
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_size)
    fp = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def inverse_transform(transform_name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    if transform_name == "log":
        return np.exp(z).astype(np.float32)
    if transform_name == "bounded_logit_0p25":
        upper = 0.25
        u = 1.0 / (1.0 + np.exp(-z))
        return (upper * u).astype(np.float32)
    return z.astype(np.float32)


def impute_apply(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32).copy()
    med = np.asarray(med, dtype=np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        X[bad] = np.take(med, np.where(bad)[1])
    return X.astype(np.float32)


def smiles_to_3d_xtb(args_tuple: Tuple[int, str]) -> Tuple[int, Optional[Data], str]:
    idx, smi = args_tuple
    try:
        smi = str(smi).strip()
        mol0 = Chem.MolFromSmiles(smi)
        if mol0 is None:
            return idx, None, "SMILES_parse_failed"
        if Chem.GetFormalCharge(mol0) != 0:
            return idx, None, "non_neutral"

        mol = Chem.AddHs(mol0)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return idx, None, "ETKDG_failed"

        xtb_env = os.environ.copy()
        xtb_env["OMP_NUM_THREADS"] = "1"
        xtb_env["MKL_NUM_THREADS"] = "1"

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "init.xyz")
            Chem.MolToXYZFile(mol, xyz_path)
            opt_xyz = os.path.join(tmpdir, "xtbopt.xyz")

            try:
                cmd = ["xtb", "init.xyz", "--opt", "normal", "--gfn", "2", "--chrg", "0"]
                subprocess.run(
                    cmd,
                    cwd=tmpdir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=45,
                    env=xtb_env,
                    check=False,
                )
            except Exception:
                pass

            if not os.path.exists(opt_xyz):
                try:
                    if AllChem.MMFFHasAllMoleculeParams(mol):
                        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=300)
                    else:
                        AllChem.UFFOptimizeMolecule(mol, maxIters=300)
                except Exception:
                    pass
                Chem.MolToXYZFile(mol, opt_xyz)

            z, pos = [], []
            with open(opt_xyz, "r", encoding="utf-8", errors="ignore") as f:
                for line in f.readlines()[2:]:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] in ATOMIC_NUMS:
                        z.append(ATOMIC_NUMS[parts[0]])
                        pos.append([float(parts[1]), float(parts[2]), float(parts[3])])

        if not z:
            return idx, None, "empty_graph"

        data = Data(
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float),
            candidate_index=int(idx),
        )
        return idx, data, "Success"
    except Exception as exc:
        return idx, None, f"exception:{exc}"


def predict_teacher(bundle: Dict[str, Any], X_teacher: np.ndarray) -> np.ndarray:
    preds = np.zeros((X_teacher.shape[0], len(bundle["targets"])), dtype=np.float32)

    for j, tb in enumerate(bundle["teacher_bundles"]):
        base_preds = []
        for model in tb.base_models:
            z = model.predict(X_teacher)
            base_preds.append(inverse_transform(tb.transform_name, z))
        stack = np.column_stack(base_preds).astype(np.float32)

        if tb.meta_model is None:
            preds[:, j] = stack[:, 0]
        else:
            preds[:, j] = tb.meta_model.predict(stack).astype(np.float32)

    return preds.astype(np.float32)


def build_feature_arrays(
    df_valid: pd.DataFrame,
    bundle: Dict[str, Any],
    fp_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    xtb_feature_names = list(bundle["xtb_feature_names"])
    has_candidate_xtb = any(c in df_valid.columns for c in xtb_feature_names)

    X_teacher, X_xtb, X2d_raw = [], [], []
    for _, row in df_valid.iterrows():
        smi = str(row["SMILES"])
        fp = morgan_fp(smi, fp_size=fp_size)
        compact = np.asarray(extract_2d_features(smi), dtype=np.float32)

        xtb_vals = []
        for c in xtb_feature_names:
            if c in row.index:
                try:
                    v = float(row[c])
                except Exception:
                    v = np.nan
            else:
                v = np.nan
            xtb_vals.append(v)
        xtb_vec = np.asarray(xtb_vals, dtype=np.float32)

        X_teacher.append(np.concatenate([fp, compact, xtb_vec], axis=0))
        X_xtb.append(xtb_vec)
        X2d_raw.append(compact)

    mode = "candidate_xtb_columns_with_median_imputation" if has_candidate_xtb else "training_median_imputed_xtb_descriptors"
    return (
        np.vstack(X_teacher).astype(np.float32),
        np.vstack(X_xtb).astype(np.float32),
        np.vstack(X2d_raw).astype(np.float32),
        mode,
    )


def predict_residual(bundle: Dict[str, Any], data_list: List[Data], X2d_raw: np.ndarray, X_xtb_imputed: np.ndarray, device: torch.device) -> np.ndarray:
    targets = list(bundle["targets"])
    model = XTBResidualEGNN(
        hidden_dim=int(bundle.get("hidden_dim", 128)),
        num_targets=len(targets),
        num_2d=len(bundle["compact_2d_features"]),
        num_xtb=len(bundle["xtb_feature_names"]),
    ).to(device)

    model.load_state_dict(bundle["residual_state_dict"])
    model.eval()

    x2d_mean = torch.tensor(bundle["x2d_mean"], dtype=torch.float)
    x2d_std = torch.tensor(bundle["x2d_std"], dtype=torch.float)
    x2d_std[x2d_std == 0] = 1.0

    xtb_mean = torch.tensor(bundle["xtb_mean"], dtype=torch.float)
    xtb_std = torch.tensor(bundle["xtb_std"], dtype=torch.float)
    xtb_std[xtb_std == 0] = 1.0

    resid_mean = torch.tensor(bundle["resid_mean"], dtype=torch.float)
    resid_std = torch.tensor(bundle["resid_std"], dtype=torch.float)

    for i, data in enumerate(data_list):
        x2d = torch.tensor((X2d_raw[i] - x2d_mean.numpy()) / x2d_std.numpy(), dtype=torch.float).unsqueeze(0)
        xxtb = torch.tensor((X_xtb_imputed[i] - xtb_mean.numpy()) / xtb_std.numpy(), dtype=torch.float).unsqueeze(0)
        data.x_2d = x2d
        data.x_xtb = xxtb

    loader = DataLoader(InferenceDataset(data_list), batch_size=256, shuffle=False)

    blocks = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Final-specialist residual EGNN inference"):
            batch = batch.to(device)
            out = model(batch.z, batch.pos, batch.batch, batch.x_2d, batch.x_xtb)
            raw = out.cpu() * resid_std + resid_mean
            blocks.append(raw.numpy())

    return np.vstack(blocks).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean-sanitized final-specialist candidate inference.")
    parser.add_argument("--input", default=INPUT_CSV)
    parser.add_argument("--output", default=OUTPUT_CSV)
    parser.add_argument("--compat_output", default=COMPAT_OUTPUT_CSV)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--fp_size", type=int, default=2048)
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    print("=" * 78)
    print("🔮 04 Ultimate Discovery | clean-sanitized final-specialist inference")
    print("=" * 78)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Production final-specialist model not found: {args.model}")
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Candidate input CSV not found: {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model: {args.model}")

    bundle = joblib.load(args.model)
    if bundle.get("model_type") != "Final-Specialist-Hybrid-v2":
        print(f"[WARN] Unexpected model_type: {bundle.get('model_type')}")

    df = pd.read_csv(args.input)
    if "SMILES" not in df.columns:
        raise ValueError(f"{args.input} must contain a SMILES column.")

    df = df.dropna(subset=["SMILES"]).copy()
    df["Canonical_SMILES"] = df["SMILES"].map(canonicalize_smiles)
    df = df.drop_duplicates(subset=["Canonical_SMILES"], keep="first").reset_index(drop=True)

    print(f"[INFO] Candidate rows after canonical deduplication: {len(df)}")

    workers = args.max_workers
    if workers is None:
        workers = min(60, max(1, (os.cpu_count() or 2) - 4))

    valid_records = []
    fail_reasons: Dict[str, int] = {}
    tasks = list(zip(range(len(df)), df["SMILES"].astype(str).tolist()))

    print(f"[INFO] Building xTB/MMFF 3D graphs with {workers} workers...")
    with mp.Pool(processes=workers) as pool:
        for idx, data, msg in tqdm(pool.imap(smiles_to_3d_xtb, tasks), total=len(tasks)):
            if data is None:
                fail_reasons[msg] = fail_reasons.get(msg, 0) + 1
            else:
                valid_records.append((idx, data))

    if fail_reasons:
        print("[INFO] 3D preprocessing failures:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1])[:20]:
            print(f"  - {reason}: {count}")

    if not valid_records:
        raise RuntimeError("No valid 3D candidate graphs were generated.")

    valid_indices = [x[0] for x in valid_records]
    valid_graphs = [x[1] for x in valid_records]
    df_valid = df.iloc[valid_indices].copy().reset_index(drop=True)

    X_teacher_raw, X_xtb_raw, X2d_raw, xtb_mode = build_feature_arrays(df_valid, bundle, fp_size=args.fp_size)
    expected_teacher_dim = len(bundle["teacher_feature_median"])
    if X_teacher_raw.shape[1] != expected_teacher_dim:
        raise RuntimeError(
            f"Teacher feature dimension mismatch: candidate={X_teacher_raw.shape[1]} "
            f"but model expects {expected_teacher_dim}. Check 04 TWO_D_FEATURE_NAMES against 03."
        )
    X_teacher = impute_apply(X_teacher_raw, bundle["teacher_feature_median"])
    X_xtb = impute_apply(X_xtb_raw, bundle["xtb_feature_median"])

    print(f"[INFO] Teacher feature dimension: {X_teacher.shape[1]}")
    print(f"[INFO] Residual xTB feature dimension: {X_xtb.shape[1]}")
    print(f"[INFO] Candidate xTB feature mode: {xtb_mode}")

    teacher_pred = predict_teacher(bundle, X_teacher)
    residual_pred = predict_residual(bundle, valid_graphs, X2d_raw, X_xtb, device=device)

    alpha = np.asarray(bundle["alpha"], dtype=np.float32).reshape(1, -1)
    final_pred = teacher_pred + alpha * residual_pred

    targets = list(bundle["targets"])
    pred_df = pd.DataFrame(final_pred, columns=[f"Pred_{c}" for c in targets])

    out = df_valid.copy()
    if "Molecule" not in out.columns:
        out.insert(0, "Molecule", [f"Candidate_{i:06d}" for i in range(len(out))])
    out = pd.concat([out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    out["Model_Release"] = os.path.basename(args.model)
    out["Inference_Model_Type"] = "Final-Specialist-Hybrid-v2-clean-sanitized"
    out["Inference_xTB_Feature_Mode"] = xtb_mode
    out["Density_Label_Note"] = bundle.get(
        "density_label_note",
        "Density_calc(g/cm3) is a molecular-volume-derived proxy unless crystal density is explicitly supplied.",
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out.to_csv(args.output, index=False)
    out.to_csv(args.compat_output, index=False)

    print("=" * 78)
    print(f"✅ Final-specialist surrogate inference completed: {len(out)} valid candidates")
    print(f"📄 Main output: {args.output}")
    print(f"📄 Compatibility output: {args.compat_output}")
    print("=" * 78)


if __name__ == "__main__":
    main()
