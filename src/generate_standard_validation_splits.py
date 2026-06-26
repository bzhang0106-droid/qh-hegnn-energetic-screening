#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina


SEEDS = [42, 7, 123]
VAL_FRACTION = 0.20
CALIB_FRACTION = 0.10
BUTINA_DISTANCE_CUTOFF = 0.45
FP_SIZE = 2048
FP_RADIUS = 2
MAX_VALIDATION_GROUP_FRACTION = 0.12


def load_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("qh_hegnn_train_for_split_generation", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return f"invalid::{str(smiles).strip()}"
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    if scaffold:
        return scaffold
    return f"acyclic::{Chem.MolToSmiles(mol, canonical=True)}"


def make_fingerprints(smiles_list: list[str]):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            mol = Chem.MolFromSmiles("")
        fps.append(gen.GetFingerprint(mol))
    return fps


def split_train_calib(train_idx: Iterable[int], seed: int) -> tuple[list[int], list[int]]:
    train = np.array(sorted(int(i) for i in train_idx), dtype=int)
    rng = np.random.default_rng(seed + 1701)
    rng.shuffle(train)
    calib_n = max(1, int(round(len(train) * CALIB_FRACTION)))
    calib = sorted(int(i) for i in train[:calib_n])
    core = sorted(int(i) for i in train[calib_n:])
    return core, calib


def split_random(n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    idx = np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    val_n = max(1, int(round(n * VAL_FRACTION)))
    val = sorted(int(i) for i in idx[:val_n])
    train = sorted(int(i) for i in idx[val_n:])
    core, calib = split_train_calib(train, seed)
    return core, calib, val


def grouped_split(groups: dict[str, list[int]], n: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    target_val_n = int(round(n * VAL_FRACTION))
    max_group_n = int(round(n * MAX_VALIDATION_GROUP_FRACTION))
    items = [(key, sorted(vals)) for key, vals in groups.items()]
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    val: list[int] = []
    val_groups: set[str] = set()

    def should_take(current_n: int, group_n: int) -> bool:
        if group_n > max_group_n:
            return False
        if current_n >= target_val_n:
            return False
        proposed = current_n + group_n
        if proposed <= target_val_n:
            return True
        # Allow a small overshoot only when it moves the split closer to 80/20.
        return abs(proposed - target_val_n) < abs(current_n - target_val_n)

    for key, vals in items:
        if key in val_groups:
            continue
        if should_take(len(val), len(vals)):
            val.extend(vals)
            val_groups.add(key)

    if len(val) < int(0.90 * target_val_n):
        remaining = [(key, vals) for key, vals in items if key not in val_groups]
        remaining.sort(key=lambda kv: len(kv[1]))
        for key, vals in remaining:
            if len(val) >= int(0.90 * target_val_n):
                break
            if should_take(len(val), len(vals)):
                val.extend(vals)
                val_groups.add(key)

    val_set = set(val)
    train = [i for i in range(n) if i not in val_set]
    core, calib = split_train_calib(train, seed)
    return core, calib, sorted(val_set)


def build_butina_groups(fps) -> dict[str, list[int]]:
    dists: list[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - float(s) for s in sims)
    clusters = Butina.ClusterData(dists, len(fps), BUTINA_DISTANCE_CUTOFF, isDistData=True)
    groups: dict[str, list[int]] = {}
    for cluster_id, cluster in enumerate(clusters):
        groups[f"butina_{cluster_id:05d}"] = sorted(int(i) for i in cluster)
    return groups


def nearest_train_similarity(fps, train_idx: list[int], val_idx: list[int]) -> list[float]:
    train_fps = [fps[i] for i in train_idx]
    out: list[float] = []
    for i in val_idx:
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)
        out.append(float(max(sims)) if sims else float("nan"))
    return out


def leakage_count(idx_a: Iterable[int], idx_b: Iterable[int], keys: list[str]) -> int:
    a = {keys[int(i)] for i in idx_a}
    b = {keys[int(i)] for i in idx_b}
    return len(a & b)


def main() -> None:
    work = Path(os.environ.get("QH_HEGNN_WORK", "/scratch/gma/bzhang/qh_hegnn_standard_validation")).resolve()
    script_dir = Path(os.environ.get("QH_HEGNN_SCRIPT_DIR", str(work / "src"))).resolve()
    split_dir = Path(os.environ.get("QH_HEGNN_SPLIT_DIR", str(work / "splits" / "standard_validation"))).resolve()
    split_dir.mkdir(parents=True, exist_ok=True)

    mod = load_training_module(script_dir / "03_egnn_painn_train.py")
    os.chdir(script_dir)
    df = mod.load_training_dataframe()
    raw_examples = mod.build_raw_examples(df)
    xtb, _ = mod.load_xtb_feature_table("../data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv")
    xtb_rows = set(int(v) for v in xtb["row_index"].tolist())
    aligned = [ex for ex in raw_examples if int(ex.row_index) in xtb_rows]
    if len(aligned) < 1000:
        raise RuntimeError(f"Too few aligned examples: {len(aligned)}")

    smiles = [str(ex.smiles) for ex in aligned]
    molnames = [str(ex.molecule) for ex in aligned]
    row_indices = [int(ex.row_index) for ex in aligned]
    scaffolds = [murcko_key(s) for s in smiles]
    fps = make_fingerprints(smiles)

    true_phys_path = Path(
        os.environ.get(
            "QH_HEGNN_PHYSICS_FEATURE_TABLE",
            "/scratch/gma/bzhang/qh_hegnn_standard_validation/data/physics_features_merged.csv",
        )
    )
    true_status: dict[int, str] = {}
    source_group: dict[int, str] = {}
    if true_phys_path.exists():
        phys = pd.read_csv(true_phys_path, usecols=lambda c: c in {"row_index", "extract_status", "esp_status", "qtaim_status", "Source_Group"})
        for _, row in phys.iterrows():
            ri = int(row["row_index"])
            ok = str(row.get("extract_status", "")).lower() == "ok" and str(row.get("esp_status", "")).lower() == "ok" and str(row.get("qtaim_status", "")).lower() == "ok"
            true_status[ri] = "ok" if ok else "partial"
            source_group[ri] = str(row.get("Source_Group", "NA"))

    scaffold_groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(scaffolds):
        scaffold_groups[key].append(i)
    butina_groups = build_butina_groups(fps)
    butina_keys = [""] * len(aligned)
    for key, vals in butina_groups.items():
        for idx in vals:
            butina_keys[int(idx)] = key

    run_cases = []
    inventory = []
    test_rows = []

    split_defs = [
        ("random_80_20", "random_row_split", None, None),
        ("scaffold_80_20", "murcko_scaffold_group_split_size_balanced", scaffold_groups, scaffolds),
        ("butina_80_20", f"butina_cluster_group_split_distance_{BUTINA_DISTANCE_CUTOFF}_size_balanced", butina_groups, butina_keys),
    ]

    for split_name, split_mode, groups, group_keys in split_defs:
        for seed in SEEDS:
            if groups is None:
                core, calib, val = split_random(len(aligned), seed)
            else:
                core, calib, val = grouped_split(groups, len(aligned), seed)
            train = sorted(core + calib)
            case = f"{split_name}_seed{seed}"
            payload = {
                "split_name": case,
                "split_mode": split_mode,
                "seed": seed,
                "n_rows": len(aligned),
                "val_fraction_effective": len(val) / len(aligned),
                "calib_fraction_effective": len(calib) / max(1, len(train)),
                "core_idx": core,
                "calib_idx": calib,
                "val_idx": val,
                "notes": {
                    "generated_at": "2026-06-26",
                    "model_branch": "QH-HEGNN final specialist",
                    "validation_policy": "Random/Scaffold/Butina 80/20 validation splits on physics-feature-aligned examples",
                    "calibration_policy": "10% of non-validation rows; validation labels held out",
                },
            }
            out_json = split_dir / f"{case}.json"
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            run_cases.append((case, seed, str(out_json)))

            nn = nearest_train_similarity(fps, train, val)
            val_rows = [row_indices[i] for i in val]
            val_status = [true_status.get(ri, "unknown") for ri in val_rows]
            val_sources = [source_group.get(ri, "NA") for ri in val_rows]
            inv = {
                "case": case,
                "seed": seed,
                "split_name": split_name,
                "split_mode": split_mode,
                "core_n": len(core),
                "calib_n": len(calib),
                "train_n": len(train),
                "val_n": len(val),
                "val_fraction": len(val) / len(aligned),
                "median_nearest_train_tanimoto": float(np.nanmedian(nn)),
                "mean_nearest_train_tanimoto": float(np.nanmean(nn)),
                "true_phys_ok_val_n": int(sum(s == "ok" for s in val_status)),
                "true_phys_partial_val_n": int(sum(s == "partial" for s in val_status)),
                "true_phys_unknown_val_n": int(sum(s == "unknown" for s in val_status)),
                "scaffold_overlap_train_val": leakage_count(train, val, scaffolds),
                "group_overlap_train_val": leakage_count(train, val, group_keys) if group_keys is not None else 0,
                "split_json": str(out_json),
            }
            inventory.append(inv)

            for local_i, sim in zip(val, nn):
                test_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "split_name": split_name,
                        "aligned_idx": int(local_i),
                        "row_index": row_indices[local_i],
                        "Molecule": molnames[local_i],
                        "SMILES": smiles[local_i],
                        "MurckoScaffold": scaffolds[local_i],
                        "NearestTrainSim": sim,
                        "true_phys_status": true_status.get(row_indices[local_i], "unknown"),
                        "Source_Group": source_group.get(row_indices[local_i], "NA"),
                    }
                )

    with (split_dir / "run_cases_standard_validation.tsv").open("w", encoding="utf-8", newline="") as f:
        for row in run_cases:
            f.write("\t".join(map(str, row)) + "\n")
    pd.DataFrame(inventory).to_csv(split_dir / "standard_validation_inventory.csv", index=False)
    pd.DataFrame(test_rows).to_csv(split_dir / "Model_Robustness_Validation_Rows.csv", index=False)

    print(f"aligned_n={len(aligned)}")
    print(f"split_dir={split_dir}")
    print(pd.DataFrame(inventory).to_string(index=False))


if __name__ == "__main__":
    main()
