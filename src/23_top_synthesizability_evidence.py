#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, RDConfig

RDLogger.DisableLog("rdApp.*")

try:
    import sys, os
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DEFAULT_TRUE = ROOT / "results/True_vs_Pred_Detonation.csv"
DEFAULT_OUT = ROOT / "results/synthesizability_10d"

BENCHMARKS = {
    "TNT": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "TATB": "Nc1c(N)c(N)c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "RDX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "HMX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "PETN": "C(CO[N+](=O)[O-])(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "NTO": "O=c1[nH]n[nH]c(=O)n1",
}

SMARTS = {
    "Nitro_or_nitrate": "[N+](=O)[O-]",
    "C_nitro": "[#6][N+](=O)[O-]",
    "N_nitro_or_nitramine": "[#7][N+](=O)[O-]",
    "Nitrate_ester": "[OX2][N+](=O)[O-]",
    "Azo_or_diazo": "[NX2]=[NX2]",
    "Azide": "[N-]=[N+]=N",
    "Peroxide_like": "[OX2][OX2]",
    "Furazan_like": "o1nncc1",
    "Tetrazole_like": "n1nnnc1",
}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items() if Chem.MolFromSmarts(v) is not None}


def mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def canonical(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else ""


def sa_score(mol) -> float:
    if mol is None:
        return np.nan
    if sascorer is None:
        return np.nan
    return float(sascorer.calculateScore(mol))


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol is not None else None


def benchmark_fps():
    out=[]
    for name, smi in BENCHMARKS.items():
        mol=mol_from_smiles(smi)
        if mol is None:
            continue
        out.append((name, Chem.MolToSmiles(mol), fp(mol)))
    return out


def nearest_benchmark(mol, benches) -> Tuple[str, str, float]:
    f=fp(mol)
    if f is None:
        return "NA", "", np.nan
    best=("NA", "", -1.0)
    for name, smi, bfp in benches:
        sim=float(DataStructs.TanimotoSimilarity(f,bfp))
        if sim>best[2]:
            best=(name,smi,sim)
    return best


def formula_counts(mol) -> Dict[str, int]:
    counts={"C":0,"H":0,"N":0,"O":0}
    if mol is None:
        return counts
    molh=Chem.AddHs(mol)
    for atom in molh.GetAtoms():
        sym=atom.GetSymbol()
        if sym in counts:
            counts[sym]+=1
    return counts


def substructure_counts(mol) -> Dict[str, int]:
    if mol is None:
        return {k:0 for k in PATTERNS}
    return {k: len(mol.GetSubstructMatches(patt)) for k,patt in PATTERNS.items()}


def route_heuristic(row: pd.Series) -> Tuple[str, str, str, str]:
    nitramine=int(row.get("N_nitro_or_nitramine",0) or 0)
    nitrate=int(row.get("Nitrate_ester",0) or 0)
    c_nitro=int(row.get("C_nitro",0) or 0)
    furazan=int(row.get("Furazan_like",0) or 0)
    tetrazole=int(row.get("Tetrazole_like",0) or 0)
    azo=int(row.get("Azo_or_diazo",0) or 0)
    sa=float(row.get("SA_Score_RDKit", np.nan)) if pd.notna(row.get("SA_Score_RDKit", np.nan)) else np.nan
    sim=float(row.get("Nearest_Benchmark_Tanimoto", np.nan)) if pd.notna(row.get("Nearest_Benchmark_Tanimoto", np.nan)) else np.nan

    if nitramine > 0:
        disconnection="late-stage N-nitration/nitramine formation on an aminated N-rich precursor"
        precursor="amino/hydrazino heterocycle or cyclic aminal precursor"
    elif nitrate > 0:
        disconnection="late-stage nitrate ester formation from a hydroxylated precursor"
        precursor="polyhydroxy or hydroxymethyl N/O-rich precursor"
    elif c_nitro > 0:
        disconnection="late-stage C-nitration or nitro-functionalization of an activated heteroaromatic core"
        precursor="activated heteroaromatic or nitro-compatible N-rich ring precursor"
    elif furazan or tetrazole:
        disconnection="assemble N/O-rich heterocyclic core first, then install energetic substituents"
        precursor="furazan/tetrazole/triazole-like heterocycle precursor"
    elif azo:
        disconnection="oxidative coupling or diazotization-like formation of N=N linkage"
        precursor="amino N-rich heterocycle precursor"
    else:
        disconnection="construct N/O-rich core followed by oxidation or nitration where compatible"
        precursor="preformed N/O-rich heterocycle"

    risk=[]
    if pd.notna(sa) and sa>5.0:
        risk.append("high SA score")
    if pd.notna(sim) and sim<0.25:
        risk.append("low similarity to benchmark energetic motifs")
    nc=float(row.get("N_C_Ratio",0) or 0)
    if nc>2.5:
        risk.append("very high N/C ratio")
    if int(row.get("Peroxide_like",0) or 0)>0:
        risk.append("peroxide-like O-O motif")
    if int(row.get("Azide",0) or 0)>0:
        risk.append("azide motif requires additional safety scrutiny")
    if not risk:
        risk.append("no major rule-based risk flag")

    if (pd.notna(sa) and sa<=5.0) and (pd.notna(sim) and sim>=0.25):
        assessment="prioritize for retrosynthesis query and precursor search"
    elif pd.notna(sa) and sa<=5.0:
        assessment="synthetically plausible by SA, but scaffold novelty requires route validation"
    else:
        assessment="route plausibility uncertain; check retrosynthesis and precursor availability before prioritization"
    return disconnection, precursor, "; ".join(risk), assessment


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_TRUE))
    ap.add_argument("--tag", default="current")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args=ap.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.input)
    if "Oracle_D(km/s)" in df.columns:
        df=df.sort_values("Oracle_D(km/s)", ascending=False)
    df=df.head(args.top_n).copy()
    benches=benchmark_fps()

    rows=[]
    sim_matrix=[]
    sim_cols=[]
    for _, r in df.iterrows():
        smi=str(r.get("SMILES", ""))
        mol=mol_from_smiles(smi)
        counts=formula_counts(mol)
        sub=substructure_counts(mol)
        nearest_name, nearest_smi, nearest_sim=nearest_benchmark(mol, benches)
        f=fp(mol)
        sim_row=[]
        for bname, bsmi, bfp in benches:
            if bname not in sim_cols:
                sim_cols.append(bname)
            sim_row.append(float(DataStructs.TanimotoSimilarity(f,bfp)) if f is not None else np.nan)
        sim_matrix.append(sim_row)
        c=max(counts.get("C",0),1)
        rec={
            "Rank_by_Oracle_D": len(rows)+1,
            "Molecule": r.get("Molecule"),
            "SMILES": smi,
            "Canonical_SMILES": canonical(smi),
            "Oracle_D(km/s)": r.get("Oracle_D(km/s)"),
            "Oracle_P(GPa)": r.get("Oracle_P(GPa)"),
            "Oracle_Density": r.get("Oracle_Density"),
            "Oracle_Density_Type": r.get("Oracle_Density_Type"),
            "Vertical_BDE(kcal/mol)": r.get("Vertical_BDE(kcal/mol)"),
            "SA_Score_RDKit": sa_score(mol),
            "ExactMolWt": Descriptors.ExactMolWt(mol) if mol is not None else np.nan,
            "Ring_Count": rdMolDescriptors.CalcNumRings(mol) if mol is not None else np.nan,
            "Bridgehead_Atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol) if mol is not None else np.nan,
            "Spiro_Atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol) if mol is not None else np.nan,
            "C": counts.get("C",0), "H": counts.get("H",0), "N": counts.get("N",0), "O": counts.get("O",0),
            "N_C_Ratio": counts.get("N",0)/c,
            "O_C_Ratio": counts.get("O",0)/c,
            "Nearest_Benchmark": nearest_name,
            "Nearest_Benchmark_SMILES": nearest_smi,
            "Nearest_Benchmark_Tanimoto": nearest_sim,
        }
        rec.update(sub)
        rows.append(rec)
    table=pd.DataFrame(rows)
    route_rows=[]
    for _, row in table.iterrows():
        dis, prec, risk, assess=route_heuristic(row)
        route_rows.append({
            "Rank_by_Oracle_D": row["Rank_by_Oracle_D"],
            "Molecule": row["Molecule"],
            "SMILES": row["SMILES"],
            "SA_Score_RDKit": row["SA_Score_RDKit"],
            "Nearest_Benchmark": row["Nearest_Benchmark"],
            "Nearest_Benchmark_Tanimoto": row["Nearest_Benchmark_Tanimoto"],
            "Suggested_Disconnection": dis,
            "Plausible_Precursor_Class": prec,
            "Main_Synthetic_Risk": risk,
            "Assessment": assess,
            "Note": "Rule-based plausibility only; final manuscript should add retrosynthesis-tool or literature/precursor evidence for selected top molecules.",
        })
    route=pd.DataFrame(route_rows)

    prefix=f"{args.tag}_Top{args.top_n}"
    synth_csv=out/f"Table_{prefix}_Synthesizability_10D.csv"
    route_csv=out/f"Table_{args.tag}_Top10_Retrosynthesis_Assessment_10D.csv"
    sim_csv=out/f"Table_{prefix}_BenchmarkSimilarity_10D.csv"
    smi_out=out/f"{args.tag}_Top10_Retrosynthesis_Input.smi"
    csv_out=out/f"{args.tag}_Top10_Retrosynthesis_Input.csv"

    table.to_csv(synth_csv,index=False)
    route.head(10).to_csv(route_csv,index=False)
    route.head(10)[["Molecule","SMILES"]].to_csv(csv_out,index=False)
    with smi_out.open("w",encoding="utf-8") as f:
        for _, row in route.head(10).iterrows():
            f.write(f"{row['SMILES']} {row['Molecule']}\n")

    sim_df=pd.DataFrame(sim_matrix, columns=sim_cols)
    sim_df.insert(0,"Molecule",table["Molecule"].values)
    sim_df.to_csv(sim_csv,index=False)

    # Figure 1: SA distribution.
    fig, ax=plt.subplots(figsize=(4.2,2.6))
    ax.hist(table["SA_Score_RDKit"].dropna(), bins=8, color="#4B8BBE", alpha=0.8, edgecolor="white")
    ax.axvline(5.0, color="#9C2F2F", linestyle="--", linewidth=1.0, label="SA=5")
    ax.set_xlabel("RDKit SA score")
    ax.set_ylabel("Top-molecule count")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out/f"Figure_{prefix}_SA_Distribution_10D.png", dpi=300, bbox_inches="tight")
    fig.savefig(out/f"Figure_{prefix}_SA_Distribution_10D.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: benchmark similarity heatmap.
    fig, ax=plt.subplots(figsize=(5.4, max(3.0, 0.22*len(table))))
    mat=sim_df.set_index("Molecule")
    if sns is not None:
        sns.heatmap(mat, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar_kws={"label":"Morgan Tanimoto"})
    else:
        im=ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, label="Morgan Tanimoto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
    ax.set_xlabel("Benchmark energetic material")
    ax.set_ylabel("Top candidates")
    fig.tight_layout()
    fig.savefig(out/f"Figure_{prefix}_BenchmarkSimilarity_10D.png", dpi=300, bbox_inches="tight")
    fig.savefig(out/f"Figure_{prefix}_BenchmarkSimilarity_10D.pdf", bbox_inches="tight")
    plt.close(fig)

    print({
        "input": str(args.input),
        "rows": int(len(table)),
        "synth_csv": str(synth_csv),
        "route_csv": str(route_csv),
        "sim_csv": str(sim_csv),
        "retrosynthesis_smi": str(smi_out),
    })

if __name__ == "__main__":
    main()
