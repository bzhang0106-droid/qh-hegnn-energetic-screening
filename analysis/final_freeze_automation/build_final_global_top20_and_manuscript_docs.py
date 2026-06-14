from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, rdMolDescriptors, RDConfig

RDLogger.DisableLog("rdApp.*")

try:
    import sys

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DATE = "20260605"
MANUSCRIPT = ROOT / "manuscript_npJ"
RESULTS = ROOT / "results" / "final_global_top20"
PACKAGE = MANUSCRIPT / f"final_submission_package_AL08_{DATE}"
GLOBAL_TOP20_CSV = RESULTS / "Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv"

BENCHMARKS = {
    "TNT": "Cc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
    "TATB": "Nc1c(N)c(N)c([N+](=O)[O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
    "FOX-7": "NC(=C([N+](=O)[O-])[N+](=O)[O-])N",
    "RDX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])C1",
    "HMX": "O=[N+]([O-])N1CN([N+](=O)[O-])CN([N+](=O)[O-])CN([N+](=O)[O-])CN1",
    "PETN": "C(CO[N+](=O)[O-])(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-]",
    "NTO": "O=c1[nH]n[nH]c(=O)n1",
}

SMARTS = {
    "C_nitro": "[#6][N+](=O)[O-]",
    "N_nitro_or_nitramine": "[#7][N+](=O)[O-]",
    "Nitrate_ester": "[OX2][N+](=O)[O-]",
    "Azo_or_diazo": "[NX2]=[NX2]",
    "Azide": "[N-]=[N+]=N",
    "Peroxide_like": "[OX2][OX2]",
    "Tetrazole_like": "n1nnnc1",
    "Furazan_like": "o1nncc1",
}
PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items() if Chem.MolFromSmarts(v) is not None}


def mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) if m is not None else None


def murcko_smiles(m) -> str:
    if m is None:
        return ""
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s is not None and s.GetNumAtoms() else ""
    except Exception:
        return ""


def sa_score(m) -> float:
    if m is None or sascorer is None:
        return float("nan")
    return float(sascorer.calculateScore(m))


def formula_counts(m) -> dict[str, int]:
    counts = {"C": 0, "H": 0, "N": 0, "O": 0}
    if m is None:
        return counts
    mh = Chem.AddHs(m)
    for atom in mh.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts


def substructure_counts(m) -> dict[str, int]:
    if m is None:
        return {k: 0 for k in PATTERNS}
    return {k: len(m.GetSubstructMatches(patt)) for k, patt in PATTERNS.items()}


def kj(c: int, h: int, n: int, o: int, mw: float, hof_kcal_mol: float, density: float) -> tuple[float, float]:
    if mw <= 0 or density <= 0 or not math.isfinite(hof_kcal_mol):
        return float("nan"), float("nan")
    o_avail = float(o)
    h2o = min(h / 2.0, o_avail)
    o_avail -= h2o
    h2 = (h / 2.0) - h2o
    co = min(float(c), o_avail)
    o_avail -= co
    co2 = min(co, o_avail)
    o_avail -= co2
    co -= co2
    o2 = max(o_avail, 0.0) / 2.0
    n2 = n / 2.0
    gas_moles = h2o + co + co2 + o2 + n2 + h2
    if gas_moles <= 0:
        return float("nan"), float("nan")
    gas_mass = h2o * 18.015 + co * 28.01 + co2 * 44.01 + o2 * 31.998 + n2 * 28.013 + h2 * 2.016
    N = gas_moles / mw
    M = gas_mass / gas_moles
    hof_products = h2o * -57.8 + co * -26.4 + co2 * -94.0
    q_heat = (hof_kcal_mol - hof_products) / mw * 1000.0
    if q_heat <= 0:
        return float("nan"), float("nan")
    D = 1.01 * (N * (M**0.5) * (q_heat**0.5)) ** 0.5 * (1.0 + 1.30 * density)
    P = 1.558 * density**2 * N * (M**0.5) * (q_heat**0.5)
    return round(float(D), 2), round(float(P), 1)


def benchmark_df() -> pd.DataFrame:
    rows = []
    for name, smi in BENCHMARKS.items():
        m = mol(smi)
        rows.append(
            {
                "Benchmark": name,
                "Canonical_SMILES": Chem.MolToSmiles(m) if m is not None else "",
                "Murcko_Scaffold": murcko_smiles(m),
                "SA_Score_RDKit": sa_score(m),
            }
        )
    return pd.DataFrame(rows)


def nearest(m, benches: pd.DataFrame) -> tuple[str, float, str, float]:
    f = fp(m)
    best = ("NA", float("nan"))
    best_s = ("NA", float("nan"))
    if f is None:
        return best[0], best[1], best_s[0], best_s[1]
    cscaf = mol(murcko_smiles(m))
    cf = fp(cscaf)
    for _, b in benches.iterrows():
        bm = mol(b["Canonical_SMILES"])
        bf = fp(bm)
        if bf is not None:
            sim = float(DataStructs.TanimotoSimilarity(f, bf))
            if not math.isfinite(best[1]) or sim > best[1]:
                best = (b["Benchmark"], sim)
        bs = mol(str(b["Murcko_Scaffold"]))
        sf = fp(bs)
        if cf is not None and sf is not None:
            sim = float(DataStructs.TanimotoSimilarity(cf, sf))
            if not math.isfinite(best_s[1]) or sim > best_s[1]:
                best_s = (b["Benchmark"], sim)
    return best[0], best[1], best_s[0], best_s[1]


def route_hypothesis(counts: dict[str, int]) -> tuple[str, str]:
    if counts.get("N_nitro_or_nitramine", 0) > 0:
        return "late-stage N-nitration/nitramine formation", "amino/hydrazino/aminal N-rich precursor"
    if counts.get("Nitrate_ester", 0) > 0:
        return "late-stage nitrate ester formation", "hydroxylated precursor"
    if counts.get("C_nitro", 0) > 0:
        return "late-stage C-nitration", "activated heteroaromatic or carbocyclic precursor"
    if counts.get("Tetrazole_like", 0) > 0 or counts.get("Furazan_like", 0) > 0:
        return "N/O-rich heterocycle-first construction", "furazan/tetrazole-like precursor"
    return "core-first assembly followed by oxidation/nitration", "external retrosynthesis required"


def evidence_tier(sa: float, q75: float, sim: float, scaf_sim: float, high_risk: int) -> tuple[str, str, str]:
    if (sim >= 0.45 or scaf_sim >= 0.55) and sa <= max(5.0, q75) and high_risk == 0:
        return "A", "main-text candidate after route check", "benchmark-adjacent synthetic-plausibility screen"
    if (sim >= 0.25 or scaf_sim >= 0.35) and sa <= 5.5 and high_risk == 0:
        return "B", "SI or cautious main text", "moderate analogue evidence; route validation needed"
    return "C", "SI only until independent route evidence exists", "new/high-uncertainty scaffold; synthesizability not established"


def source_prefix(mid: str) -> str:
    s = str(mid)
    for p in ["AL08", "AL07", "AL06", "AL05", "AL04", "GPT_AL"]:
        if s.startswith(p):
            return p
    return "BASE"


def build_global_table() -> pd.DataFrame:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
    benches = benchmark_df()
    q75 = float(benches["SA_Score_RDKit"].quantile(0.75))
    rows = []
    for _, r in df.iterrows():
        smi = r.get("SMILES")
        m = mol(smi)
        if m is None:
            continue
        counts = formula_counts(m)
        mw = float(r.get("Molecular_Weight", Descriptors.MolWt(m)))
        density = r.get("Final_Detonation_Density_Used(g/cm3)", float("nan"))
        if not pd.notna(density):
            density = r.get("Density_calibrated(g/cm3)", float("nan"))
        if not pd.notna(density):
            density = r.get("Density_calc(g/cm3)", float("nan"))
        hof = r.get("Heat_of_Formation(kcal/mol)", float("nan"))
        D = r.get("Final_Detonation_D(km/s)", float("nan"))
        P = r.get("Final_Detonation_P(GPa)", float("nan"))
        if not pd.notna(D) or not pd.notna(P):
            try:
                D, P = kj(counts["C"], counts["H"], counts["N"], counts["O"], mw, float(hof), float(density))
            except Exception:
                D, P = float("nan"), float("nan")
        if not pd.notna(D):
            continue
        sub = substructure_counts(m)
        nb, sim, ns, scaf_sim = nearest(m, benches)
        high_risk = int(sub.get("Peroxide_like", 0) + sub.get("Azide", 0))
        sa = float(r.get("SAscore", r.get("SA_Score", sa_score(m))))
        route, prec = route_hypothesis(sub)
        tier, use, claim = evidence_tier(sa, q75, sim, scaf_sim, high_risk)
        rows.append(
            {
                "Molecule": r.get("Molecule"),
                "Source_Group": source_prefix(str(r.get("Molecule"))),
                "SMILES": smi,
                "Canonical_SMILES": Chem.MolToSmiles(m),
                "Murcko_Scaffold": murcko_smiles(m),
                "Final_Global_D(km/s)": D,
                "Final_Global_P(GPa)": P,
                "Density_used(g/cm3)": round(float(density), 4) if pd.notna(density) else float("nan"),
                "Heat_of_Formation(kcal/mol)": r.get("Heat_of_Formation(kcal/mol)"),
                "HOMO_LUMO_Gap(eV)": r.get("HOMO_LUMO_Gap(eV)"),
                "Vertical_BDE(kcal/mol)": r.get("Vertical_BDE(kcal/mol)"),
                "VS_max": r.get("VS_max"),
                "Sigma2_tot": r.get("Sigma2_tot"),
                "Nu": r.get("Nu"),
                "Trigger_Bond_Rho": r.get("Trigger_Bond_Rho"),
                "Molecular_Weight": mw,
                "SA_Score": sa,
                "ExactMolWt": Descriptors.ExactMolWt(m),
                "cLogP_RDKit": Crippen.MolLogP(m),
                "TPSA_RDKit": rdMolDescriptors.CalcTPSA(m),
                "Ring_Count": rdMolDescriptors.CalcNumRings(m),
                "Nearest_Benchmark": nb,
                "Nearest_Benchmark_Tanimoto": sim,
                "Nearest_Benchmark_Scaffold": ns,
                "Nearest_Benchmark_Scaffold_Tanimoto": scaf_sim,
                "High_Risk_Motif_Count": high_risk,
                "Evidence_Tier": tier,
                "Synthesizability_Evidence": claim,
                "Recommended_Manuscript_Use": use,
                "Route_Hypothesis": route,
                "Likely_Precursor_Family": prec,
                "External_Retrosynthesis_Status": "pending_not_installed",
                "Manual_Literature_Precursor_Status": "pending",
                "Submission_Claim_Status": "screening_only_until_route_validated",
                **sub,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(["Final_Global_D(km/s)", "Final_Global_P(GPa)"], ascending=False).reset_index(drop=True)
    ranked.insert(0, "Final_Global_Rank", range(1, len(ranked) + 1))
    top20 = ranked.head(20).copy()
    top20.to_csv(GLOBAL_TOP20_CSV, index=False)
    ranked.head(100).to_csv(RESULTS / "Table_Final_Global_Top100_Structure_Property_Synthesizability_10D.csv", index=False)
    ranked[["Final_Global_Rank", "Molecule", "Source_Group", "SMILES", "Final_Global_D(km/s)", "Final_Global_P(GPa)", "Evidence_Tier"]].head(200).to_csv(
        RESULTS / "Table_Final_Global_Top200_Rank_Index_10D.csv", index=False
    )
    return top20


def draw_structures(top20: pd.DataFrame) -> tuple[Path, Path]:
    mols = [mol(s) for s in top20["SMILES"]]
    legends = []
    for _, r in top20.iterrows():
        legends.append(
            f"{int(r['Final_Global_Rank'])} {r['Molecule']}\n"
            f"D={float(r['Final_Global_D(km/s)']):.2f} km/s | Tier {r['Evidence_Tier']}"
        )
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(360, 300), legends=legends, useSVG=False)
    png = RESULTS / "Figure_Final_Global_Top20_Structures_10D.png"
    img.save(str(png))
    # RDKit grid PDF support is unreliable via PIL; embed PNG in a PDF with matplotlib.
    pdf = RESULTS / "Figure_Final_Global_Top20_Structures_10D.pdf"
    fig, ax = plt.subplots(figsize=(15, 12))
    ax.imshow(img)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_retrosynthesis_inputs(top20: pd.DataFrame) -> dict[str, Path]:
    out = {}
    smi = RESULTS / "Final_Global_Top20_Retrosynthesis_Input.smi"
    top20[["Molecule", "SMILES"]].to_csv(RESULTS / "Final_Global_Top20_Retrosynthesis_Input.csv", index=False)
    with smi.open("w", encoding="utf-8") as fh:
        for _, r in top20.iterrows():
            fh.write(f"{r['SMILES']} {r['Molecule']}\n")
    out["smi"] = smi
    checklist = top20[
        [
            "Final_Global_Rank",
            "Molecule",
            "SMILES",
            "Evidence_Tier",
            "Nearest_Benchmark",
            "Nearest_Benchmark_Tanimoto",
            "Nearest_Benchmark_Scaffold_Tanimoto",
            "SA_Score",
            "Route_Hypothesis",
            "Likely_Precursor_Family",
            "External_Retrosynthesis_Status",
            "Manual_Literature_Precursor_Status",
            "Submission_Claim_Status",
        ]
    ].copy()
    checklist.to_csv(RESULTS / "Table_Final_Global_Top20_Retrosynthesis_Checklist_10D.csv", index=False)
    out["checklist"] = RESULTS / "Table_Final_Global_Top20_Retrosynthesis_Checklist_10D.csv"
    return out


def write_job_templates(inputs: dict[str, Path]) -> dict[str, Path]:
    scripts = ROOT / "scripts"
    ai = scripts / "run_aizynthfinder_final_global_top20_template.slurm"
    ai.write_text(
        f"""#!/usr/bin/env bash
#SBATCH --job-name=aizynth_top20
#SBATCH --output=results/logs/aizynth_top20_%j.out
#SBATCH --error=results/logs/aizynth_top20_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
cd {ROOT}

# This template requires a separate AiZynthFinder installation plus policy/stock files.
# Example expected command after environment setup:
# conda activate retrosynthesis
# aizynthcli --config config/aizynthfinder.yml --smiles {inputs['smi']} --output results/final_global_top20/aizynthfinder_top20_results.json.gz

echo "AiZynthFinder is not installed/configured in the current cluster environment."
echo "Input: {inputs['smi']}"
exit 42
""",
        encoding="utf-8",
    )
    ask = scripts / "run_askcos_final_global_top20_template.py"
    ask.write_text(
        f'''#!/usr/bin/env python3
"""ASKCOS query template for final global Top20.

Requires an accessible ASKCOS v2/v3 endpoint and credentials/API style chosen by the user.
This file intentionally does not hard-code tokens.
"""
from pathlib import Path

INPUT = Path("{inputs['smi']}")
OUTPUT = Path("{ROOT}/results/final_global_top20/askcos_top20_results_pending.json")

print("ASKCOS endpoint/credentials are not configured. Input SMILES:", INPUT)
raise SystemExit(42)
''',
        encoding="utf-8",
    )
    return {"aizynth_slurm": ai, "askcos_template": ask}


def write_docs(top20: pd.DataFrame, figs: tuple[Path, Path], inputs: dict[str, Path], templates: dict[str, Path]) -> None:
    docs = MANUSCRIPT / "submission_docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "README_ENVIRONMENT_AND_REPRODUCIBILITY_AL08_FINAL.md").write_text(
        f"""# Environment and Reproducibility Notes

## Frozen Dataset

- Final database: `data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv` (5432 rows)
- 10D target matrix: `data/baselines/target_matrix_10d.csv`
- xTB aligned feature matrix: `data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv`
- Final production model card: `results/final_model_release/production_v4_10d/PRODUCTION_MODEL_CARD_10D_BDE.txt`

## Current Runtime

- Cluster root: `{ROOT}`
- Conda env used for workflow scripts: `/home/gma/bzhang/software/miniconda3/envs/energetic_gnn`
- Python scripts selected for clean release are staged in `manuscript_npJ/final_submission_package_AL08_20260605/code_release/`.

## Minimal Reproducibility Smoke Test

1. Create/activate an environment with Python, pandas, NumPy, scikit-learn, PyTorch/PyG dependencies used by `03_egnn_painn_train.py`, RDKit, matplotlib, seaborn, xgboost/joblib as required by the release scripts.
2. Verify data integrity:

```bash
python - <<'PY'
import pandas as pd
root = "/home/gma/bzhang/bzhang/Workflow2.0"
for p in [
    "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv",
    "data/baselines/target_matrix_10d.csv",
    "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv",
]:
    df = pd.read_csv(f"{{root}}/{{p}}")
    print(p, df.shape)
PY
```

3. Regenerate final figures/tables that do not require ORCA/xTB:

```bash
python scripts/09_eda_analysis.py
python scripts/finalize_al08_manuscript_package.py
python scripts/build_final_global_top20_and_manuscript_docs.py
```

4. Full quantum reproduction requires ORCA, xTB, Multiwfn, Critic2 and the Slurm scripts in `scripts/`; this is computationally expensive and should be described as provenance rather than a smoke test.

## Retrosynthesis Dependencies

ASKCOS/AiZynthFinder are not currently installed/configured on this cluster. Final Top20 query inputs and templates have been prepared:

- `{inputs['smi'].relative_to(ROOT)}`
- `{templates['aizynth_slurm'].relative_to(ROOT)}`
- `{templates['askcos_template'].relative_to(ROOT)}`
""",
        encoding="utf-8",
    )

    (docs / "DATA_AVAILABILITY_STATEMENT_DRAFT.md").write_text(
        """# Data Availability Statement Draft

The curated final molecular database, 10-target modeling matrix, aligned xTB feature matrix, validation predictions, final Top20 candidate table, and supplementary diagnostic tables generated in this study are available in the project repository / supplementary data package associated with this manuscript. Large raw quantum-chemistry scratch files are not included in the main release because of size, but the final parsed ORCA/BDE/xTB-derived properties and job provenance fields are retained in the released tables. Additional raw calculation files can be made available from the corresponding author upon reasonable request, subject to storage constraints and institutional policies.

Before submission, replace this paragraph with the DOI or reviewer-accessible private link for the deposited dataset.
""",
        encoding="utf-8",
    )

    (docs / "CODE_AVAILABILITY_STATEMENT_DRAFT.md").write_text(
        """# Code Availability Statement Draft

The code used for molecular generation, active-learning selection, ORCA/xTB post-processing, feature alignment, 10-target model training, model diagnostics, density calibration, candidate ranking, and figure/table generation is staged in `manuscript_npJ/final_submission_package_AL08_20260605/code_release/`. The release excludes historical backups, Slurm logs, and large binary model artifacts. Before submission, the cleaned code should be deposited in a persistent public or reviewer-accessible repository together with an environment file and a minimal smoke-test dataset.

Before submission, replace this paragraph with the DOI, GitHub/Zenodo link, or private reviewer link.
""",
        encoding="utf-8",
    )

    (docs / "MANUSCRIPT_OUTLINE_AND_DISPLAY_PLAN_NPJ_FINAL.md").write_text(
        """# Manuscript Outline and Display Plan (npj Computational Materials)

## Title Direction

Physics-informed active learning for closed-loop discovery of high-energy molecular candidates with calibrated quantum validation

## Core Narrative

The paper should be framed as a closed-loop, physics-informed molecular discovery workflow rather than a pure prediction benchmark. The strongest story is: generator -> physics purifier -> quantum validation -> 10D specialist model -> active-learning expansion to a 5432-molecule frozen database -> uncertainty-aware ranking -> conservative synthesizability triage.

## Main Text Structure

1. **Introduction**
   - Need for safer and faster energetic-molecule discovery.
   - Gap: high-throughput generative design often lacks quantum validation, leakage audits, and synthesis-aware triage.
   - Contribution: a frozen 5432-molecule 10D database and active-learning workflow with BDE/xTB-enhanced models and conservative lead prioritization.

2. **Workflow Overview**
   - Describe generation, filtering, ORCA/BDE/xTB calculations, active-learning loops AL04-AL08, and final freeze.
   - **Fig. 1:** workflow schematic; still needs manuscript-quality redraw.

3. **Final Database and Chemical Space**
   - Show source composition and property distributions.
   - **Fig. 2 or Fig. 3:** density-HOF/property-space map from final package; consider redesign into a multi-panel figure with source-group color.
   - **Supplementary Data:** final 5432-row database, target matrix, xTB feature matrix.

4. **10D Model Performance and Robustness**
   - Present final seed42 metrics, parity plot, multi-seed stability, leakage/error-vs-sim, teacher/residual ablation.
   - **Main Fig. 2:** 2 x 5 parity plot.
   - **SI:** multi-seed table, leakage plot, teacher/residual gain.

5. **Quantum-Validated Candidate Ranking**
   - Define final global Top20 from the frozen 5432-molecule database, not only AL08.
   - Explain K-J calculation, density calibration, BDE and uncertainty treatment.
   - **Main Table 1:** final model performance.
   - **Main Table 2 or SI Table:** final global Top20 property/synthesizability table.
   - **Fig. 4:** Top20 structures + D/P/rank uncertainty, assembled from global Top20 structure figure and uncertainty tables.

6. **Synthesizability and Lead Triage**
   - Conservative statement: candidates are computationally prioritized and screened for synthetic plausibility.
   - Do not claim demonstrated synthesizability until ASKCOS/AiZynthFinder/manual route evidence is added.
   - **SI Fig./Table:** strict P6 evidence tiers and route-validation checklist.

7. **Discussion and Limitations**
   - Moderate R2 targets, density calibration limited by 8 benchmarks, route validation pending, raw quantum data storage constraints.
   - Safety/dual-use framing: computational screening, no operational formulations.

## Current Display Assets

- `final_submission_package_AL08_20260605/main_figures/Figure_2_Model_Parity_10D_AL08_seed42.pdf`
- `final_submission_package_AL08_20260605/main_figures/Figure_3_Density_HOF_Chemical_Space_AL08.png`
- `results/final_global_top20/Figure_Final_Global_Top20_Structures_10D.pdf`
- `final_submission_package_AL08_20260605/si_figures/`

## Key Manuscript Boundary

Final naming should say `final frozen database` or `final global Top20`; reserve `AL08` for provenance only.
""",
        encoding="utf-8",
    )


def sync_package(top20: pd.DataFrame, figs: tuple[Path, Path]) -> None:
    if not PACKAGE.exists():
        return
    # Add global-top20 assets to final package without promoting synthesis claims too strongly.
    dst_fig = PACKAGE / "main_figures" / "Figure_4_Final_Global_Top20_Structures_10D.pdf"
    dst_fig.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figs[1], dst_fig)
    dst_png = PACKAGE / "main_figures" / "Figure_4_Final_Global_Top20_Structures_10D.png"
    shutil.copy2(figs[0], dst_png)
    dst_table = PACKAGE / "si_tables" / "Table_S_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv"
    dst_table.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GLOBAL_TOP20_CSV, dst_table)
    query = RESULTS / "Final_Global_Top20_Retrosynthesis_Input.smi"
    if query.exists():
        dst = PACKAGE / "supplementary_data" / "Supplementary_Data_Final_Global_Top20_Retrosynthesis_Input.smi"
        shutil.copy2(query, dst)


def main() -> None:
    top20 = build_global_table()
    figs = draw_structures(top20)
    inputs = write_retrosynthesis_inputs(top20)
    templates = write_job_templates(inputs)
    write_docs(top20, figs, inputs, templates)
    sync_package(top20, figs)
    print(
        json.dumps(
            {
                "global_top20": str(GLOBAL_TOP20_CSV.relative_to(ROOT)),
                "structure_png": str(figs[0].relative_to(ROOT)),
                "structure_pdf": str(figs[1].relative_to(ROOT)),
                "retrosynthesis_input": str(inputs["smi"].relative_to(ROOT)),
                "retrosynthesis_checklist": str(inputs["checklist"].relative_to(ROOT)),
                "aizynth_template": str(templates["aizynth_slurm"].relative_to(ROOT)),
                "askcos_template": str(templates["askcos_template"].relative_to(ROOT)),
                "submission_docs": "manuscript_npJ/submission_docs",
                "package_synced": str(PACKAGE.relative_to(ROOT)),
                "top20_source": "final_frozen_5432_database",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
