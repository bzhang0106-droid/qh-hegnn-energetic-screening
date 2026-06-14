from __future__ import annotations

from pathlib import Path

ROOT = Path("/home/gma/bzhang/bzhang/Workflow2.0")
DOCS = ROOT / "manuscript_npJ" / "submission_docs"
DOCS.mkdir(parents=True, exist_ok=True)


def write(name: str, text: str) -> None:
    (DOCS / name).write_text(text.strip() + "\n", encoding="utf-8")


def main() -> None:
    swp = DOCS / ".CODE_AVAILABILITY_STATEMENT_DRAFT.md.swp"
    if swp.exists():
        swp.unlink()

    write(
        "README_ENVIRONMENT_AND_REPRODUCIBILITY_FINAL_FROZEN.md",
        """
# README: Final Frozen Workflow2.0 Reproducibility Package

This document describes the reproducibility entry points for the final frozen Workflow2.0 dataset and manuscript package. Use `AL08` only as provenance for the last active-learning iteration; the manuscript-facing dataset should be called the **final frozen 5432-molecule database**.

## Project Root

`/home/gma/bzhang/bzhang/Workflow2.0`

## Core Final Data

| Item | Path | Status |
|---|---|---|
| Final sorted database | `data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv` | 5432 molecules, sorted by `Final_Detonation_D(km/s)` |
| 10D target matrix | `data/baselines/target_matrix_10d.csv` | 5432 rows, aligned by `Molecule` |
| xTB feature matrix | `data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv` | 5432 rows, aligned by `Molecule` |
| Final model metrics | `results/final_model_release/production_v4_10d/final_specialist_10d_bde_metrics.csv` | final seed42 production metrics |
| Final model card | `results/final_model_release/production_v4_10d/PRODUCTION_MODEL_CARD_10D_BDE.txt` | AL08 provenance, Training rows = 5432 |
| Final global Top20 table | `results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv` | Top20 from full 5432 database |
| Final global Top20 structures | `results/final_global_top20/Figure_Final_Global_Top20_Structures_10D.pdf` | manuscript/SI candidate |
| Clean manuscript package | `manuscript_npJ/final_submission_package_AL08_20260605/` | final working set; package name retains AL08 provenance |

## Runtime Environment

Primary conda environment:

`/home/gma/bzhang/software/miniconda3/envs/energetic_gnn`

Key runtime requirements used by the final scripts include Python, pandas, NumPy, RDKit, matplotlib, seaborn, scikit-learn/joblib, PyTorch/PyG-related model dependencies, ORCA/xTB/Multiwfn/Critic2 for full quantum reproduction, and Slurm for cluster execution.

ASKCOS/AiZynthFinder are **not currently installed/configured** in the checked cluster environment. Prepared inputs/templates:

- `results/final_global_top20/Final_Global_Top20_Retrosynthesis_Input.smi`
- `scripts/run_aizynthfinder_final_global_top20_template.slurm`
- `scripts/run_askcos_final_global_top20_template.py`

## Minimal Smoke Test

This smoke test validates table integrity and regenerates manuscript-facing non-quantum outputs. It does not rerun ORCA/xTB calculations.

```bash
cd /home/gma/bzhang/bzhang/Workflow2.0
PY=/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python

$PY - <<'PY'
import pandas as pd
checks = {
    "final_db": "data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv",
    "target_10d": "data/baselines/target_matrix_10d.csv",
    "xtb_10d": "data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv",
    "top20": "results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv",
}
for name, path in checks.items():
    df = pd.read_csv(path)
    print(name, df.shape)
assert len(pd.read_csv(checks["final_db"])) == 5432
assert len(pd.read_csv(checks["target_10d"])) == 5432
assert len(pd.read_csv(checks["xtb_10d"])) == 5432
assert len(pd.read_csv(checks["top20"])) == 20
PY

$PY scripts/sort_integrate_final_database_by_detonation.py
$PY scripts/build_final_global_top20_and_manuscript_docs.py
$PY scripts/finalize_al08_manuscript_package.py
```

## Full Reproduction Boundary

Full reproduction of the quantum-derived database requires ORCA, xTB, Multiwfn, Critic2, and the corresponding Slurm scripts. The manuscript release should include parsed final data and provenance fields rather than raw scratch directories, because `temp_calc/`, `xtb_calc/`, and `orca_bde_full_library/` are large generated artifacts.
""",
    )

    write(
        "MINIMAL_REPRODUCIBILITY_EXAMPLE_FINAL_FROZEN.md",
        """
# Minimal Reproducibility Example

The goal of the minimal example is to verify that the final frozen database, target matrix, xTB matrix, model metrics, and global Top20 table are internally consistent without rerunning expensive quantum calculations.

## Expected Outputs

- `old_dataset_molecule_clean.csv`: 5432 rows
- `target_matrix_10d.csv`: 5432 rows
- `xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv`: 5432 rows
- `Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv`: 20 rows
- `Figure_Final_Global_Top20_Structures_10D.pdf`: generated from final global Top20

## Commands

```bash
cd /home/gma/bzhang/bzhang/Workflow2.0
PY=/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python

$PY - <<'PY'
import pandas as pd
db = pd.read_csv("data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv")
target = pd.read_csv("data/baselines/target_matrix_10d.csv", usecols=["Molecule"])
xtb = pd.read_csv("data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv", usecols=["Molecule"])
top20 = pd.read_csv("results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv")

assert len(db) == 5432
assert len(target) == 5432
assert len(xtb) == 5432
assert set(db["Molecule"]) == set(target["Molecule"]) == set(xtb["Molecule"])
assert list(db["Molecule"].head(20)) == list(top20["Molecule"])
print(db[["Final_Detonation_Rank", "Molecule", "Final_Detonation_D(km/s)", "Final_Detonation_P(GPa)"]].head(5))
PY
```

## What This Demonstrates

This verifies the final database freeze, molecule-key alignment, final detonation ranking, and global Top20 definition. It does not validate ASKCOS/AiZynthFinder route predictions because those tools are not yet installed/configured in the current cluster environment.
""",
    )

    write(
        "DATA_AVAILABILITY_STATEMENT_DRAFT_FINAL_FROZEN.md",
        """
# Data Availability Statement Draft

The final frozen 5432-molecule database, 10-target matrix, aligned xTB feature matrix, final validation predictions, final global Top20 candidate table, density-calibration/ranking-uncertainty tables, and model-diagnostic supplementary tables generated in this study are available in the manuscript supplementary data package and should be deposited in a public or reviewer-accessible repository before submission.

The current release-ready data entry points are staged under:

- `manuscript_npJ/final_submission_package_AL08_20260605/supplementary_data/`
- `results/final_global_top20/`
- `manuscript_npJ/SI/model_diagnostics/`

Large raw quantum-chemistry scratch files are not included in the main data package because of size. The released tables retain parsed ORCA/BDE/xTB-derived properties and job provenance fields needed to audit the calculations. Additional raw calculation files can be made available from the corresponding author upon reasonable request, subject to storage and institutional constraints.

Before submission, replace this paragraph with the DOI, Zenodo record, institutional repository link, or reviewer-accessible private link.
""",
    )

    write(
        "CODE_AVAILABILITY_STATEMENT_DRAFT_FINAL_FROZEN.md",
        """
# Code Availability Statement Draft

The code used for molecular generation, physics-based filtering, active-learning selection, ORCA/BDE/xTB post-processing, target-matrix construction, model training, model diagnostics, density calibration, global Top20 ranking, synthesizability triage, and figure/table generation is staged in:

`manuscript_npJ/final_submission_package_AL08_20260605/code_release/`

The code-release draft intentionally excludes historical backups, Slurm logs, raw scratch directories, and GB-scale binary model artifacts. Before submission, this directory should be deposited in a persistent public or reviewer-accessible repository together with an environment file, exact command examples, and a minimal smoke-test dataset.

ASKCOS/AiZynthFinder route-validation scripts are currently templates because no configured ASKCOS/AiZynthFinder installation was found in the cluster environment. If route-validation results are used in the manuscript, the exact tool version, policy/stock files, endpoint or local configuration, query settings, and output tables must be included in the repository.

Before submission, replace this paragraph with the DOI, GitHub/Zenodo link, or private reviewer link.
""",
    )

    write(
        "MANUSCRIPT_OUTLINE_AND_DISPLAY_PLAN_NPJ_FINAL_FROZEN.md",
        """
# Manuscript Outline and Display Plan (npj Computational Materials)

## Target-Journal Logic

Comparable npj Computational Materials papers on active-learning molecular discovery and energetic-material de novo design typically show: a closed-loop computational pipeline, database/chemical-space expansion, model benchmarking, Pareto or multi-objective candidate selection, quantum validation, uncertainty or robustness checks, and data/code availability. The current manuscript should follow that structure and avoid over-claiming synthesizability until external route evidence is available.

Relevant comparison points:

- Active-learning molecular generation papers emphasize a closed-loop pipeline, Pareto-front advancement, DFT verification, and surrogate self-correction in new chemical space.
- Energetic-material de novo design papers emphasize multi-objective generation, QM validation, energy/stability trade-off, benchmark comparison, and synthetic feasibility evaluation.
- Nature/npj reporting guidance expects Data Availability and Code Availability sections and discourages unsupported “data not shown” statements.

## Proposed Title Direction

Physics-informed active learning for 10-target discovery of high-energy molecular candidates with calibrated quantum validation

## Central Claim

The study establishes a final frozen 5432-molecule CHNO energetic-molecule database and a closed-loop active-learning workflow that combines generative design, physics filtering, ORCA/BDE/xTB-derived descriptors, 10-target surrogate modeling, density-calibrated detonation ranking, and conservative synthesizability triage.

## Manuscript Structure

### 1. Introduction

Purpose: motivate safer, faster computational discovery of energetic molecules and the need for workflows that combine generative design, quantum validation, BDE/stability information, and transparent ML diagnostics.

Main claim to land: the paper is not only a high-R2 prediction model; it is a closed-loop, evidence-audited discovery workflow.

### 2. Closed-loop Workflow and Final Database

Content:

- Generator and physics purifier.
- ORCA/BDE/xTB validation.
- Active-learning iterations ending at the final frozen database.
- Final database composition: BASE, GPT_AL, AL04-AL08.
- Explain that `AL08` is provenance; manuscript-facing claims use the final frozen 5432-molecule database.

**Figure 1 design:** full workflow schematic.

Recommended panels:

- Fig. 1a: generator -> purifier -> quantum validation -> feature extraction -> model update -> candidate ranking.
- Fig. 1b: cumulative database growth by source group.
- Fig. 1c: final source composition pie/bar chart.

Current assets: source counts are in the sorted final DB; schematic still needs final drawing.

### 3. Chemical Space and Property Landscape

Content:

- Distribution of 10 target properties.
- Density-HOF map and active-learning enrichment of high-performance region.
- Discuss density source/calibration and BDE availability.

**Figure 2 design:** final database chemical/property-space map.

Recommended panels:

- Fig. 2a: density vs heat of formation colored by source group.
- Fig. 2b: final detonation D vs P, highlighting global Top20.
- Fig. 2c: BDE vs detonation D or sensitivity proxy.
- Fig. 2d: distribution/ridge plot for major targets.

Current assets:

- `manuscript_npJ/final_submission_package_AL08_20260605/main_figures/Figure_3_Density_HOF_Chemical_Space_AL08.png`
- `results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv`

### 4. 10D Model Accuracy, Robustness, and Leakage Audit

Content:

- Final seed42 metrics.
- Multi-seed stability seed7/42/123.
- Error-vs-similarity redundancy/leakage audit.
- Teacher vs residual ablation.

**Figure 3 design:** model parity and robustness.

Recommended panels:

- Fig. 3a: 2 x 5 parity plot for the 10 targets.
- Fig. 3b: targetwise R2 bar with multi-seed error bars.
- Fig. 3c: error vs nearest-training/scaffold similarity.
- Fig. 3d: teacher vs residual gain.

Current assets:

- `main_figures/Figure_2_Model_Parity_10D_AL08_seed42.pdf`
- `si_figures/Figure_S_Error_vs_Similarity_AL08.pdf`
- `si_figures/Figure_S_Teacher_Residual_Gain_AL08.pdf`
- `si_tables/Table_S_MultiSeed_Stability_AL08.csv`

### 5. Final Global Top20 Candidate Ranking

Content:

- Define Top20 from all 5432 frozen molecules, not only AL08.
- Ranking by `Final_Detonation_D(km/s)` with `P`, density, HOF, BDE, electronic/sensitivity descriptors, and source group.
- Explain K-J calculation and density-calibration uncertainty.

**Figure 4 design:** final Top20 candidate panel.

Recommended panels:

- Fig. 4a: structure grid of global Top20.
- Fig. 4b: bar/lollipop chart of D and P for Top20.
- Fig. 4c: density-calibration uncertainty interval for D/P.
- Fig. 4d: rank stability or Top20 source-group composition.

Current assets:

- `results/final_global_top20/Figure_Final_Global_Top20_Structures_10D.pdf`
- `results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv`
- `si_figures/Figure_S_Density_Calibration_Uncertainty_AL08.pdf`
- `si_figures/Figure_S_Ranking_Stability_AL08.pdf`

**Main/SI Table design:** final Top20 all-property table.

Columns should include:

- rank, molecule ID, source group, structure/SMILES;
- D, P, density used, HOF, BDE;
- HOMO-LUMO gap, VS_max, Sigma2_tot, Nu, Trigger_Bond_Rho, MW;
- synthetic evidence tier, benchmark similarity, scaffold similarity, route hypothesis, ASKCOS/AiZynthFinder status, manual precursor/literature status.

Current table:

- `results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv`

### 6. Synthesizability and Route-Validation Boundary

Content:

- Current data support computational triage, not demonstrated synthesis.
- SA score alone is insufficient.
- Present strict evidence tiers, benchmark/scaffold similarity, route hypotheses, and external-retrosynthesis status.
- If ASKCOS/AiZynthFinder is later run, add route depth, solved/unsolved status, stock precursor availability, best route score, and route diagrams for top candidates.

**Figure 5 design:** only if route validation is added.

Recommended panels after route validation:

- Fig. 5a: Top lead structures with route-validation status.
- Fig. 5b: solved-route fraction / route depth / precursor availability.
- Fig. 5c: representative route for 3-5 leads.
- Fig. 5d: performance vs synthesizability trade-off.

Current status:

- ASKCOS/AiZynthFinder not installed/configured in the current cluster environment.
- Inputs/templates are ready:
  - `results/final_global_top20/Final_Global_Top20_Retrosynthesis_Input.smi`
  - `scripts/run_aizynthfinder_final_global_top20_template.slurm`
  - `scripts/run_askcos_final_global_top20_template.py`

### 7. Discussion and Limitations

Content:

- Moderate R2 targets should be reported honestly.
- Density calibration uses a small benchmark set; present as sensitivity analysis.
- Route validation is pending.
- Raw quantum files are large; parsed data and provenance are released.
- Safety/dual-use framing: computational property screening only, no operational formulation guidance.

## Proposed Main Tables

| Table | Content | Current source |
|---|---|---|
| Table 1 | Final 10D model performance and target units | `main_tables/Table_1_Final_Model_Performance_10D_AL08.csv` |
| Table 2 | Final global Top20 structures/properties/synthesizability evidence | `results/final_global_top20/Table_Final_Global_Top20_Structure_Property_Synthesizability_10D.csv` |
| Table 3 or SI | Data provenance and source composition | final sorted database |

## Proposed SI Tables/Figures

- Multi-seed stability.
- Redundancy/leakage audit.
- Teacher vs residual ablation.
- Density calibration and uncertainty.
- Final Top20 retrosynthesis checklist.
- AL08 failed/excluded verification rows.
- Full validation predictions.

## Naming Rule

Use `final frozen database`, `final global Top20`, and `5432-molecule database` in manuscript-facing text. Use `AL08` only for workflow provenance or file names retained from the computational pipeline.
""",
    )

    print("Updated final frozen submission docs in", DOCS)


if __name__ == "__main__":
    main()
