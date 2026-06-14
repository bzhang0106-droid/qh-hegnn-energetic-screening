# QH-HEGNN energetic-molecule screening workflow

This repository contains the code release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware neural-network screening of energetic molecules.

## What is included

- `src/`: core scripts for molecular screening, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection, OOD diagnostics, and manuscript figure/source-data generation.
- `analysis/final_freeze_automation/`: final-freeze analysis and ranking scripts used to assemble reviewer-facing evidence.
- `legacy_workflow_snapshot/`: selected historical workflow scripts retained for provenance.
- `data/processed/`: processed source/supplementary tables copied from the submission package for lightweight reproducibility checks.
- `scripts/minimal_reproduce_reviewer_check.py`: a small check that validates key table dimensions and regenerates compact preview outputs.
- `slurm_templates/`: example Slurm templates documenting HPC execution modes.

## Quick start

```bash
conda env create -f docs/paper1_code_package/environment_energetic_gnn_no_builds.yml
conda activate energetic_gnn
python scripts/minimal_reproduce_reviewer_check.py
```

If the exported environment cannot be used directly, install the lightweight reviewer-check dependencies manually:

```bash
pip install pandas matplotlib
python scripts/minimal_reproduce_reviewer_check.py
```

The check writes outputs to `reviewer_minimal_check/`.

## Reproducibility boundary

The included scripts reproduce parsed-table analyses and lightweight source-data checks from released CSV/SMI files. Full end-to-end quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies, and access to suitable HPC resources. Some provenance scripts retain path conventions from the original computing environment and should be configured before rerunning on a new machine.

## Citation and DOI

For npj/Nature Portfolio submission, archive a tagged GitHub release through Zenodo or Code Ocean and add the resulting DOI to the manuscript Code availability statement and reference list.
