# QH-HEGNN energetic-molecule screening workflow

This repository contains the code release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware neural-network screening of energetic molecules.

## What is included

- `src/`: core scripts for molecular screening, QH-HEGNN/PaiNN training, true-phys/HGS evaluation, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection, OOD diagnostics, and manuscript figure/source-data generation.
- `analysis/final_freeze_automation/`: retained computational analysis and ranking scripts used to assemble reviewer-facing evidence.
- `legacy_workflow_snapshot/`: selected historical workflow scripts retained for provenance.
- `data/processed/`: processed source/supplementary tables copied from the submission package, including the updated Fig. 1 and Fig. 4 source-data files.
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

The check writes outputs to `reviewer_minimal_check/` and verifies the updated Fig. 1/Fig. 4 source-data tables in addition to the curated database and Top20 evidence files.

## Reproducibility boundary

The included scripts reproduce parsed-table analyses and lightweight source-data checks from released CSV/SMI files. Manuscript-writing, reference-management and superseded draft plotting scripts are intentionally excluded from this cleaned release. Full end-to-end quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies, and access to suitable HPC resources. Some provenance scripts retain path conventions from the original computing environment and should be configured before rerunning on a new machine.

## Citation and DOI

The archived manuscript code release is available on Zenodo:

- Version DOI: https://doi.org/10.5281/zenodo.20685580
- All-versions DOI: https://doi.org/10.5281/zenodo.20685579

Please cite the fixed version DOI for the manuscript release.
