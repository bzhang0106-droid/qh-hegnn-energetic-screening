# Code Release Draft

This directory stages a reviewer/repository code subset for the npj Computational Materials manuscript.

## Contents

- `final_freeze_automation/`: scripts created for the AL08/final-freeze manuscript package, diagnostics, ranking, and final documentation.
- `workflow_scripts_local_snapshot/`: local workflow scripts from `scrips/` and selected script snapshots from `results/`.
- `slurm_templates/`: Slurm submission templates used or prepared for remote Workflow2.0 runs.

## Excluded By Design

- Raw ORCA, xTB, Multiwfn, Critic2, BDE, and Slurm scratch directories.
- Binary model artifacts such as `*.joblib`.
- Large Office/PDF/image files not central to reproducible code review.
- Private credentials, tokens, and local-only SSH configuration.
- Historical backups and archive folders.

## Before Public Deposition

1. Add a license.
2. Add an environment file, ideally `environment.yml` or `requirements.txt` plus CUDA/PyTorch/PyG notes.
3. Replace hard-coded absolute remote paths with configurable paths where feasible.
4. Add a small smoke-test dataset that exercises parsing, table generation, and figure generation without running full quantum calculations.
5. Deposit the cleaned repository in GitHub and archive a release in Zenodo or Code Ocean for a DOI.

## Reproducibility Boundary

The released code supports reproducing parsed-table analyses and manuscript figures from released CSV data. Full end-to-end quantum reproduction requires ORCA, xTB, Multiwfn, Critic2, Slurm, and substantial compute resources; this should be described as computational provenance rather than a short smoke test.

