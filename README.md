# QH-HEGNN energetic-molecule screening workflow

This repository contains the formal code and processed-data release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware neural-network screening of energetic molecules.

## What is included

- `src/`: core scripts for molecular curation, QH-HEGNN/PaiNN training, physics-feature extraction, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection and model-validation diagnostics.
- `data/processed/`: processed manuscript data, supplementary tables and source-data tables needed to inspect the reported results.
- `slurm_templates/`: Slurm templates documenting the HPC execution modes used for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and the current code-package manifest.

## Use

Create the recorded Python environment, then run the relevant source scripts for the manuscript result being inspected:

```bash
conda env create -f docs/paper1_code_package/environment_energetic_gnn_no_builds.yml
conda activate energetic_gnn
```

The canonical model-training entry point is `src/03_egnn_painn_train.py`. Controlled Random/Scaffold/Butina validation runs use `src/generate_standard_validation_splits.py` and `src/run_qh_hegnn_controlled_split.py`; the corresponding processed validation summaries are provided in `data/processed/`.

## Reproducibility boundary

The repository contains code and processed data directly related to the manuscript calculations, source tables and model-validation results. Manuscript-writing, reference-management, smoke-test wrappers, historical workflow snapshots and draft plotting scripts are intentionally excluded. Full end-to-end quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and access to suitable HPC resources. Some scripts retain path conventions from the original computing environment and should be configured before rerunning on a new machine.

## Citation and DOI

Current local package version: `v1.0.3` (2026-06-26).

- All-versions DOI: https://doi.org/10.5281/zenodo.20685579

Mint a new Zenodo archive for `v1.0.3` before final submission if the manuscript Code availability section needs a fixed version DOI rather than the GitHub URL plus concept DOI.
