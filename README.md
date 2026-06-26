# QH-HEGNN energetic-molecule screening workflow

This repository contains the code release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware neural-network screening of energetic molecules.

## What is included

- `src/`: core scripts for molecular screening, QH-HEGNN/PaiNN training, true-phys/HGS evaluation, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection, OOD diagnostics, and manuscript figure/source-data generation.
- `data/processed/`: processed source/supplementary tables copied from the submission package, including the updated Fig. 1 and Fig. 4 source-data files.
- `slurm_templates/`: Slurm templates documenting the HPC execution modes used for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and the current code-package manifest.

## Use

Create the recorded Python environment, then run the relevant source scripts for the manuscript result being inspected:

```bash
conda env create -f docs/paper1_code_package/environment_energetic_gnn_no_builds.yml
conda activate energetic_gnn
```

The true-phys/HGS Fig. 4 evaluation uses the scripts `src/true_phys_feature_extractor_20260624.py`, `src/03_egnn_painn_train_truephys_hgs_repair_20260625.py`, `src/qh_hegnn_truephys_hgs_repair_wrapper_20260625.py`, `src/generate_truephys_hgs_standard_splits_20260626.py` and `src/plot_fig4_truephys_hgs_standard_3split_release_20260626.py`. The updated Fig. 1 source panel uses `src/redesign_figure1a_screening_stream_refined_20260626.py`.

## Reproducibility boundary

The repository contains code and processed data directly related to the manuscript calculations, source tables and final figure generation. Manuscript-writing, reference-management, smoke-test wrappers, historical workflow snapshots and superseded draft plotting scripts are intentionally excluded. Full end-to-end quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies, and access to suitable HPC resources. Some scripts retain path conventions from the original computing environment and should be configured before rerunning on a new machine.

## Citation and DOI

Current local package version: `v1.0.2` (2026-06-26).

- All-versions DOI: https://doi.org/10.5281/zenodo.20685579

Mint a new Zenodo archive for `v1.0.2` before final submission if the manuscript Code availability section needs a fixed version DOI rather than the GitHub URL plus concept DOI.
