# Code Release

This directory contains the computational code subset for the npj Computational Materials manuscript.

## Contents

- Core workflow scripts for molecular generation, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection, OOD diagnostics and source-data generation.
- `03_egnn_painn_train_truephys_hgs_repair_20260625.py`, `qh_hegnn_truephys_hgs_repair_wrapper_20260625.py`, `true_phys_feature_extractor_20260624.py`, `generate_truephys_hgs_standard_splits_20260626.py` and `plot_fig4_truephys_hgs_standard_3split_release_20260626.py`: scripts tied to the updated true-phys/HGS Fig. 4 evaluation.
- `redesign_figure1a_screening_stream_refined_20260626.py`: source-data-backed script for the updated Fig. 1a workflow panel.
- `final_freeze_automation/`: retained computational diagnostics, ranking and source-table scripts.
- `slurm_templates/`: Slurm templates documenting the HPC execution modes used for true-phys feature extraction, true-phys/HGS evaluation and other expensive calculations.

## Excluded By Design

- Raw ORCA, xTB, Multiwfn, Critic2, BDE, and Slurm scratch directories.
- Binary model artifacts such as `*.joblib`.
- Large Office/PDF/image files not central to reproducible code review.
- Private credentials, tokens, and local-only SSH configuration.
- Manuscript-writing, reference-management, DOCX/SI-generation and superseded draft plotting scripts.
- Historical backups and local archive folders.

## Reproducibility Boundary

The released code supports reproducing parsed-table analyses and manuscript figure source data from released CSV files. Full end-to-end quantum and graph-learning reproduction requires ORCA, xTB, Multiwfn, Critic2, Slurm, PyTorch/PyG-style graph-learning dependencies and substantial compute resources.
