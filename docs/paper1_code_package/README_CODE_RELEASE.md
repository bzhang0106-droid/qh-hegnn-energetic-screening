# Code Release

This directory documents the computational code subset for the npj Computational Materials manuscript.

## Contents

- Core workflow scripts for molecular generation, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE merging, Pareto selection, applicability-domain diagnostics and processed source-data generation.
- `03_egnn_painn_train.py`: canonical final training script for the QH-HEGNN/PaiNN model family.
- `generate_standard_validation_splits.py` and `run_qh_hegnn_controlled_split.py`: controlled Random, Scaffold and Butina validation helpers.
- `true_phys_feature_extractor_20260624.py`: physics-feature extraction script for surface-ESP/QTAIM descriptors used by the final model.
- `slurm_templates/`: Slurm templates documenting the HPC execution modes used for physics-feature extraction, model validation and other expensive calculations.

## Excluded By Design

- Raw ORCA, xTB, Multiwfn, Critic2, BDE and Slurm scratch directories.
- Binary model artifacts such as `*.joblib`.
- Large Office/PDF/image files not central to reproducible code review.
- Private credentials, tokens and local-only SSH configuration.
- Manuscript-writing, reference-management, DOCX/SI-generation, smoke-test wrappers and draft plotting scripts.
- Historical workflow snapshots, backups and local archive folders.

## Reproducibility Boundary

The released code supports reproducing parsed-table analyses, model-validation summaries and manuscript source data from released CSV files. Full end-to-end quantum and graph-learning reproduction requires ORCA, xTB, Multiwfn, Critic2, Slurm, PyTorch/PyG-style graph-learning dependencies and substantial compute resources.
