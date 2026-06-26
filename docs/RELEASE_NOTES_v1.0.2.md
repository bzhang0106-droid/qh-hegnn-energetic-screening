# QH-HEGNN energetic-molecule screening workflow v1.0.2

This package refreshes the manuscript code/data release as a formal calculation and validation bundle.

## Changes relative to v1.0.1

- Consolidated the final model-training implementation into the canonical `src/03_egnn_painn_train.py` entry point.
- Added controlled Random, Scaffold and Butina validation split generation and execution helpers.
- Added processed source-data tables and target-wise validation summaries for the final manuscript results.
- Updated package metadata to `v1.0.2` with release date 2026-06-26.
- Removed manuscript-writing scripts, reference-management scripts, smoke-test wrappers, historical workflow snapshots, draft plotting scripts and local automation artifacts from the distributed release boundary.

## Current release boundary

- `src/`: manuscript-related computational, data-processing and validation scripts.
- `data/processed/`: processed manuscript source/supplementary data needed to inspect the reported results.
- `slurm_templates/`: HPC execution templates for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and manifest.

Full end-to-end reproduction still requires ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and HPC resources. Raw scratch files and binary model artifacts are intentionally excluded.
