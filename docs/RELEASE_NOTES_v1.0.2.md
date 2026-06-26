# QH-HEGNN energetic-molecule screening workflow v1.0.2

This cleaned package refreshes the manuscript code/data release after the final true-phys/HGS Fig. 4 evaluation and refined Fig. 1 update.

## Changes relative to v1.0.1

- Added the final true-phys/HGS scripts used for the Random, Scaffold and Butina Fig. 4 evaluation.
- Added source data for the updated Fig. 1 and Fig. 4 panels.
- Updated package metadata to `v1.0.2` with release date 2026-06-26.
- Removed smoke-test wrappers, manuscript-writing scripts, reference-management scripts, historical workflow snapshots and superseded draft plotting scripts from the distributed release boundary.
- Kept only final figure source scripts that are directly tied to manuscript figures.

## Current release boundary

- `src/`: manuscript-related computational, data-processing and final figure source scripts.
- `data/processed/`: processed manuscript source/supplementary data needed to inspect the reported results.
- `slurm_templates/`: HPC execution templates for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and manifest.

Full end-to-end reproduction still requires ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and HPC resources. Raw scratch files and binary model artifacts are intentionally excluded.
