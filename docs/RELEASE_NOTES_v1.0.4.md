# QH-HEGNN energetic-molecule screening workflow v1.0.4

This patch release makes the public GitHub release wording consistent with npj/Nature-style code-availability expectations after the `v1.0.3` processed-data update.

## Changes relative to v1.0.3

- Clarified that the recorded conda/pip environment files are already included in the release.
- Reworded environment notes so they no longer describe environment export as a pre-publication task.
- Retained the `v1.0.3` processed-data contents, true-phys/HGS repair data, manifest files and release boundary.
- Rebuilt release metadata for `v1.0.4`.

## Current release boundary

- `src/`: manuscript-related computational, data-processing and validation scripts.
- `data/processed/`: processed manuscript source/supplementary data needed to inspect the reported results.
- `slurm_templates/`: HPC execution templates for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and manifest.

Full end-to-end reproduction still requires ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and HPC resources. Raw scratch files, binary model artifacts and large compressed parsed-record archives are intentionally excluded from GitHub; the complete supplementary data archive is retained in the manuscript SI package.
