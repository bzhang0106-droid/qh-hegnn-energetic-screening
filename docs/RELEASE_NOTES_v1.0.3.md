# QH-HEGNN energetic-molecule screening workflow v1.0.3

This patch release aligns the public processed-data package with the final npj submission data folder after the true-phys/HGS repair and Fig. 4 robustness update.

## Changes relative to v1.0.2

- Synchronized `data/processed/` with the current Supplementary Information data package.
- Added final-scale active-learning `valid10d` labels and true surface-ESP/QTAIM feature tables.
- Added repaired Fig. 4 Random, Scaffold and Butina validation records: long-form errors, R2 summary, metrics, validation-row metadata, split inventory and run-completion QA.
- Retained `Supplementary_Data_4_Validation_Predictions_Final_Hybrid_seed42.csv` for the Fig. 3 observed-predicted parity analysis.
- Removed superseded QTAIM counterfactual and route-audit CSV files from the GitHub processed-data boundary.
- Rebuilt release manifests and package metadata for `v1.0.3`.

## Current release boundary

- `src/`: manuscript-related computational, data-processing and validation scripts.
- `data/processed/`: processed manuscript source/supplementary data needed to inspect the reported results.
- `slurm_templates/`: HPC execution templates for expensive calculations.
- `docs/paper1_code_package/`: environment notes, requirements and manifest.

Full end-to-end reproduction still requires ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and HPC resources. Raw scratch files, binary model artifacts and large compressed parsed-record archives are intentionally excluded from GitHub; the complete supplementary data archive is retained in the manuscript SI package.
