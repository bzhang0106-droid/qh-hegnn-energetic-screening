# QH-HEGNN energetic-molecule screening workflow v1.0.1

This is the Zenodo-triggering archival release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware graph-neural-network screening of energetic molecules.

## Changes relative to v1.0.0

- Updated `CITATION.cff` version metadata from `1.0.0` to `1.0.1`.
- Added this release-notes file so the Zenodo-ingested GitHub release has explicit archival context.
- Refreshed the local manuscript/repository package on 2026-06-26 with the final true-phys/HGS Random, Scaffold and Butina Fig. 4 evaluation scripts and source-data tables.
- Added the refined Fig. 1a source-data-backed plotting script and source CSV.
- Removed manuscript-writing, reference-management, DOCX/SI-generation and superseded draft plotting scripts from the cleaned release boundary.
- Extended the lightweight reproducibility check to validate the updated Fig. 1/Fig. 4 source-data files.

## Included in this release

- Core Python scripts for molecular generation, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE post-processing, Pareto screening, OOD diagnostics and manuscript source-data generation.
- Final-freeze analysis scripts used to assemble model evidence, ranking stability checks and source tables.
- Selected historical workflow snapshots retained for computational provenance.
- Processed source and supplementary tables required by the lightweight reviewer reproducibility check, including the updated Fig. 1 and Fig. 4 source data.
- Slurm templates documenting HPC execution modes.
- Repository metadata, including `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.gitattributes`, file manifest and a minimal reproducibility checklist.

## Lightweight reproducibility check

After installing the lightweight dependencies, run:

```bash
python scripts/minimal_reproduce_reviewer_check.py
```

The check verifies the curated 5432-molecule table, the QTAIM-aware Top20 table, the model parity/source-metric table and the updated Fig. 1/Fig. 4 source-data tables, then regenerates compact preview outputs under `reviewer_minimal_check/`.

## Reproducibility boundary

The repository supports reproducibility from processed manuscript tables and documents the end-to-end workflow. Full quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and access to suitable HPC resources. Raw quantum-chemistry scratch files, licensed executables, credentials, office documents, superseded draft figures, manuscript-writing scripts and binary model artifacts are intentionally excluded.
