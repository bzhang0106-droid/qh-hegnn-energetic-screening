# QH-HEGNN energetic-molecule screening workflow v1.0.1

This is the Zenodo-triggering archival release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware graph-neural-network screening of energetic molecules.

## Changes relative to v1.0.0

- Updated `CITATION.cff` version metadata from `1.0.0` to `1.0.1`.
- Added this release-notes file so the Zenodo-ingested GitHub release has explicit archival context.
- No core workflow code, processed source table, training script or reproducibility-check logic was changed.

## Included in this release

- Core Python scripts for molecular generation, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE post-processing, Pareto screening, OOD diagnostics and manuscript source-data generation.
- Final-freeze analysis scripts used to assemble model evidence, ranking stability checks and source tables.
- Selected historical workflow snapshots retained for computational provenance.
- Processed source and supplementary tables required by the lightweight reviewer reproducibility check.
- Slurm templates documenting HPC execution modes.
- Repository metadata, including `README.md`, `LICENSE`, `CITATION.cff`, `.gitignore`, `.gitattributes`, file manifest and a minimal reproducibility checklist.

## Lightweight reproducibility check

After installing the lightweight dependencies, run:

```bash
python scripts/minimal_reproduce_reviewer_check.py
```

The check verifies the curated 5432-molecule table, the QTAIM-aware Top20 table and the model parity/source-metric table, then regenerates compact preview outputs under `reviewer_minimal_check/`.

## Reproducibility boundary

The repository supports reproducibility from processed manuscript tables and documents the end-to-end workflow. Full quantum-chemical reproduction requires separately installed ORCA, xTB, Multiwfn, Critic2, graph-learning dependencies and access to suitable HPC resources. Raw quantum-chemistry scratch files, licensed executables, credentials, office documents, figure files and binary model artifacts are intentionally excluded.
