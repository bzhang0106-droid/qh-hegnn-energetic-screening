# QH-HEGNN energetic-molecule screening workflow v1.0.0

This is the first archival code release for the npj Computational Materials manuscript on QTAIM-guided, weakest-bond-aware graph-neural-network screening of energetic molecules.

## Included in this release

- Core Python scripts for molecular generation, QH-HEGNN/PaiNN training, xTB/ORCA/Multiwfn/Critic2 parsing, QTAIM/BDE post-processing, Pareto screening, OOD diagnostics, and manuscript source-data generation.
- Final-freeze analysis scripts used to assemble the reviewer-facing model evidence, ranking stability checks and final source tables.
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

## Archival DOI

For journal submission, archive this tagged release through Zenodo or Code Ocean and cite the resulting DOI in the manuscript Code availability statement and reference list.
