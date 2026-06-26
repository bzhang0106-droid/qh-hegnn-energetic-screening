# File selection notes

This release directory was generated from `NPJ_submission/code` and the processed supplementary/source tables already present in the submission package.

## Included

- Core workflow scripts for molecular generation, model training, screening, QTAIM/BDE/ESP parsing and diagnostic analyses.
- Controlled validation scripts for Random, Scaffold and Butina split families.
- Slurm templates documenting the HPC execution context.
- Processed CSV/SMI tables needed to inspect the manuscript results and source data.

## Excluded

- Manuscript DOCX generation, Zotero/reference update and revision automation scripts.
- Smoke-test wrappers and auxiliary preview/check scripts.
- Historical workflow snapshots and draft plotting scripts.
- SFTP helper files, remote probe scripts and local transfer instructions.
- Office documents, PDFs, figures, compressed archives and binary model artifacts.
- Raw ORCA/xTB/Multiwfn/Critic2 scratch outputs and HPC temporary directories.
- Local SSH/Bitvise configuration and credentials.

## Note before public release

After creating a GitHub release, connect the repository to Zenodo or Code Ocean to mint a permanent DOI and add that DOI to the manuscript Code availability statement and reference list.
