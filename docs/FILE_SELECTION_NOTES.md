# File selection notes

This release directory was generated from `NPJ_submission/code` and the processed supplementary/source tables already present in the submission package.

## Included

- Core workflow scripts for molecular generation, model training, screening, QTAIM/BDE/ESP parsing and diagnostic analyses.
- Final-freeze analysis scripts that produced source-data tables and model-evidence summaries.
- Selected historical workflow snapshots for provenance.
- Slurm templates documenting the HPC execution context.
- Processed CSV/SMI tables needed by the minimal reviewer check.

## Excluded

- Manuscript DOCX generation, Zotero/reference update and revision automation scripts.
- SFTP helper files, remote probe scripts and local transfer instructions.
- Office documents, PDFs, figures, compressed archives and binary model artifacts.
- Raw ORCA/xTB/Multiwfn/Critic2 scratch outputs and HPC temporary directories.
- Local SSH/Bitvise configuration and credentials.

## Note before public release

Review `CITATION.cff` and replace the placeholder GitHub URL with the final repository URL. After creating a GitHub release, connect the repository to Zenodo or Code Ocean to mint a permanent DOI.
