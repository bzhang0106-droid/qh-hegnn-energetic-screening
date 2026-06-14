# Minimal reproducibility checklist

Use this checklist before pushing the repository to GitHub and before creating a DOI-bearing release.

## Repository metadata

- [ ] `README.md` explains the workflow, included files and reproducibility boundary.
- [ ] `LICENSE` has been approved by all authors or replaced with the preferred open-source license.
- [ ] `CITATION.cff` has the final GitHub URL, author metadata and release DOI after Zenodo/Code Ocean archiving.
- [ ] `requirements.txt` or `environment_energetic_gnn_no_builds.yml` is present.

## Data and test files

- [ ] `data/processed/` contains the processed CSV/SMI files copied from the submission package.
- [ ] No raw ORCA, xTB, Multiwfn, Critic2, GBW, density, Hessian or Slurm scratch files are committed.
- [ ] No Bitvise, SSH, SFTP, token, password or private local configuration files are committed.

## Lightweight check

Run:

```bash
python scripts/minimal_reproduce_reviewer_check.py
```

Expected outputs:

- `reviewer_minimal_check/minimal_reviewer_check_summary.csv`
- `reviewer_minimal_check/minimal_top20_preview.csv`
- `reviewer_minimal_check/minimal_parity_metric_preview.png` or `minimal_plot_warning.txt`

Expected checks:

- curated molecule table has 5432 rows.
- QTAIM-aware Top20 table has at least 20 rows.
- model parity/source metric table is readable.

## Before public release

- [ ] Create a GitHub repository and push this directory.
- [ ] Tag a release, for example `v1.0.0`.
- [ ] Archive the release in Zenodo or Code Ocean and obtain a DOI.
- [ ] Update the manuscript Code availability statement and reference list with the DOI.
