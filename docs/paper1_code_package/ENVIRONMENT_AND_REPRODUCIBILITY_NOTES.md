# Environment and Reproducibility Notes

## Observed Environment

- Remote workflow root used by scripts: `/home/gma/bzhang/bzhang/Workflow2.0`
- Remote conda activation path referenced by scripts: `/home/gma/bzhang/software/miniconda3/etc/profile.d/conda.sh`
- Remote conda environment referenced by Slurm scripts: `energetic_gnn`

## Recorded Python Environment

This release provides a lightweight conda environment file and a pip-style requirements file:

- `docs/paper1_code_package/environment_energetic_gnn_no_builds.yml`
- `docs/paper1_code_package/requirements.txt`

The Python dependency set includes pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, RDKit, PyTorch, PyTorch Geometric or an equivalent graph-learning stack, XGBoost, joblib, Pillow and python-docx.

## Optional Environment Refresh Command

The current release is ready for code and processed-data inspection. If the authors later rerun the remote workflow and want to refresh the environment snapshot, use:

```bash
source /home/gma/bzhang/software/miniconda3/etc/profile.d/conda.sh
conda activate energetic_gnn
conda env export --no-builds > docs/paper1_code_package/environment_energetic_gnn_no_builds.yml
python -V > docs/paper1_code_package/python_version.txt
```

## Reproducibility Boundary

The package provides manuscript-related scripts and processed source data. It does not include a separate smoke-test wrapper. Full end-to-end reruns require local configuration of ORCA, xTB, Multiwfn, Critic2, Slurm and graph-learning dependencies; readers should inspect or rerun the script corresponding to the manuscript result of interest.
