# Environment and Reproducibility Notes

## Observed Environment

- Remote workflow root used by scripts: `/home/gma/bzhang/bzhang/Workflow2.0`
- Remote conda activation path referenced by scripts: `/home/gma/bzhang/software/miniconda3/etc/profile.d/conda.sh`
- Remote conda environment referenced by Slurm scripts: `energetic_gnn`

## Expected Python Dependencies

The code release likely requires:

- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- rdkit
- torch
- torch-geometric or equivalent graph neural network dependencies
- xgboost
- joblib

The exact pinned environment should be exported from the remote `energetic_gnn` environment before public deposition.

## Suggested Remote Export Command

```bash
source /home/gma/bzhang/software/miniconda3/etc/profile.d/conda.sh
conda activate energetic_gnn
conda env export --no-builds > docs/paper1_code_package/environment_energetic_gnn_no_builds.yml
python -V > docs/paper1_code_package/python_version.txt
```

## Reproducibility Boundary

The package provides manuscript-related scripts and processed source data. It does not include a separate smoke-test wrapper; readers should inspect or rerun the script corresponding to the manuscript result of interest.
