# Environment and Reproducibility Notes

## Observed Environment

- Remote workflow root used by scripts: `/home/gma/bzhang/bzhang/Workflow2.0`
- Remote conda activation path referenced by scripts: `/home/gma/bzhang/software/miniconda3/etc/profile.d/conda.sh`
- Remote conda environment referenced by Slurm scripts: `energetic_gnn`
- Local Codex Python used for package inspection: `C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

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
conda env export --no-builds > manuscript_npJ/final_submission_package_AL08_20260605/code_release/environment_energetic_gnn_no_builds.yml
python -V > manuscript_npJ/final_submission_package_AL08_20260605/code_release/python_version.txt
```

## Minimal Verification

For reviewer access, the repository should provide a small command that:

1. reads the released final frozen database CSV,
2. verifies expected row and column counts,
3. regenerates one small summary table,
4. regenerates one non-sensitive figure from released data.

