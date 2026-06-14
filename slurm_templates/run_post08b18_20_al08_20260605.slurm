#!/bin/bash
#SBATCH -J post08b_AL08
#SBATCH -p plat
#SBATCH --reservation=bin
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o /home/gma/bzhang/bzhang/Workflow2.0/results/logs/post08b_AL08_%j.log
#SBATCH -e /home/gma/bzhang/bzhang/Workflow2.0/results/logs/post08b_AL08_%j.err

set -eo pipefail
ROOT=/home/gma/bzhang/bzhang/Workflow2.0
PY=/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python
TAG=AL08_20260605
echo "[START] post08b_AL08 $(date) host=$(hostname)"
cd "$ROOT/scripts"
$PY "$ROOT/scripts/check_08a_al08_ready_20260605.py"
$PY -u 08b_ultimate_merge.py
if [ -d "$ROOT/data/backups" ]; then
  find "$ROOT/data/backups" -maxdepth 1 -type f -name "old_dataset_molecule_clean*.bak.csv" -delete
fi
$PY -u 18_build_10d_target_matrix.py
$PY -u 20_prepare_xtb_missing_for_current_db.py --tag "$TAG"
$PY "$ROOT/scripts/post_al08_counts_20260605.py"
echo "[DONE] post08b_AL08 $(date)"
