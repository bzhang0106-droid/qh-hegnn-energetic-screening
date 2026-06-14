#!/bin/bash
#SBATCH -J post08a_AL08
#SBATCH -p plat
#SBATCH --reservation=bin
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /home/gma/bzhang/bzhang/Workflow2.0/results/logs/08a_AL08_%j.log
#SBATCH -e /home/gma/bzhang/bzhang/Workflow2.0/results/logs/08a_AL08_%j.err

set -eo pipefail
echo "[START] post08a_AL08 $(date) host=$(hostname)"
cd /home/gma/bzhang/bzhang/Workflow2.0/scripts
mapfile -t DIRS < /home/gma/bzhang/bzhang/Workflow2.0/data/al08_08a_dirs_20260605.txt
echo "[INFO] dirs=${#DIRS[@]}"
/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python -u 08a_run_multiwfn_critic2.py --dirs "${DIRS[@]}" --max_workers 8
/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python /home/gma/bzhang/bzhang/Workflow2.0/scripts/check_08a_al08_ready_20260605.py
echo "[DONE] post08a_AL08 $(date)"
