#!/bin/bash
#SBATCH --job-name=sum_seed_AL08
#SBATCH --partition=plat
#SBATCH --reservation=bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/sum_seed_AL08_%j.log
#SBATCH --error=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/sum_seed_AL08_%j.err

set -euo pipefail
cd /home/gma/bzhang/bzhang/Workflow2.0/scripts
/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python -u summarize_multiseed_al08_20260605.py
