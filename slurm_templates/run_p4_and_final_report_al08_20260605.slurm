#!/bin/bash
#SBATCH --job-name=p4_report_AL08
#SBATCH --partition=plat
#SBATCH --reservation=bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/p4_report_AL08_%j.log
#SBATCH --error=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/p4_report_AL08_%j.err

set -euo pipefail
cd /home/gma/bzhang/bzhang/Workflow2.0/scripts
/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python -u generate_al08_p4_ranking_stability.py
/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python -u write_final_al08_completion_report.py
