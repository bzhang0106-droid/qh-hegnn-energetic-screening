#!/usr/bin/env bash
#SBATCH --job-name=true_phys_feat
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --array=0-99%8
#SBATCH --output=logs/physics_features_%A_%a.out
#SBATCH --error=logs/physics_features_%A_%a.err

set -euo pipefail

ROOT=${QH_HEGNN_PHYSICS_FEATURE_ROOT:-/scratch/gma/bzhang/qh_hegnn_physics_features}
PY=${PY:-/home/gma/bzhang/software/miniconda3/envs/energetic_gnn/bin/python}
SCRIPT=${SCRIPT:-$ROOT/scripts/true_phys_feature_extractor_20260624.py}
MANIFEST=${MANIFEST:-$ROOT/physics_feature_manifest.csv}
CHUNK_SIZE=${CHUNK_SIZE:-57}

mkdir -p "$ROOT/logs" "$ROOT/features_parts" "$ROOT/work"

echo "===== physics-feature extraction ====="
echo "job=${SLURM_JOB_ID:-manual} task=${SLURM_ARRAY_TASK_ID:-0}"
echo "host=$(hostname)"
echo "root=$ROOT"
echo "manifest=$MANIFEST"
echo "chunk_size=$CHUNK_SIZE"
date -u

"$PY" "$SCRIPT" extract \
  --manifest "$MANIFEST" \
  --out-root "$ROOT" \
  --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --chunk-size "$CHUNK_SIZE" \
  --threads "${SLURM_CPUS_PER_TASK:-2}"

date -u
echo "DONE"
