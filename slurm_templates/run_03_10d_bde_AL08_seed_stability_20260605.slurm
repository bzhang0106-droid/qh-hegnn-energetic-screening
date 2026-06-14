#!/bin/bash
#SBATCH --job-name=03_10d_AL08_seed
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=1-2%2
#SBATCH --output=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/03_10d_AL08_seed_%A_%a.log
#SBATCH --error=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/03_10d_AL08_seed_%A_%a.err

set -euo pipefail

SEEDS=(7 123)
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID-1))]}"

echo "=================================================="
echo "03 10D BDE/npj seed-stability training started (AL08 final)"
echo "Seed: ${SEED}"
echo "Start time: $(date)"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "=================================================="

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate energetic_gnn

cd /home/gma/bzhang/bzhang/Workflow2.0/scripts

export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

RUN_TAG="final_specialist_10d_bde_xtbfull_AL08_seed${SEED}_20260605"
OUT_DIR="../results/final_model_release/production_v4_10d/seed_stability_AL08"
MODEL_PATH="${OUT_DIR}/final_specialist_10d_bde_seed${SEED}.joblib"

mkdir -p "${OUT_DIR}" ../results/logs ../manuscript_npJ/SI/model_diagnostics ../results/sensitivity_v3/seed_stability_AL08

python -u 03_egnn_painn_train.py \
  --model_variant final_specialist \
  --epochs 120 \
  --batch_size 64 \
  --target_profile sensitivity \
  --density_label_mode crystal_preferred \
  --loss_weight_mode focus_electronic \
  --early_stopping_patience 25 \
  --coord_noise_std 0.0 \
  --coord_noise_prob 0.0 \
  --teacher_n_jobs 24 \
  --teacher_cv_folds 5 \
  --teacher_top_k 4 \
  --include_mlp_teacher \
  --xtb_feature_path ../data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv \
  --enable_bde_features \
  --bde_feature_path ../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv \
  --residual_policy conservative \
  --residual_min_corr 0.05 \
  --alpha_max 0.75 \
  --alpha_step 0.025 \
  --alpha_min_improvement 0.005 \
  --sensitivity_v3 \
  --sensitivity_output_dir ../results/sensitivity_v3/seed_stability_AL08 \
  --export_npj_evidence \
  --npj_evidence_dir ../manuscript_npJ/SI/model_diagnostics \
  --run_tag "${RUN_TAG}" \
  --seed "${SEED}" \
  --output_dir "${OUT_DIR}" \
  --final_model_path "${MODEL_PATH}"

SRC_METRICS="${OUT_DIR}/final_specialist_final_metrics_${RUN_TAG}.csv"
SRC_TEACHER="${OUT_DIR}/final_specialist_teacher_selection_${RUN_TAG}.csv"
SRC_SPLIT="${OUT_DIR}/split_${RUN_TAG}.json"

for f in "${MODEL_PATH}" "${SRC_METRICS}" "${SRC_TEACHER}" "${SRC_SPLIT}"; do
  if [ ! -s "$f" ]; then
    echo "[ABORT] Expected seed-stability artifact missing: $f"
    exit 3
  fi
done

echo "=================================================="
echo "03 10D BDE/npj seed-stability training finished"
echo "End time: $(date)"
echo "Seed: ${SEED}"
echo "Metrics: ${SRC_METRICS}"
echo "Run tag: ${RUN_TAG}"
echo "=================================================="
