#!/bin/bash
#SBATCH --job-name=03_10d_AL08
#SBATCH --partition=epyc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/03_10d_AL08_%j.log
#SBATCH --error=/home/gma/bzhang/bzhang/Workflow2.0/results/logs/03_10d_AL08_%j.err

set -euo pipefail

echo "=================================================="
echo "03 10D BDE/npj production training started (AL08 final, full xTB aligned)"
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

RUN_TAG="final_specialist_10d_bde_xtbfull_AL08_seed42_20260605"
OUT_DIR="../results/final_model_release/production_v4_10d"
MODEL_PATH="${OUT_DIR}/final_specialist_10d_bde_production.joblib"

mkdir -p "${OUT_DIR}" ../results/logs ../manuscript_npJ/SI/model_diagnostics ../results/sensitivity_v3

BACKUP_STAMP=$(date -u "+%Y%m%d_%H%M%S")
BACKUP_DIR="${OUT_DIR}/pre_AL08_retrain_${BACKUP_STAMP}"
mkdir -p "${BACKUP_DIR}"
for f in \
  "${MODEL_PATH}" \
  "${OUT_DIR}/final_specialist_10d_bde_metrics.csv" \
  "${OUT_DIR}/final_specialist_10d_bde_all_metrics.csv" \
  "${OUT_DIR}/final_specialist_10d_bde_teacher_selection.csv" \
  "${OUT_DIR}/split_final_specialist_10d_bde_production.json" \
  "${OUT_DIR}/PRODUCTION_MODEL_CARD_10D_BDE.txt"; do
  if [ -e "$f" ]; then
    cp -p "$f" "${BACKUP_DIR}/"
  fi
done
echo "[INFO] Existing canonical production model artifacts backed up to ${BACKUP_DIR}"

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
  --sensitivity_output_dir ../results/sensitivity_v3 \
  --export_npj_evidence \
  --npj_evidence_dir ../manuscript_npJ/SI/model_diagnostics \
  --run_tag "${RUN_TAG}" \
  --seed 42 \
  --output_dir "${OUT_DIR}" \
  --final_model_path "${MODEL_PATH}"

SRC_METRICS="${OUT_DIR}/final_specialist_final_metrics_${RUN_TAG}.csv"
SRC_ALL_METRICS="${OUT_DIR}/final_specialist_metrics_${RUN_TAG}.csv"
SRC_TEACHER="${OUT_DIR}/final_specialist_teacher_selection_${RUN_TAG}.csv"
SRC_SPLIT="${OUT_DIR}/split_${RUN_TAG}.json"
SRC_SENS="../results/sensitivity_v3/sensitivity_classification_report_${RUN_TAG}.csv"

for f in "${MODEL_PATH}" "${SRC_METRICS}" "${SRC_TEACHER}" "${SRC_SPLIT}"; do
  if [ ! -s "$f" ]; then
    echo "[ABORT] Expected training artifact missing: $f"
    exit 3
  fi
done

cp -f "${SRC_METRICS}" "${OUT_DIR}/final_specialist_10d_bde_metrics.csv"
cp -f "${SRC_ALL_METRICS}" "${OUT_DIR}/final_specialist_10d_bde_all_metrics.csv"
cp -f "${SRC_TEACHER}" "${OUT_DIR}/final_specialist_10d_bde_teacher_selection.csv"
cp -f "${SRC_SPLIT}" "${OUT_DIR}/split_final_specialist_10d_bde_production.json"
if [ -s "${SRC_SENS}" ]; then
  cp -f "${SRC_SENS}" "${OUT_DIR}/final_specialist_10d_bde_sensitivity_classification.csv"
fi

cat > "${OUT_DIR}/PRODUCTION_MODEL_CARD_10D_BDE.txt" <<EOF
Production model: final_specialist_10d_bde_production.joblib
Updated_at_UTC: $(date -u "+%Y-%m-%dT%H:%M:%SZ")
Source_run_tag: ${RUN_TAG}
AL_iteration: AL08
Training_rows: 5432
Training_table: ../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv
Targets: 10D including Vertical_BDE(kcal/mol)
Feature_policy: weak-bond descriptors enabled from aligned 10D training table; completed BDE label value columns excluded from features to prevent leakage
Residual_policy: conservative alpha gate
Density_label_mode: crystal_preferred
EOF

echo "[INFO] Recording workflow_state for AL08 / iter 8"
python -u 11_log_evaluation.py --iter 8

echo "=================================================="
echo "03 10D BDE/npj production training finished (AL08 final, full xTB aligned)"
echo "End time: $(date)"
echo "Model path: ${MODEL_PATH}"
echo "Metrics: ${OUT_DIR}/final_specialist_10d_bde_metrics.csv"
echo "Run tag: ${RUN_TAG}"
echo "=================================================="
