#!/bin/bash
#SBATCH --job-name=qh_std_eval
#SBATCH --partition=epyc,long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/qh_standard_validation_%A_%a.out
#SBATCH --error=logs/qh_standard_validation_%A_%a.err

set -euo pipefail

WORK="${QH_HEGNN_WORK:-/scratch/gma/bzhang/qh_hegnn_standard_validation}"
MANIFEST="${QH_HEGNN_SPLIT_MANIFEST:-${WORK}/splits/standard_validation/run_cases_standard_validation.tsv}"
TASK_INDEX=$((SLURM_ARRAY_TASK_ID + 1))

IFS=$'\t' read -r CASE_TAG SEED SPLIT_JSON < <(awk -v n="${TASK_INDEX}" 'NR==n {print $1"\t"$2"\t"$3}' "${MANIFEST}")
if [ -z "${CASE_TAG:-}" ] || [ -z "${SPLIT_JSON:-}" ]; then
  echo "[ABORT] Could not resolve task ${SLURM_ARRAY_TASK_ID} from ${MANIFEST}"
  exit 2
fi

echo "=================================================="
echo "Standard Random/Scaffold/Butina model validation"
echo "Case: ${CASE_TAG}"
echo "Seed: ${SEED}"
echo "Split: ${SPLIT_JSON}"
echo "Start: $(date)"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "=================================================="

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate energetic_gnn

SCRIPT_DIR="${QH_HEGNN_SCRIPT_DIR:-${WORK}/src}"
cd "${SCRIPT_DIR}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export QH_HEGNN_CONTROLLED_SPLIT_JSON="${SPLIT_JSON}"

OUT_ROOT="${QH_HEGNN_RESULTS_DIR:-${WORK}/results/standard_validation}"
OUT_DIR="${OUT_ROOT}/${CASE_TAG}"
EVIDENCE_DIR="${OUT_DIR}/model_diagnostics"
mkdir -p "${OUT_DIR}" "${EVIDENCE_DIR}"

python -u run_qh_hegnn_controlled_split.py \
  --model_variant final_specialist \
  --epochs 120 \
  --batch_size 64 \
  --val_fraction 0.20 \
  --specialist_calib_fraction 0.10 \
  --target_profile sensitivity \
  --density_label_mode crystal_preferred \
  --loss_weight_mode focus_electronic \
  --early_stopping_patience 25 \
  --coord_noise_std 0.0 \
  --coord_noise_prob 0.0 \
  --teacher_n_jobs 16 \
  --teacher_cv_folds 5 \
  --teacher_top_k 5 \
  --disable_mlp_teacher \
  --force_hgs_robust_teachers \
  --hgs_robust_blend_max 0.35 \
  --hgs_robust_blend_step 0.05 \
  --hgs_robust_blend_min_improvement 0.0 \
  --xtb_feature_path ../data/curated_molecule_clean_v1/xtb_features_molecule_clean_10d_aligned_sanitized_for_03.csv \
  --enable_bde_features \
  --bde_feature_path ../data/curated_molecule_clean_v1/old_dataset_molecule_clean.csv \
  --enable_true_phys_features \
  --true_phys_feature_path "${QH_HEGNN_PHYSICS_FEATURE_TABLE:-${WORK}/data/physics_features_merged.csv}" \
  --residual_policy conservative \
  --residual_min_corr 0.05 \
  --alpha_max 0.75 \
  --alpha_step 0.025 \
  --alpha_min_improvement 0.005 \
  --export_npj_evidence \
  --npj_evidence_dir "${EVIDENCE_DIR}" \
  --run_tag "standard_validation_${CASE_TAG}" \
  --seed "${SEED}" \
  --output_dir "${OUT_DIR}" \
  --final_model_path "${OUT_DIR}/final_specialist_standard_validation_${CASE_TAG}.joblib"

METRICS="${OUT_DIR}/final_specialist_final_metrics_standard_validation_${CASE_TAG}.csv"
SPLIT_OUT="${OUT_DIR}/split_standard_validation_${CASE_TAG}.json"
PREDICTIONS="${EVIDENCE_DIR}/Supplementary_NPJ_Validation_Predictions_standard_validation_${CASE_TAG}.csv"
for f in "${METRICS}" "${SPLIT_OUT}" "${PREDICTIONS}"; do
  if [ ! -s "$f" ]; then
    echo "[ABORT] Missing expected artifact: $f"
    exit 3
  fi
done

echo "=================================================="
echo "Standard validation case finished"
echo "Case: ${CASE_TAG}"
echo "End: $(date)"
echo "Metrics: ${METRICS}"
echo "Predictions: ${PREDICTIONS}"
echo "=================================================="
