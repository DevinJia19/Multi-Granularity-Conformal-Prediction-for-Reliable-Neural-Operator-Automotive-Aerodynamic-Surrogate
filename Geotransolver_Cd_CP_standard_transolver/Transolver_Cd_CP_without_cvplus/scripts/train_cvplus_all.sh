#!/usr/bin/env bash
#SBATCH --job-name=geo-cvplus
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/cvplus_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/cvplus_%j.err
#
# ========== Submit with sbatch only (do not run bash on login node) ==========
#
# Example:
#   cd /path/to/Transolver_Cd_CP_without_cvplus && sbatch scripts/train_cvplus_all.sh
# Resume from fold 2:
#   export CVPLUS_START_FOLD=2 && sbatch scripts/train_cvplus_all.sh
#
# Train K folds (default K=5), merge OOF via merge_cvplus_oof.py -> hat_q.json
#
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
  else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
  fi
fi
export PROJECT_ROOT

K="${CVPLUS_N_SPLITS:-${SPLIT_N_SPLITS:-5}}"
export SPLIT_N_SPLITS="${SPLIT_N_SPLITS:-$K}"
START_FOLD="${CVPLUS_START_FOLD:-0}"
if ! [[ "${START_FOLD}" =~ ^[0-9]+$ ]] || ((START_FOLD < 0 || START_FOLD >= K)); then
  echo "[ERROR] CVPLUS_START_FOLD=${START_FOLD} must be in [0, $((K - 1))]" >&2
  exit 1
fi
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CV+ folds: ${START_FOLD} .. $((K - 1)) (SPLIT_N_SPLITS=${SPLIT_N_SPLITS}, CVPLUS_START_FOLD=${START_FOLD})"

for ((FOLD = START_FOLD; FOLD < K; FOLD++)); do
  echo ""
  echo "==================== CV+ fold ${FOLD}/${K} ===================="
  export SPLIT_CALIB_FOLD="${FOLD}"
  export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/fold_${FOLD}"
  export LOG_DIR="${PROJECT_ROOT}/logs/fold_${FOLD}"
  export LOSS_CURVE_FILE="${PROJECT_ROOT}/logs/fold_${FOLD}/loss_curves.png"
  export GLOBAL_DESC_STATS_CACHE="${CHECKPOINT_DIR}/global_descriptor_stats.json"
  export CVPLUS_SAVE_OOF=1
  export CVPLUS_OOF_DIR="${PROJECT_ROOT}/results/cvplus"
  export RUN_SPLIT=1
  export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  export VAL_CSV="${PROJECT_ROOT}/data_splits/validation_split.csv"
  export VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
  export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-80}"
  export NUM_EPOCHS="${NUM_EPOCHS:-600}"
  export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
  export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
  export DROPOUT="${DROPOUT:-0.05}"
  export USE_COSINE_SCHEDULER="${USE_COSINE_SCHEDULER:-1}"
  export POINT_SURFACE_FEATURES="${POINT_SURFACE_FEATURES:-1}"
  export POINT_USE_CURVATURE="${POINT_USE_CURVATURE:-0}"
  export USE_AREA_WEIGHTED_POOLING="${USE_AREA_WEIGHTED_POOLING:-0}"
  export Q50_LOSS_WEIGHT="${Q50_LOSS_WEIGHT:-0.5}"
  export POINT_CACHE_VERSION="${POINT_CACHE_VERSION:-v2_surface}"
  export MESH_CACHE_VERSION="${MESH_CACHE_VERSION:-v2_faces}"

  bash "${PROJECT_ROOT}/scripts/train.sh"
done

echo ""
echo "[INFO] Merging OOF data ..."
python "${PROJECT_ROOT}/merge_cvplus_oof.py" --oof-dir "${PROJECT_ROOT}/results/cvplus"
echo "[OK] CV+ calibration JSON: ${PROJECT_ROOT}/results/cvplus/hat_q.json"
echo "[NEXT] Train final model: sbatch scripts/train_full_90.sh"
echo "[NEXT] Test final model + CV+: TEST_MODE=cvplus_final sbatch scripts/test.sh"
