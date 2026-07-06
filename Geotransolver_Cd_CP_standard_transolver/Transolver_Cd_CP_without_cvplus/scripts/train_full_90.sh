#!/usr/bin/env bash
#SBATCH --job-name=geo-full90
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/full90_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/full90_%j.err
#
# Train final model on official train pool (400 cases); select best on official validation.
# Submit with sbatch, not interactive bash on login node.
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

export RUN_SPLIT="${RUN_SPLIT:-0}"
export TRAIN_CSV="${TRAIN_CSV:-${PROJECT_ROOT}/data_splits/train_pool_90_with_cv_fold.csv}"
export VAL_CSV="${VAL_CSV:-${PROJECT_ROOT}/data_splits/validation_split.csv}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/final}"
export LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/final}"
export LOSS_CURVE_FILE="${LOG_DIR}/loss_curves.png"
export GLOBAL_DESC_STATS_CACHE="${CHECKPOINT_DIR}/global_descriptor_stats.json"
export CVPLUS_SAVE_OOF=0
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

if [[ ! -f "${TRAIN_CSV}" ]]; then
  echo "[ERROR] Missing ${TRAIN_CSV}. Run: python split_dataset.py"
  exit 1
fi

if [[ ! -f "${VAL_CSV}" ]]; then
  echo "[ERROR] Missing ${VAL_CSV}. Run: python split_dataset.py"
  exit 1
fi

echo "[INFO] TRAIN_CSV=${TRAIN_CSV}"
echo "[INFO] VAL_CSV=${VAL_CSV}"
echo "[INFO] CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "[NEXT] Test with CV+ OOF: TEST_MODE=cvplus_final sbatch scripts/test.sh"
bash "${PROJECT_ROOT}/scripts/train.sh"
