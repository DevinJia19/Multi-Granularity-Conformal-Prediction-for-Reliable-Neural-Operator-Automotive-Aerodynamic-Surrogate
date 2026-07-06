#!/usr/bin/env bash
#SBATCH --job-name=geo-normal-cp
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/normal_cp_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/normal_cp_%j.err
#
# ?? split conformal ?????? CV+?
# ??: cd /cephyr/.../Transolver_Cd_CP_without_cvplus && sbatch scripts/train_normal_cp.sh
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
cd "${PROJECT_ROOT}"

mkdir -p logs

# ???????????
export RUN_SPLIT=1
export SPLIT_CALIB_FOLD="${SPLIT_CALIB_FOLD:-0}"

# ?? CP???? train_split??? CV+
export TRAIN_CSV="${PROJECT_ROOT}/data_splits/train_split.csv"
export VAL_CSV="${PROJECT_ROOT}/data_splits/validation_split.csv"
export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/normal_cp"
export LOG_DIR="${PROJECT_ROOT}/logs/normal_cp"
export LOSS_CURVE_FILE="${LOG_DIR}/loss_curves.png"
export GLOBAL_DESC_STATS_CACHE="${CHECKPOINT_DIR}/global_descriptor_stats.json"

# ???? CV+
export CVPLUS_SAVE_OOF=0
export CQR_HAT_Q_JSON=""

export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

bash "${PROJECT_ROOT}/scripts/train.sh"
