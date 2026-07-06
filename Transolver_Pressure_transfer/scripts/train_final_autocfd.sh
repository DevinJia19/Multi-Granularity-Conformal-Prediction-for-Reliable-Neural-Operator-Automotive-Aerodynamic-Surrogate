#!/usr/bin/env bash
#SBATCH --job-name=pressure-final
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/pressure_final_%j.out
#SBATCH --error=logs/pressure_final_%j.err

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
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda not found."
    exit 1
fi
conda activate geotran_pressure
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

case "${WARP_CACHE_PATH:-}" in
  /path|/path/|/path/to|/path/to/your/scratch_or_project/warp_cache|/path/to/*)
    unset WARP_CACHE_PATH
    ;;
esac
if [[ -z "${WARP_CACHE_PATH:-}" ]]; then
  if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export WARP_CACHE_PATH="${SLURM_TMPDIR}/warp_cache"
  else
    export WARP_CACHE_PATH="${PROJECT_ROOT}/.warp_cache"
  fi
fi
mkdir -p "${WARP_CACHE_PATH}"

if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
  export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
fi
mkdir -p "${XDG_CACHE_HOME}"

python "${PROJECT_ROOT}/split_dataset.py" \
  --final-train \
  --symlink \
  --force

RUN_SPLIT=0 \
RUN_NORM=1 \
NPROC_PER_NODE=4 \
bash "${PROJECT_ROOT}/scripts/train.sh" \
  "run_id=final_autocfd" \
  "data.resolution=8192" \
  "training.skip_validation=false" \
  "training.validate_every=1" \
  "training.early_stopping_patience=${EARLY_STOPPING_PATIENCE:-0}"

echo "[INFO] Final AutoCFD training finished."
