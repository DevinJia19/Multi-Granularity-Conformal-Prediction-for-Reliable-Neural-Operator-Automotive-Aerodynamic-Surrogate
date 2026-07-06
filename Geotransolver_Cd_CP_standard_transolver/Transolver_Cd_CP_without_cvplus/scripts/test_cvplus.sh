#!/usr/bin/env bash
# CV+ final model test (after train_cvplus_all.sh + train_full_90.sh).
#
# Pipeline:
#   sbatch scripts/train_cvplus_all.sh   # fold_0..4 + results/cvplus/hat_q.json
#   sbatch scripts/train_full_90.sh      # checkpoints/final/best_model.pth
#   sbatch scripts/test_cvplus.sh        # this script
#
# Or:
#   TEST_MODE=cvplus_final sbatch scripts/test.sh

#SBATCH --job-name=cvplus-test
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/test_cvplus_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/test_cvplus_%j.err

set -euo pipefail

# sbatch copies only this script to /tmp/slurmd/job*/ — use SLURM_SUBMIT_DIR, not dirname $0
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

export TEST_MODE=cvplus_final
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/final/best_model.pth}"
export CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-${PROJECT_ROOT}/results/cvplus/hat_q.json}"

exec bash "${PROJECT_ROOT}/scripts/test.sh" "$@"
