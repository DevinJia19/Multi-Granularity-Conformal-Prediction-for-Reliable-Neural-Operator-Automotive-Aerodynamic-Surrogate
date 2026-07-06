#!/usr/bin/env bash
# Test checkpoint from overfit smoke training (scripts/overfit_single.sh).
#
# Alvis:
#   sbatch scripts/test_overfit.sh
#
# Or:
#   TEST_MODE=overfit sbatch scripts/test.sh

#SBATCH --job-name=overfit-test
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_overfit_%j.out
#SBATCH --error=logs/test_overfit_%j.err

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

export TEST_MODE=overfit
export CQR_HAT_Q_JSON=""

if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  for _candidate in \
    "${PROJECT_ROOT}/checkpoints/overfit_8/best_model.pth" \
    "${PROJECT_ROOT}/checkpoints/overfit_single/best_model.pth"
  do
    if [[ -f "${_candidate}" ]]; then
      export CHECKPOINT_PATH="${_candidate}"
      break
    fi
  done
fi

exec bash "${PROJECT_ROOT}/scripts/test.sh" "$@"
