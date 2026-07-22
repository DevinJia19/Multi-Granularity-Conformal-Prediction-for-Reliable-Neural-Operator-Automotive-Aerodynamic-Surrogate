#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# If your cluster uses Slurm, submit with:
#   sbatch scripts/test.sh
# Default to single-GPU, single-process evaluation.
#SBATCH --job-name=transolver-test
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/test_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/test_%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    _submit="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    if [[ "$(basename "${_submit}")" == "scripts" ]]; then
      PROJECT_ROOT="$(cd "${_submit}/.." && pwd)"
    else
      PROJECT_ROOT="${_submit}"
    fi
  else
    _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
  fi
fi
export PROJECT_ROOT
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"
export PYTHONUNBUFFERED=1
# Backbone: PhysicsNeMo Transolver only
export BACKBONE_TYPE="${BACKBONE_TYPE:-transolver}"
# conda �?activate.d 会展开 $LD_LIBRARY_PATH；在 set -u 下未定义会报 unbound variable
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Activate conda env and prefer its runtime libraries (fixes GLIBCXX mismatch).
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh" ]]; then
    # Fallback for batch shells where conda is not initialized.
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda not found. Please initialize conda in this shell."
    exit 1
fi
conda activate geotrans_py311
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Dependency check before launching evaluation.
python - <<PY
import importlib.util
import os
import shutil
import sys

required_modules = [
    "torch",
    "numpy",
    "pandas",
    "sklearn",
    "trimesh",
    "pyvista",
    "seaborn",
    "jaxtyping",
    "physicsnemo",
]

missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing))
    _root = os.environ.get("PROJECT_ROOT", ".")
    print("Install with: python -m pip install -r " + os.path.join(_root, "requirements.txt"))
    sys.exit(1)

if shutil.which("python") is None:
    print("[ERROR] python not found in PATH.")
    sys.exit(1)

print("[OK] Dependency check passed.")
PY

# Default to single-process test. Override with:
#   NPROC_PER_NODE=2 sbatch scripts/test.sh
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Choose a valid test CSV automatically unless user provided TEST_CSV.
FULL_CSV_DEFAULT="/path/to/dataset/targets.csv"
if [[ -z "${TEST_CSV:-}" ]]; then
    if [[ -f "./data_splits/test_split.csv" ]]; then
        TEST_CSV="./data_splits/test_split.csv"
    elif [[ -f "./data_splits/train_split.csv" ]]; then
        echo "[WARN] ./data_splits/test_split.csv not found, fallback to train split for smoke test."
        TEST_CSV="./data_splits/train_split.csv"
    else
        echo "[WARN] data_splits/*.csv not found, fallback to full targets CSV."
        TEST_CSV="${FULL_CSV_DEFAULT}"
    fi
fi
export TEST_CSV
echo "[INFO] TEST_CSV=${TEST_CSV}"
echo "[INFO] BACKBONE_TYPE=${BACKBONE_TYPE}"
# TEST_MODE: normal_cp | cvplus_final | plain | overfit
#   overfit -> checkpoints/overfit_8/best_model.pth (from scripts/overfit_single.sh)
TEST_MODE="${TEST_MODE:-normal_cp}"
case "${TEST_MODE}" in
  normal_cp)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/normal_cp/best_model.pth}"
    CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-}"
    ;;
  cvplus_final)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/final/best_model.pth}"
    CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-${PROJECT_ROOT}/results/cvplus/hat_q.json}"
    if [[ ! -f "${CQR_HAT_Q_JSON}" ]]; then
      echo "[ERROR] TEST_MODE=cvplus_final requires ${CQR_HAT_Q_JSON}. Run train_cvplus_all.sh first." >&2
      exit 1
    fi
    ;;
  plain)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/best_model.pth}"
    CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-}"
    ;;
  overfit)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/overfit_8/best_model.pth}"
    CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-}"
    ;;
  *)
    echo "[ERROR] Unknown TEST_MODE=${TEST_MODE}. Use: normal_cp | cvplus_final | plain | overfit" >&2
    exit 1
    ;;
esac

# Auto-resolve overfit checkpoint if default path was renamed (overfit_single / overfit_8)
if [[ ! -f "${CHECKPOINT_PATH}" && "${TEST_MODE}" == "overfit" && -z "${CHECKPOINT_PATH_SET:-}" ]]; then
  for _candidate in \
    "${PROJECT_ROOT}/checkpoints/overfit_8/best_model.pth" \
    "${PROJECT_ROOT}/checkpoints/overfit_single/best_model.pth" \
    "${PROJECT_ROOT}/checkpoints/overfit/best_model.pth"
  do
    if [[ -f "${_candidate}" ]]; then
      echo "[WARN] Using discovered checkpoint: ${_candidate}"
      CHECKPOINT_PATH="${_candidate}"
      break
    fi
  done
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  echo "[HINT] Available checkpoints:" >&2
  find "${PROJECT_ROOT}/checkpoints" -name "best_model.pth" 2>/dev/null | sed 's/^/  /' || true
  if [[ "${TEST_MODE}" == "normal_cp" ]]; then
    if [[ -f "${PROJECT_ROOT}/checkpoints/final/best_model.pth" ]] || \
       ls "${PROJECT_ROOT}"/checkpoints/fold_*/best_model.pth >/dev/null 2>&1; then
      echo "[HINT] Detected CV+ artifacts. CV+ test requires:" >&2
      echo "  1) train_cvplus_all.sh  -> checkpoints/fold_*/ + results/cvplus/hat_q.json" >&2
      echo "  2) train_full_90.sh     -> checkpoints/final/best_model.pth" >&2
      echo "  3) TEST_MODE=cvplus_final sbatch scripts/test.sh" >&2
      echo "  Or: sbatch scripts/test_cvplus.sh" >&2
    fi
  fi
  echo "[HINT] Overfit: TEST_MODE=overfit sbatch scripts/test.sh" >&2
  exit 1
fi

if [[ "${CHECKPOINT_PATH}" == *"/final/"* ]] || [[ "${CHECKPOINT_PATH}" == *"checkpoints/final"* ]]; then
  if [[ -z "${CQR_HAT_Q_JSON}" ]]; then
    echo "[ERROR] Final model checkpoint detected but CQR_HAT_Q_JSON is empty." >&2
    echo "[ERROR] Use TEST_MODE=cvplus_final or set CQR_HAT_Q_JSON=./results/cvplus/hat_q.json" >&2
    exit 1
  fi
fi

export TEST_MODE
export CHECKPOINT_PATH
export CQR_HAT_Q_JSON
echo "[INFO] TEST_MODE=${TEST_MODE}"
echo "[INFO] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[INFO] CQR_HAT_Q_JSON=${CQR_HAT_Q_JSON}"

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    torchrun --standalone \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${PROJECT_ROOT}/test.py" "$@"
else
    python "${PROJECT_ROOT}/test.py" "$@"
fi
