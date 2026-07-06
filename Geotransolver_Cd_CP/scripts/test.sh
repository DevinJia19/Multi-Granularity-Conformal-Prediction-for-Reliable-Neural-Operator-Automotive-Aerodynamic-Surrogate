#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# If your cluster uses Slurm, submit with:
#   sbatch scripts/test.sh
# Default to single-GPU, single-process evaluation.
#SBATCH --job-name=geotransolver-test
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/test_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/test_%j.err

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
# conda 的 activate.d 会展开 $LD_LIBRARY_PATH；在 set -u 下未定义会报 unbound variable
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
FULL_CSV_DEFAULT="/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv"
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
# CV+ 合并后的 hat_q（可选；与 CHECKPOINT_PATH=checkpoints/final 联用）
export CQR_HAT_Q_JSON="${CQR_HAT_Q_JSON:-${PROJECT_ROOT}/results/cvplus/hat_q.json}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/final/best_model.pth}"
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
