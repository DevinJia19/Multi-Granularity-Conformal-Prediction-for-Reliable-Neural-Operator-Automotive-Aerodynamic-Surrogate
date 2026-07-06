#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# Submit from repository root:
#   sbatch scripts/test.sh
# Default: 3×A100 via torchrun — runs inference_on_zarr.py (Zarr evaluation).
# Single GPU: NPROC_PER_NODE=1 sbatch scripts/test.sh
# CP workflow (calib then test, physical npz).
# Datapipe only accepts phase=train|val; override data.val.data_path for calib/test:
#   sbatch scripts/test.sh inference.phase=val data.val.data_path=data_splits/calib cp_output.save_pointwise_npz=true cp_output.subdir=cp_pointwise_calib
#   sbatch scripts/test.sh inference.phase=val data.val.data_path=data_splits/test cp_output.save_pointwise_npz=true cp_output.subdir=cp_pointwise_test
#   python cp_calibrate_and_write_vtp.py --calib-dir runs/<run_id>/cp_pointwise_calib ...
# Hydra overrides (optional):
#   sbatch scripts/test.sh -- training.num_epochs=...   # not typical; use trailing "$@"
# Better: pass args after script name if your sbatch supports wrapping, or set EXTRA_ARGS.
#
#SBATCH --job-name=transolver-test
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

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

# ----- Conda -----
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda not found. Initialize conda or source conda.sh."
    exit 1
fi
conda activate geotran_pressure
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# NVIDIA Warp: avoid ~/.cache on home (quota); SLURM jobs use node temp dir by default.
# Override with a real directory, e.g. export WARP_CACHE_PATH=/cephyr/users/$USER/scratch/warp_cache
# Do NOT use doc placeholders like /path or /path/to/...
case "${WARP_CACHE_PATH:-}" in
  /path|/path/|/path/to|/path/to/your/scratch_or_project/warp_cache|/path/to/*)
    echo "[WARN] Ignoring invalid WARP_CACHE_PATH='${WARP_CACHE_PATH}' (placeholder). Using default."
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

# Optional cluster modules:
# module load Python/3.11.5-GCCcore-12.3.0
# module load CUDA/12.1.1

# ----- Dependency check -----
python - <<'PY'
import importlib.util
import os
import shutil
import sys

required_modules = [
    "torch",
    "numpy",
    "hydra",
    "omegaconf",
    "physicsnemo",
    "tabulate",
    "sklearn",
]
missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing))
    root = os.environ.get("PROJECT_ROOT", ".")
    print("Install with: python -m pip install -r " + os.path.join(root, "requirements.txt"))
    sys.exit(1)
if shutil.which("python") is None:
    print("[ERROR] python not found in PATH.")
    sys.exit(1)
print("[OK] Dependency check passed.")
PY

# ----- Evaluation entry (this repo has no test.py; use Zarr validation inference) -----
# INFERENCE_SCRIPT: default inference_on_zarr.py; set to inference_on_vtk.py if you pass VTK Hydra overrides.
INFERENCE_SCRIPT="${INFERENCE_SCRIPT:-inference_on_zarr.py}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"

EXTRA_HYDRA_ARGS=(
    "data.resolution=8192"
)

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] INFERENCE_SCRIPT=${INFERENCE_SCRIPT}"
echo "[INFO] GPUs: ${NPROC_PER_NODE}"
echo "[INFO] data.resolution=8192 (chunk size for full-surface inference)"
echo "[INFO] Config comes from conf/*.yaml (default config_name inside each Python entry)."
echo "[INFO] Launching torchrun with nproc_per_node=${NPROC_PER_NODE}, master_port=${MASTER_PORT}"
torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/${INFERENCE_SCRIPT}" \
    "${EXTRA_HYDRA_ARGS[@]}" \
    "$@"

echo "[INFO] Evaluation finished."

# ----- Optional: VTK inference (uncomment and set vtk_inference.* via Hydra or env) -----
# RUN_VTK=1 sbatch scripts/test.sh
# if [[ "${RUN_VTK:-0}" == "1" ]]; then
#     python "${PROJECT_ROOT}/inference_on_vtk.py" \
#         +vtk_inference.input_dir=/path/to/in \
#         +vtk_inference.output_dir=/path/to/out
# fi
