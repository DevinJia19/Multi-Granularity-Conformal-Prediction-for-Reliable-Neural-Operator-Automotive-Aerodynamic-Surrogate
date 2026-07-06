#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# Submit from repository root, e.g.:
#   sbatch scripts/train.sh
#   RUN_SPLIT=0 sbatch scripts/train.sh   # skip split, use existing data_splits
#   RUN_SPLIT=0 RUN_NORM=0 NPROC_PER_NODE=1 sbatch scripts/train.sh --config-name=config_overfit_sigma
#   DATASET_DIR=/path/to/zarr_root sbatch scripts/train.sh   # override zarr location for split_dataset.py
#   SPLIT_FORCE=1 sbatch scripts/train.sh                    # replace split symlinks if paths conflict
# Optional Hydra overrides after the script is not typical with sbatch; use:
#   torchrun ... train.py key=value
# or set env and extend this script if needed.
#
#SBATCH --job-name=geotransolver-train
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# ----- Project root: env > SLURM submit dir > infer from script location -----
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

# Slurm / no TTY: avoid block-buffered Python stdout
export PYTHONUNBUFFERED=1

# ----- Conda (same init pattern as sibling projects; env name: geotrans_py311) -----
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda not found. Initialize conda or set CONDA_EXE / source conda.sh."
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

# Linux/macOS: 未设置时把通用用户缓存指到仓库 .cache，减轻 ~/.cache 压力
if [[ -z "${XDG_CACHE_HOME:-}" ]]; then
  export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache"
fi
mkdir -p "${XDG_CACHE_HOME}"

# Optional: cluster modules (uncomment if you do not use conda-provided CUDA)
# module load Python/3.11.5-GCCcore-12.3.0
# module load CUDA/12.1.1

# ----- Dependency check -----
python - <<'PY'
import importlib.util
import os
import sys

required_modules = [
    "torch",
    "numpy",
    "hydra",
    "omegaconf",
    "physicsnemo",
    "tabulate",
]
missing = [m for m in required_modules if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing Python packages:", ", ".join(missing))
    root = os.environ.get("PROJECT_ROOT", ".")
    print("Install with: python -m pip install -r " + os.path.join(root, "requirements.txt"))
    sys.exit(1)
print("[OK] Dependency check passed.")
PY

# ----- Step 1: split dataset (optional) -----
# split_dataset.py only writes train_files.txt etc. unless --symlink/--copy/--move is given.
# CAEDataset expects .../data_splits/train/ (with run_*.zarr), so we default to --symlink.
# Zarr root: split_dataset.py default path or set DATASET_DIR=/path/to/run_zarr_parent before sbatch.
# SPLIT_FORCE=1 → pass --force (overwrite conflicting train/calib/test entries).
RUN_SPLIT="${RUN_SPLIT:-1}"
if [[ "${RUN_SPLIT}" == "1" ]]; then
    SPLIT_ARGS=(--symlink)
    if [[ -n "${DATASET_DIR:-}" ]]; then
        SPLIT_ARGS+=(--dataset-dir "${DATASET_DIR}")
    fi
    if [[ "${SPLIT_FORCE:-0}" == "1" ]]; then
        SPLIT_ARGS+=(--force)
    fi
    echo "[INFO] Step 1/3: split_dataset.py ${SPLIT_ARGS[*]} ..."
    python "${PROJECT_ROOT}/split_dataset.py" "${SPLIT_ARGS[@]}"
else
    echo "[INFO] Skip split_dataset.py (RUN_SPLIT=${RUN_SPLIT}); using existing data_splits/"
fi

# ----- Step 2: normalization stats (optional) -----
RUN_NORM="${RUN_NORM:-1}"
if [[ "${RUN_NORM}" == "1" ]]; then
    echo "[INFO] Step 2/3: compute_normalizations.py (Hydra: conf/config.yaml) ..."
    python "${PROJECT_ROOT}/compute_normalizations.py"
else
    echo "[INFO] Skip compute_normalizations.py (RUN_NORM=${RUN_NORM})"
fi

# ----- Step 3: GeoTransolver 场训练 (train.py + conf/config.yaml) -----
# 默认 3×A100；单机多卡可覆盖：NPROC_PER_NODE=1 torchrun ...
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
MASTER_PORT="${MASTER_PORT:-29500}"

echo "[INFO] Step 3/3: train.py — Hydra 配置: conf/config.yaml"
echo "[INFO] Launching torchrun with nproc_per_node=${NPROC_PER_NODE}, master_port=${MASTER_PORT}"
torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/train.py" "$@"

echo "[INFO] Training finished."
echo "[INFO] Checkpoints: <output_dir>/<run_id>/checkpoints/ (see conf yaml)."
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
