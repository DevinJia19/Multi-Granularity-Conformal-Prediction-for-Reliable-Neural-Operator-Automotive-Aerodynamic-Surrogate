#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# If your cluster uses Slurm, submit with:
#   sbatch scripts/train.sh
#SBATCH --job-name=transolver-train
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=18:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/train_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Transolver_Cd_CP_without_cvplus/logs/train_%j.err

set -euo pipefail

# Project root: SLURM_SUBMIT_DIR when sbatch from repo root, else infer from script path
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
# Backbone: PhysicsNeMo Transolver only
export BACKBONE_TYPE="${BACKBONE_TYPE:-transolver}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Activate conda env and prefer its runtime libraries (fixes GLIBCXX mismatch).
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [[ -f "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh" ]]; then
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
else
    echo "[ERROR] conda not found. Please initialize conda in this shell."
    exit 1
fi
conda activate geotrans_py311
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python - <<PY
import importlib.util
import os
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

print("[OK] Dependency check passed.")
PY

RUN_SPLIT="${RUN_SPLIT:-1}"
FULL_CSV_DEFAULT="/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv"
if [[ "${RUN_SPLIT}" == "1" ]]; then
    echo "[INFO] Step 1/2: split dataset ..."
    python "${PROJECT_ROOT}/split_dataset.py"
    export TRAIN_CSV="${TRAIN_CSV:-./data_splits/train_split.csv}"
else
    echo "[INFO] Skip split_dataset.py (RUN_SPLIT=${RUN_SPLIT}). Use existing data_splits/*.csv"
    if [[ -z "${TRAIN_CSV:-}" ]]; then
        if [[ -f "./data_splits/train_split.csv" ]]; then
            export TRAIN_CSV="./data_splits/train_split.csv"
        else
            export TRAIN_CSV="${FULL_CSV_DEFAULT}"
        fi
    fi
fi
echo "[INFO] TRAIN_CSV=${TRAIN_CSV}"
echo "[INFO] BACKBONE_TYPE=${BACKBONE_TYPE}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
DEFAULT_NUM_WORKERS=$(( CPUS_PER_TASK / NPROC_PER_NODE ))
if [[ "${DEFAULT_NUM_WORKERS}" -lt 1 ]]; then
    DEFAULT_NUM_WORKERS=1
fi
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
NUM_POINTS="${NUM_POINTS:-8192}"
VAL_CSV="${VAL_CSV:-./data_splits/validation_split.csv}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-80}"
export NUM_POINTS
export VAL_CSV
export VALIDATE_EVERY
export EARLY_STOPPING_PATIENCE
OVERFIT_MODE="${OVERFIT_MODE:-0}"
USE_AMP="${USE_AMP:-0}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
ENABLE_POINT_CACHE="${ENABLE_POINT_CACHE:-0}"
POINT_CACHE_DIR="${POINT_CACHE_DIR:-./cache/pointclouds}"
POINT_CACHE_VERSION="${POINT_CACHE_VERSION:-v2_surface}"
ENABLE_MESH_CACHE="${ENABLE_MESH_CACHE:-1}"
MESH_CACHE_DIR="${MESH_CACHE_DIR:-./cache/meshes}"
MESH_CACHE_VERSION="${MESH_CACHE_VERSION:-v2_faces}"
POINT_SURFACE_FEATURES="${POINT_SURFACE_FEATURES:-1}"
POINT_USE_CURVATURE="${POINT_USE_CURVATURE:-0}"
USE_AREA_WEIGHTED_POOLING="${USE_AREA_WEIGHTED_POOLING:-0}"
export POINT_SURFACE_FEATURES
export POINT_USE_CURVATURE
export USE_AREA_WEIGHTED_POOLING
export NUM_WORKERS
export USE_AMP
export PERSISTENT_WORKERS
export PREFETCH_FACTOR
export ENABLE_POINT_CACHE
export POINT_CACHE_DIR
export POINT_CACHE_VERSION
export ENABLE_MESH_CACHE
export MESH_CACHE_DIR
export MESH_CACHE_VERSION
OVERFIT_SUBSET_SIZE="${OVERFIT_SUBSET_SIZE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-600}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
APPLY_AUGMENTATIONS="${APPLY_AUGMENTATIONS:-1}"
DETERMINISTIC_SAMPLING="${DETERMINISTIC_SAMPLING:-0}"
DROPOUT="${DROPOUT:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
USE_COSINE_SCHEDULER="${USE_COSINE_SCHEDULER:-1}"
STD_REG_WEIGHT="${STD_REG_WEIGHT:-0.0}"
INTERVAL_WIDTH_WEIGHT="${INTERVAL_WIDTH_WEIGHT:-0.0}"
Q50_LOSS_WEIGHT="${Q50_LOSS_WEIGHT:-0.5}"
export OVERFIT_MODE
export OVERFIT_SUBSET_SIZE
export BATCH_SIZE
export NUM_EPOCHS
export LEARNING_RATE
export APPLY_AUGMENTATIONS
export DETERMINISTIC_SAMPLING
export DROPOUT
export WEIGHT_DECAY
export USE_COSINE_SCHEDULER
export STD_REG_WEIGHT
export INTERVAL_WIDTH_WEIGHT
export Q50_LOSS_WEIGHT

export SPLIT_CALIB_FOLD="${SPLIT_CALIB_FOLD:-0}"
export CVPLUS_SAVE_OOF="${CVPLUS_SAVE_OOF:-0}"
export CVPLUS_OOF_DIR="${CVPLUS_OOF_DIR:-./results/cvplus}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
export LOG_DIR="${LOG_DIR:-./logs}"
export LOSS_CURVE_FILE="${LOSS_CURVE_FILE:-}"
export GLOBAL_DESC_STATS_CACHE="${GLOBAL_DESC_STATS_CACHE:-}"

MASTER_PORT="${MASTER_PORT:-29500}"
TOTAL_DATALOADER_WORKERS=$(( NPROC_PER_NODE * NUM_WORKERS ))
echo "[INFO] VAL_CSV=${VAL_CSV}"
echo "[INFO] VALIDATE_EVERY=${VALIDATE_EVERY}"
echo "[INFO] EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}"
echo "[INFO] NUM_POINTS=${NUM_POINTS}"
echo "[INFO] Launch training on ${NPROC_PER_NODE} GPUs, NUM_WORKERS(per proc)=${NUM_WORKERS}, TOTAL_DATALOADER_WORKERS=${TOTAL_DATALOADER_WORKERS}, OVERFIT_MODE=${OVERFIT_MODE}, BATCH_SIZE=${BATCH_SIZE}, OVERFIT_SUBSET_SIZE=${OVERFIT_SUBSET_SIZE}, NUM_EPOCHS=${NUM_EPOCHS}, LR=${LEARNING_RATE}, APPLY_AUGMENTATIONS=${APPLY_AUGMENTATIONS}, DETERMINISTIC_SAMPLING=${DETERMINISTIC_SAMPLING}, DROPOUT=${DROPOUT}, WEIGHT_DECAY=${WEIGHT_DECAY}, USE_COSINE_SCHEDULER=${USE_COSINE_SCHEDULER}, STD_REG_WEIGHT=${STD_REG_WEIGHT}, Q50_LOSS_WEIGHT=${Q50_LOSS_WEIGHT}, INTERVAL_WIDTH_WEIGHT=${INTERVAL_WIDTH_WEIGHT}, USE_AMP=${USE_AMP}, PERSISTENT_WORKERS=${PERSISTENT_WORKERS}, PREFETCH_FACTOR=${PREFETCH_FACTOR}, ENABLE_POINT_CACHE=${ENABLE_POINT_CACHE}, POINT_CACHE_DIR=${POINT_CACHE_DIR}, POINT_CACHE_VERSION=${POINT_CACHE_VERSION}, ENABLE_MESH_CACHE=${ENABLE_MESH_CACHE}, MESH_CACHE_DIR=${MESH_CACHE_DIR}, MESH_CACHE_VERSION=${MESH_CACHE_VERSION}, SPLIT_CALIB_FOLD=${SPLIT_CALIB_FOLD}, CVPLUS_SAVE_OOF=${CVPLUS_SAVE_OOF}, CHECKPOINT_DIR=${CHECKPOINT_DIR}, LOG_DIR=${LOG_DIR} ..."
torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/train.py" "$@"
