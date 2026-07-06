#!/usr/bin/env bash

# ================= SLURM resources (optional) =================
# If your cluster uses Slurm, submit with:
#   sbatch scripts/train.sh
#SBATCH --job-name=geotransolver-train
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=18:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/train_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/train_%j.err

set -euo pipefail

# 项目根：优先环境变量；sbatch 时用 SLURM_SUBMIT_DIR（应在仓库根目录执行 sbatch）；否则用脚本路径推断
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
# Slurm/后台任务无 TTY 时 Python 默认块缓冲，stdout 长时间不落盘看起来像「空日志」
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

# Dependency check before launching training.
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

# Optional Step 1: split dataset before training.
# Full-sample overfit diagnosis default skips split.
RUN_SPLIT="${RUN_SPLIT:-1}"
FULL_CSV_DEFAULT="/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv"
if [[ "${RUN_SPLIT}" == "1" ]]; then
    echo "[INFO] Step 1/2: split dataset ..."
    python "${PROJECT_ROOT}/split_dataset.py"
    # Full pipeline: train on split file
    export TRAIN_CSV="${TRAIN_CSV:-./data_splits/train_split.csv}"
else
    echo "[INFO] Skip split_dataset.py (RUN_SPLIT=${RUN_SPLIT}). Use existing data_splits/*.csv"
    # Overfit/debug: if split file is absent and TRAIN_CSV not set, fallback to full drivaer_ml targets.
    if [[ -z "${TRAIN_CSV:-}" ]]; then
        if [[ -f "./data_splits/train_split.csv" ]]; then
            export TRAIN_CSV="./data_splits/train_split.csv"
        else
            export TRAIN_CSV="${FULL_CSV_DEFAULT}"
        fi
    fi
fi
echo "[INFO] TRAIN_CSV=${TRAIN_CSV}"

# Formal training default: 4 GPUs (DDP). NUM_WORKERS is per-process (per GPU).
# Override when submitting, e.g.:
#   NPROC_PER_NODE=4 NUM_WORKERS=8 OVERFIT_MODE=0 OVERFIT_SUBSET_SIZE=0 BATCH_SIZE=8 sbatch scripts/train.sh
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
# NUM_WORKERS is per DDP process. By default, split allocated CPUs across GPUs.
DEFAULT_NUM_WORKERS=$(( CPUS_PER_TASK / NPROC_PER_NODE ))
if [[ "${DEFAULT_NUM_WORKERS}" -lt 1 ]]; then
    DEFAULT_NUM_WORKERS=1
fi
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
NUM_POINTS="${NUM_POINTS:-8192}"
VAL_CSV="${VAL_CSV:-./data_splits/validation_split.csv}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
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
POINT_CACHE_VERSION="${POINT_CACHE_VERSION:-v1}"
ENABLE_MESH_CACHE="${ENABLE_MESH_CACHE:-1}"
MESH_CACHE_DIR="${MESH_CACHE_DIR:-./cache/meshes}"
MESH_CACHE_VERSION="${MESH_CACHE_VERSION:-v1}"
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
NUM_EPOCHS="${NUM_EPOCHS:-720}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
APPLY_AUGMENTATIONS="${APPLY_AUGMENTATIONS:-1}"
DETERMINISTIC_SAMPLING="${DETERMINISTIC_SAMPLING:-0}"
DROPOUT="${DROPOUT:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
USE_COSINE_SCHEDULER="${USE_COSINE_SCHEDULER:-0}"
STD_REG_WEIGHT="${STD_REG_WEIGHT:-0.0}"
INTERVAL_WIDTH_WEIGHT="${INTERVAL_WIDTH_WEIGHT:-0.0}"
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

# 数据划分折索引（CV+ 五折训练时由 train_cvplus_all.sh 传入 0–4）
export SPLIT_CALIB_FOLD="${SPLIT_CALIB_FOLD:-0}"
# CV+：训练结束后在当折 calibration_split 上保存 OOF conformity scores
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
echo "[INFO] Launch training on ${NPROC_PER_NODE} GPUs, NUM_WORKERS(per proc)=${NUM_WORKERS}, TOTAL_DATALOADER_WORKERS=${TOTAL_DATALOADER_WORKERS}, OVERFIT_MODE=${OVERFIT_MODE}, BATCH_SIZE=${BATCH_SIZE}, OVERFIT_SUBSET_SIZE=${OVERFIT_SUBSET_SIZE}, NUM_EPOCHS=${NUM_EPOCHS}, LR=${LEARNING_RATE}, APPLY_AUGMENTATIONS=${APPLY_AUGMENTATIONS}, DETERMINISTIC_SAMPLING=${DETERMINISTIC_SAMPLING}, DROPOUT=${DROPOUT}, WEIGHT_DECAY=${WEIGHT_DECAY}, USE_COSINE_SCHEDULER=${USE_COSINE_SCHEDULER}, STD_REG_WEIGHT=${STD_REG_WEIGHT}, INTERVAL_WIDTH_WEIGHT=${INTERVAL_WIDTH_WEIGHT}, USE_AMP=${USE_AMP}, PERSISTENT_WORKERS=${PERSISTENT_WORKERS}, PREFETCH_FACTOR=${PREFETCH_FACTOR}, ENABLE_POINT_CACHE=${ENABLE_POINT_CACHE}, POINT_CACHE_DIR=${POINT_CACHE_DIR}, POINT_CACHE_VERSION=${POINT_CACHE_VERSION}, ENABLE_MESH_CACHE=${ENABLE_MESH_CACHE}, MESH_CACHE_DIR=${MESH_CACHE_DIR}, MESH_CACHE_VERSION=${MESH_CACHE_VERSION}, SPLIT_CALIB_FOLD=${SPLIT_CALIB_FOLD}, CVPLUS_SAVE_OOF=${CVPLUS_SAVE_OOF}, CHECKPOINT_DIR=${CHECKPOINT_DIR}, LOG_DIR=${LOG_DIR} ..."
torchrun --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/train.py" "$@"
