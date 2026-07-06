#!/usr/bin/env bash
# 8 样本过拟合冒烟测试：验证 Transolver 接口与训练链路是否正常。
#
# Alvis 提交（在仓库根目录执行）:
#   sbatch scripts/overfit_single.sh
#
# 交互式单卡调试:
#   srun -A naiss2025-22-1747 -p alvis --gres=gpu:A100:1 --cpus-per-task=8 --pty bash
#   bash scripts/overfit_single.sh
#
# 常用覆盖:
#   TRAIN_CSV=./data_splits/train_split.csv \
#   STL_ROOT_DIR=/path/to/stl \
#   OVERFIT_SUBSET_SIZE=8 \
#   NUM_EPOCHS=200 \
#   sbatch scripts/overfit_single.sh

#SBATCH --job-name=overfit-8
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/overfit_single_%j.out
#SBATCH --error=logs/overfit_single_%j.err

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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/checkpoints/overfit_8"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Activate conda (same as scripts/train.sh)
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

python - <<'PY'
import importlib.util
import sys

required = ["torch", "numpy", "pandas", "trimesh", "physicsnemo"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("[ERROR] Missing:", ", ".join(missing))
    sys.exit(1)
print("[OK] Dependency check passed.")
PY

export BACKBONE_TYPE="${BACKBONE_TYPE:-transolver}"
export OVERFIT_MODE=1
export OVERFIT_SUBSET_SIZE="${OVERFIT_SUBSET_SIZE:-8}"
export RUN_SPLIT="${RUN_SPLIT:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export NUM_EPOCHS="${NUM_EPOCHS:-500}"
export NUM_POINTS="${NUM_POINTS:-2048}"
export USE_AMP="${USE_AMP:-0}"
export ENABLE_POINT_CACHE="${ENABLE_POINT_CACHE:-0}"
export ENABLE_MESH_CACHE="${ENABLE_MESH_CACHE:-1}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints/overfit_8}"
export LOG_DIR="${LOG_DIR:-./logs/overfit_8}"
export VAL_CSV="${VAL_CSV:-}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"

# 数据路径（按你的 Alvis 目录修改；也可用 sbatch 时 export 覆盖）
export STL_ROOT_DIR="${STL_ROOT_DIR:-/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/stl}"

if [[ -z "${TRAIN_CSV:-}" ]]; then
  if [[ -f "./data_splits/train_split.csv" ]]; then
    export TRAIN_CSV="./data_splits/train_split.csv"
  else
    export TRAIN_CSV="/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Chao/PVT3/cfd-aero-pytorch/inputs/drivaer_ml/targets.csv"
  fi
fi

echo "[INFO] OVERFIT_MODE=1 过拟合测试 (${OVERFIT_SUBSET_SIZE} 样本)"
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] TRAIN_CSV=${TRAIN_CSV}"
echo "[INFO] STL_ROOT_DIR=${STL_ROOT_DIR}"
echo "[INFO] BACKBONE_TYPE=${BACKBONE_TYPE}"
echo "[INFO] OVERFIT_SUBSET_SIZE=${OVERFIT_SUBSET_SIZE}"
echo "[INFO] NUM_EPOCHS=${NUM_EPOCHS}, NUM_POINTS=${NUM_POINTS}, BATCH_SIZE=${BATCH_SIZE}, LEARNING_RATE=${LEARNING_RATE}"

python "${PROJECT_ROOT}/train.py" "$@"
