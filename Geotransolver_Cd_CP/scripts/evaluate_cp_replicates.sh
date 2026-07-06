#!/usr/bin/env bash
# ================= Slurm: CP / CQR 蒙特卡洛评估（evaluate_cp_replicates.py） =================
# 提交示例（请在仓库根目录）:
#   sbatch scripts/evaluate_cp_replicates.sh
# 覆盖默认参数:
#   CP_EVAL_N_CAL=400 CP_EVAL_R=1000 sbatch scripts/evaluate_cp_replicates.sh
# 若已有缓存可跳过推理（仅占位 GPU 时间短），但仍需能加载 torch 等依赖。
#
#SBATCH --job-name=geo-cp-repl
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/cp_replicates_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/cp_replicates_%j.err

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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

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

# 可选：数据路径与 evaluate_cp_replicates.py / Config 一致（若 checkpoint 里已有则不必设）
# export STL_ROOT_DIR="/path/to/stl/root"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/final/best_model.pth}"
EVAL_CSV="${CP_EVAL_CSV:-${PROJECT_ROOT}/data_splits/train_pool_90_with_cv_fold.csv}"
OOF_DIR="${CP_EVAL_OOF_DIR:-${PROJECT_ROOT}/results/cvplus}"
N_CAL="${CP_EVAL_N_CAL:-200}"
R="${CP_EVAL_R:-500}"
ALPHA="${CP_EVAL_ALPHA:-0.1}"
CACHE_NPZ="${CP_EVAL_CACHE:-${PROJECT_ROOT}/results/cp_eval_oof_cache.npz}"
OUT_JSON="${CP_EVAL_OUT_JSON:-${PROJECT_ROOT}/results/cp_replicates_oof_summary.json}"
BATCH_SIZE="${CP_EVAL_BATCH_SIZE:-0}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "[INFO] CP_EVAL_CSV=${EVAL_CSV}"
echo "[INFO] CP_EVAL_OOF_DIR=${OOF_DIR}"
echo "[INFO] CP_EVAL_N_CAL=${N_CAL}  CP_EVAL_R=${R}  CP_EVAL_ALPHA=${ALPHA}"
echo "[INFO] CP_EVAL_CACHE=${CACHE_NPZ}"
echo "[INFO] CP_EVAL_OUT_JSON=${OUT_JSON}"

CMD=(python "${PROJECT_ROOT}/evaluate_cp_replicates.py"
  --oof-dir "${OOF_DIR}"
  --n-cal "${N_CAL}"
  --R "${R}"
  --alpha "${ALPHA}"
  --cache "${CACHE_NPZ}"
  --out-json "${OUT_JSON}"
)
if [[ "${BATCH_SIZE}" != "0" ]]; then
  CMD+=(--batch-size "${BATCH_SIZE}")
fi

"${CMD[@]}" "$@"
