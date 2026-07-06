#!/usr/bin/env bash
#SBATCH --job-name=geo-cvplus
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/cvplus_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/cvplus_%j.err
#
# ========== 必须用 sbatch 提交；勿在登录节点 bash 本脚本 ==========
#
# 提交示例:
#   cd /path/to/Geotransolver_Cd_CP && sbatch scripts/train_cvplus_all.sh
# 从第 3 折起续跑（fold 索引 2）:
#   export CVPLUS_START_FOLD=2 && sbatch scripts/train_cvplus_all.sh
#
# 依次训练 K 折（默认 K=5）：每折 OOF → merge_cvplus_oof.py → hat_q.json
# CV+ OOF 仅来自 official train 内部 fold holdout；official validation 仅用于选 best model。
#
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

K="${CVPLUS_N_SPLITS:-${SPLIT_N_SPLITS:-5}}"
export SPLIT_N_SPLITS="${SPLIT_N_SPLITS:-$K}"
START_FOLD="${CVPLUS_START_FOLD:-0}"
if ! [[ "${START_FOLD}" =~ ^[0-9]+$ ]] || ((START_FOLD < 0 || START_FOLD >= K)); then
  echo "[ERROR] CVPLUS_START_FOLD=${START_FOLD} 无效，须在 [0, $((K - 1))] 内" >&2
  exit 1
fi
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CV+ folds: ${START_FOLD} .. $((K - 1)) (SPLIT_N_SPLITS=${SPLIT_N_SPLITS}, CVPLUS_START_FOLD=${START_FOLD})"

for ((FOLD = START_FOLD; FOLD < K; FOLD++)); do
  echo ""
  echo "==================== CV+ fold ${FOLD}/${K} ===================="
  export SPLIT_CALIB_FOLD="${FOLD}"
  export CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/fold_${FOLD}"
  export LOG_DIR="${PROJECT_ROOT}/logs/fold_${FOLD}"
  export LOSS_CURVE_FILE="${PROJECT_ROOT}/logs/fold_${FOLD}/loss_curves.png"
  export GLOBAL_DESC_STATS_CACHE="${CHECKPOINT_DIR}/global_descriptor_stats.json"
  export CVPLUS_SAVE_OOF=1
  export CVPLUS_OOF_DIR="${PROJECT_ROOT}/results/cvplus"
  export RUN_SPLIT=1
  export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  export VAL_CSV="${PROJECT_ROOT}/data_splits/validation_split.csv"
  export VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
  export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"

  bash "${PROJECT_ROOT}/scripts/train.sh"
done

echo ""
echo "[INFO] Merging OOF data ..."
python "${PROJECT_ROOT}/merge_cvplus_oof.py" --oof-dir "${PROJECT_ROOT}/results/cvplus"
echo "[OK] CV+ 校准 JSON 已写入 ${PROJECT_ROOT}/results/cvplus/hat_q.json"
echo "[NEXT] 在 official train 400 cases 上训练最终模型: sbatch scripts/train_full_90.sh"
echo "[NEXT] 在 official test 上评估: sbatch scripts/test.sh"
