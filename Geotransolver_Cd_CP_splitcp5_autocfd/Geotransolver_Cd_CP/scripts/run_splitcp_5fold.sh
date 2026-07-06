#!/usr/bin/env bash
#SBATCH --job-name=geo-splitcp5
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=120:00:00
#SBATCH --output=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/splitcp5_%j.out
#SBATCH --error=/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-6-430/ChundongJia/Geotransolver_Cd_CP_morehidden/Geotransolver_Cd_CP/logs/splitcp5_%j.err
#
# Ordinary split CP 5-fold baseline under the AutoCFD official split.
#
# IMPORTANT: this script is NOT CV+.
#   - AutoCFD official train pool (400 cases) is split into 5 folds.
#   - For each fold: 4 folds are used to train the model, 1 fold is used only
#     to estimate q_l/q_u on that fold's calibration set.
#   - No OOF predictions are merged. No final 400-case refit is used.
#   - The official AutoCFD test set is evaluated 5 times, once per fold model,
#     and metrics are averaged at the end.
#   - Optional Monte-Carlo evaluation is also run 5 times and averaged.
#
# Usage:
#   cd /path/to/Geotransolver_Cd_CP
#   sbatch scripts/run_splitcp_5fold.sh
#
# Resume / partial usage:
#   RUN_TRAIN=0 RUN_TEST=1 RUN_MC=0 sbatch scripts/run_splitcp_5fold.sh
#   SPLITCP_START_FOLD=2 sbatch scripts/run_splitcp_5fold.sh
#
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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/splitcp_5fold"

K="${SPLITCP_N_SPLITS:-${SPLIT_N_SPLITS:-5}}"
START_FOLD="${SPLITCP_START_FOLD:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_TEST="${RUN_TEST:-1}"
RUN_MC="${RUN_MC:-1}"

if ! [[ "${K}" =~ ^[0-9]+$ ]] || (( K < 2 )); then
  echo "[ERROR] SPLITCP_N_SPLITS=${K} invalid; expected integer >=2" >&2
  exit 1
fi
if ! [[ "${START_FOLD}" =~ ^[0-9]+$ ]] || (( START_FOLD < 0 || START_FOLD >= K )); then
  echo "[ERROR] SPLITCP_START_FOLD=${START_FOLD} invalid; expected [0, $((K - 1))]" >&2
  exit 1
fi

export SPLIT_N_SPLITS="${K}"
export CVPLUS_SAVE_OOF=0
# Use single-fold calibration in test.py. "none" prevents scripts/test.sh from
# falling back to a CV+ merged hat_q.json if such a file exists from another run.
export CQR_HAT_Q_JSON="none"

# Default Monte-Carlo settings for this ordinary split CP baseline.
export CP_EVAL_N_CAL="${CP_EVAL_N_CAL:-40}"
export CP_EVAL_R="${CP_EVAL_R:-500}"
export CP_EVAL_ALPHA="${CP_EVAL_ALPHA:-0.1}"

# Training defaults can still be overridden by exporting env vars before sbatch.
TRAIN_NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"

printf '\n[INFO] PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
printf '[INFO] Ordinary split CP folds: %s .. %s, K=%s\n' "${START_FOLD}" "$((K - 1))" "${K}"
printf '[INFO] RUN_TRAIN=%s RUN_TEST=%s RUN_MC=%s\n' "${RUN_TRAIN}" "${RUN_TEST}" "${RUN_MC}"
printf '[INFO] This is NOT CV+: no OOF merge and no final full-400 refit.\n\n'

for ((FOLD = START_FOLD; FOLD < K; FOLD++)); do
  echo ""
  echo "==================== Split CP fold ${FOLD}/${K} ===================="

  FOLD_SPLIT_DIR="${PROJECT_ROOT}/data_splits/splitcp_fold_${FOLD}"
  FOLD_CKPT_DIR="${PROJECT_ROOT}/checkpoints/splitcp_fold_${FOLD}"
  FOLD_LOG_DIR="${PROJECT_ROOT}/logs/splitcp_fold_${FOLD}"
  FOLD_RESULT_DIR="${PROJECT_ROOT}/results/splitcp_5fold/fold_${FOLD}"
  mkdir -p "${FOLD_SPLIT_DIR}" "${FOLD_CKPT_DIR}" "${FOLD_LOG_DIR}" "${FOLD_RESULT_DIR}"

  export SPLIT_CALIB_FOLD="${FOLD}"
  export DATA_SPLITS_DIR="${FOLD_SPLIT_DIR}"
  export TRAIN_CSV="${FOLD_SPLIT_DIR}/train_split.csv"
  export CALIBRATION_CSV="${FOLD_SPLIT_DIR}/calibration_split.csv"
  export VAL_CSV="${FOLD_SPLIT_DIR}/validation_split.csv"
  export TEST_CSV="${FOLD_SPLIT_DIR}/test_split.csv"
  export CHECKPOINT_DIR="${FOLD_CKPT_DIR}"
  export LOG_DIR="${FOLD_LOG_DIR}"
  export LOSS_CURVE_FILE="${FOLD_LOG_DIR}/loss_curves.png"
  export GLOBAL_DESC_STATS_CACHE="${FOLD_CKPT_DIR}/global_descriptor_stats.json"
  export RESULTS_DIR="${FOLD_RESULT_DIR}"
  export RUN_SPLIT=1
  export MASTER_PORT="$((29500 + FOLD))"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "[INFO] Training fold ${FOLD}: train=${TRAIN_CSV}, calibration=${CALIBRATION_CSV}"
    bash "${PROJECT_ROOT}/scripts/train.sh"
  else
    echo "[INFO] Skip training fold ${FOLD} (RUN_TRAIN=${RUN_TRAIN})"
    # Ensure split CSVs exist for test/MC. This is cheap and deterministic.
    python "${PROJECT_ROOT}/split_dataset.py"
  fi

  if [[ "${RUN_TEST}" == "1" ]]; then
    export CHECKPOINT_PATH="${FOLD_CKPT_DIR}/best_model.pth"
    if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
      echo "[ERROR] Missing checkpoint for fold ${FOLD}: ${CHECKPOINT_PATH}" >&2
      exit 1
    fi
    echo "[INFO] Testing fold ${FOLD} on AutoCFD official test set"
    # Override to single-process evaluation even though the allocation has 4 GPUs.
    export NPROC_PER_NODE=1
    bash "${PROJECT_ROOT}/scripts/test.sh"
    export NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}"
  else
    echo "[INFO] Skip test fold ${FOLD} (RUN_TEST=${RUN_TEST})"
  fi

  if [[ "${RUN_MC}" == "1" ]]; then
    PRED_CSV="${FOLD_RESULT_DIR}/predictions.csv"
    if [[ ! -f "${PRED_CSV}" ]]; then
      PRED_CSV="${FOLD_RESULT_DIR}/per_sample_cd_cp_intervals.csv"
    fi
    if [[ ! -f "${PRED_CSV}" ]]; then
      echo "[ERROR] Missing prediction CSV for fold ${FOLD}: ${FOLD_RESULT_DIR}/predictions.csv" >&2
      echo "[ERROR] Also tried: ${FOLD_RESULT_DIR}/per_sample_cd_cp_intervals.csv" >&2
      exit 1
    fi
    export CP_EVAL_CACHE="${FOLD_RESULT_DIR}/mc_eval_prediction_csv_cache.npz"
    export CP_EVAL_OUT_JSON="${FOLD_RESULT_DIR}/mc_replicates_summary.json"
    echo "[INFO] Monte-Carlo fold ${FOLD}: predictions_csv=${PRED_CSV}, n_cal=${CP_EVAL_N_CAL}, R=${CP_EVAL_R}"
    python "${PROJECT_ROOT}/evaluate_cp_replicates.py" \
      --predictions-csv "${PRED_CSV}" \
      --n-cal "${CP_EVAL_N_CAL}" \
      --R "${CP_EVAL_R}" \
      --alpha "${CP_EVAL_ALPHA}" \
      --cache "${CP_EVAL_CACHE}" \
      --out-json "${CP_EVAL_OUT_JSON}"
  else
    echo "[INFO] Skip Monte-Carlo fold ${FOLD} (RUN_MC=${RUN_MC})"
  fi
done

echo ""
echo "[INFO] Aggregating split CP 5-fold results ..."
python "${PROJECT_ROOT}/aggregate_splitcp_results.py" \
  --root "${PROJECT_ROOT}/results/splitcp_5fold" \
  --n-folds "${K}"

echo "[OK] Done. Summary: ${PROJECT_ROOT}/results/splitcp_5fold/splitcp_5fold_summary.json"

