#!/usr/bin/env bash
#SBATCH --job-name=transolver-single-cp
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=logs/transolver_single_cp_%j.out
#SBATCH --error=logs/transolver_single_cp_%j.err
#
# Single-run ordinary split CP with a plain Transolver model under the AutoCFD official split.
#
# IMPORTANT:
#   - This script runs exactly one train/infer/calibrate/test CP pipeline.
#   - It does not merge OOF scores and it does not average multiple models.
#   - AutoCFD official train pool (400 cases) is split into 4 folds.
#   - One selected split is run by default: 300 cases train the model, 100 cases estimate qhat.
#   - No OOF predictions are merged. No final full-400 refit model is used.
#
# Usage:
#   cd /path/to/Transolver_Pressure_transfer
#   sbatch scripts/run_transolver_single_cp.sh
#
# Resume / partial usage:
#   RUN_TRAIN=0 RUN_INFER=1 RUN_CP=1 sbatch scripts/run_transolver_single_cp.sh
#   SPLITCP_START_SPLIT=2 sbatch scripts/run_transolver_single_cp.sh
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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/pressure_transolver_single_cp"
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

K="${SPLITCP_N_SPLITS:-${SPLIT_N_SPLITS:-4}}"
START_FOLD="${SPLITCP_START_SPLIT:-${SPLITCP_START_FOLD:-0}}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_CP="${RUN_CP:-1}"
RUN_AGG="${RUN_AGG:-0}"
RUN_VTP="${RUN_VTP:-0}"
MAKE_AVG_NPZ="${MAKE_AVG_NPZ:-1}"

if ! [[ "${K}" =~ ^[0-9]+$ ]] || (( K != 4 )); then
  echo "[ERROR] Transolver single-run CP expects 4 splits so the official train pool becomes 300 train + 100 calibration. Got K=${K}." >&2
  exit 1
fi
if ! [[ "${START_FOLD}" =~ ^[0-9]+$ ]] || (( START_FOLD < 0 || START_FOLD >= K )); then
  echo "[ERROR] SPLITCP_START_FOLD=${START_FOLD} invalid; expected [0, $((K - 1))]" >&2
  exit 1
fi

export SPLIT_N_SPLITS="${K}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
export CP_ALPHA="${CP_ALPHA:-0.1}"
export CASE_SCORE="${CASE_SCORE:-quantile}"
export SCORE_SAMPLE_PER_FILE="${SCORE_SAMPLE_PER_FILE:-}"
TRAIN_NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

printf '\n[INFO] PROJECT_ROOT=%s\n' "${PROJECT_ROOT}"
END_FOLD="$((START_FOLD + 1))"

printf '[INFO] Transolver single-run split CP: calibration split %s, K=%s\n' "${START_FOLD}" "${K}"
printf '[INFO] RUN_TRAIN=%s RUN_INFER=%s RUN_CP=%s RUN_AGG=%s RUN_VTP=%s MAKE_AVG_NPZ=%s\n' \
  "${RUN_TRAIN}" "${RUN_INFER}" "${RUN_CP}" "${RUN_AGG}" "${RUN_VTP}" "${MAKE_AVG_NPZ}"
printf '[INFO] Expected counts: 300 train, 100 calibration, 34 validation, 50 official test.\n'
printf '[INFO] Plain Transolver CP: one run only, no OOF merge, no model averaging, no final full-400 refit.\n\n'

for ((FOLD = START_FOLD; FOLD < END_FOLD; FOLD++)); do
  echo ""
  echo "==================== Transolver single CP split ${FOLD}/${K} ===================="

  FOLD_RUN_ID="transolver_single_cp/split_${FOLD}"
  FOLD_RESULT_DIR="${PROJECT_ROOT}/results/pressure_transolver_single_cp/split_${FOLD}"
  FOLD_NORM_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/normalization"
  mkdir -p "${FOLD_RESULT_DIR}" "${FOLD_NORM_DIR}"

  echo "[INFO] Generating AutoCFD official split with split ${FOLD} as the 100-case calibration holdout"
  SPLIT_ARGS=(--n-splits "${K}" --calib-fold "${FOLD}" --symlink --force)
  if [[ -n "${DATASET_DIR:-}" ]]; then
    SPLIT_ARGS+=(--dataset-dir "${DATASET_DIR}")
  fi
  python "${PROJECT_ROOT}/split_dataset.py" "${SPLIT_ARGS[@]}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "[INFO] Training split ${FOLD}: AutoCFD train 400 -> 300 train + 100 calib"
    RUN_SPLIT=0 \
    RUN_NORM="${RUN_NORM:-1}" \
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" \
    MASTER_PORT="$((29500 + FOLD))" \
    bash "${PROJECT_ROOT}/scripts/train.sh" \
      "run_id=${FOLD_RUN_ID}" \
      "data.resolution=8192" \
      "training.skip_validation=false" \
      "training.validate_every=${VALIDATE_EVERY}" \
      "training.early_stopping_patience=${EARLY_STOPPING_PATIENCE}"

    if [[ -f "${PROJECT_ROOT}/surface_fields_normalization.npz" ]]; then
      cp "${PROJECT_ROOT}/surface_fields_normalization.npz" "${FOLD_NORM_DIR}/surface_fields_normalization.npz"
      echo "[INFO] Saved fold normalization to ${FOLD_NORM_DIR}/surface_fields_normalization.npz"
    else
      echo "[WARN] surface_fields_normalization.npz was not found after training fold ${FOLD}."
    fi
  else
    echo "[INFO] Skip training split ${FOLD} (RUN_TRAIN=${RUN_TRAIN})"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    CKPT_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/checkpoints_best"
    if [[ ! -d "${CKPT_DIR}" ]]; then
      echo "[ERROR] Missing checkpoint directory for split ${FOLD}: ${CKPT_DIR}" >&2
      exit 1
    fi
    if [[ ! -f "${FOLD_NORM_DIR}/surface_fields_normalization.npz" ]]; then
      echo "[WARN] Split normalization file missing: ${FOLD_NORM_DIR}/surface_fields_normalization.npz"
      echo "[WARN] Inference will fall back to current data.normalization_dir default if not overridden correctly."
    fi

    echo "[INFO] Inference on the 100-case calibration set only; this qhat will not be merged with other splits."
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" \
    MASTER_PORT="$((29600 + FOLD))" \
    bash "${PROJECT_ROOT}/scripts/test.sh" \
      "run_id=${FOLD_RUN_ID}" \
      "checkpoint_dir=${CKPT_DIR}" \
      "inference.phase=val" \
      "data.val.data_path=data_splits/calib" \
      "data.normalization_dir=${FOLD_NORM_DIR}" \
      "cp_output.save_pointwise_npz=true" \
      "cp_output.subdir=single_cp_calib/split_${FOLD}" \
      "data.resolution=8192"

    echo "[INFO] Inference on the AutoCFD official 50-case test set."
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" \
    MASTER_PORT="$((29700 + FOLD))" \
    bash "${PROJECT_ROOT}/scripts/test.sh" \
      "run_id=${FOLD_RUN_ID}" \
      "checkpoint_dir=${CKPT_DIR}" \
      "inference.phase=val" \
      "data.val.data_path=data_splits/test" \
      "data.normalization_dir=${FOLD_NORM_DIR}" \
      "cp_output.save_pointwise_npz=true" \
      "cp_output.subdir=single_cp_test/split_${FOLD}" \
      "data.resolution=8192"
  else
    echo "[INFO] Skip inference split ${FOLD} (RUN_INFER=${RUN_INFER})"
  fi

  if [[ "${RUN_CP}" == "1" ]]; then
    CALIB_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/single_cp_calib/split_${FOLD}"
    TEST_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/single_cp_test/split_${FOLD}"
    if [[ ! -d "${CALIB_DIR}" || ! -d "${TEST_DIR}" ]]; then
      echo "[ERROR] Missing calibration/test npz directory for split ${FOLD}." >&2
      echo "        CALIB_DIR=${CALIB_DIR}" >&2
      echo "        TEST_DIR=${TEST_DIR}" >&2
      exit 1
    fi

    CP_ARGS=(
      --calib-dir "${CALIB_DIR}"
      --test-dir "${TEST_DIR}"
      --alpha "${CP_ALPHA}"
      --case-score "${CASE_SCORE}"
      --out "${FOLD_RESULT_DIR}"
    )
    if [[ -n "${SCORE_SAMPLE_PER_FILE}" ]]; then
      CP_ARGS+=(--score-sample-per-file "${SCORE_SAMPLE_PER_FILE}")
    fi
    echo "[INFO] Computing single-split qhat and test metrics for split ${FOLD}."
    python "${PROJECT_ROOT}/cp_compare_global_point_case.py" "${CP_ARGS[@]}"
  else
    echo "[INFO] Skip CP comparison split ${FOLD} (RUN_CP=${RUN_CP})"
  fi
done

if [[ "${RUN_AGG}" == "1" ]]; then
  echo "[WARN] RUN_AGG=1 was requested, but this script intentionally runs one Transolver CP split."
  echo "[WARN] Skipping aggregation/averaging."
else
  echo "[INFO] Skip aggregation (RUN_AGG=${RUN_AGG})"
fi

if [[ "${RUN_VTP}" == "1" ]]; then
  echo "[WARN] RUN_VTP=1 was requested, but this single-run CP script does not generate averaged VTP inputs."
  echo "[WARN] Skipping VTP writing for the single-fold transfer check."
fi

echo "[OK] Done. Summary: ${PROJECT_ROOT}/results/pressure_transolver_single_cp/split_${START_FOLD}/summary.json"
