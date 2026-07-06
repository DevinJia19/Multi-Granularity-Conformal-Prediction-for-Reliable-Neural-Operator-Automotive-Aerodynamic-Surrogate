#!/usr/bin/env bash
#SBATCH --job-name=pressure-splitcp4
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00
#SBATCH --output=logs/pressure_splitcp4_%j.out
#SBATCH --error=logs/pressure_splitcp4_%j.err
#
# Ordinary split CP 4-fold baseline under the AutoCFD official split.
#
# IMPORTANT: this script is NOT CV+.
#   - AutoCFD official train pool (400 cases) is split into 4 folds.
#   - For each fold: 3 folds are used to train the model, 1 fold is used only
#     to estimate qhat on that fold's calibration set.
#   - No OOF predictions are merged. No final full-400 refit model is used.
#   - The official AutoCFD test set is evaluated 4 times, once per fold model,
#     and metrics are averaged at the end.
#   - The script also builds averaged test NPZ + averaged_qhat.json so the
#     existing VTP writer can directly generate visualization files.
#
# Usage:
#   cd /path/to/Geotransolver_Pressure
#   sbatch scripts/run_splitcp_4fold.sh
#
# Resume / partial usage:
#   RUN_TRAIN=0 RUN_INFER=1 RUN_CP=1 sbatch scripts/run_splitcp_4fold.sh
#   SPLITCP_START_FOLD=2 sbatch scripts/run_splitcp_4fold.sh
#   RUN_VTP=1 sbatch scripts/run_splitcp_4fold.sh
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
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/pressure_splitcp_4fold"
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
START_FOLD="${SPLITCP_START_FOLD:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_CP="${RUN_CP:-1}"
RUN_AGG="${RUN_AGG:-1}"
RUN_VTP="${RUN_VTP:-0}"
MAKE_AVG_NPZ="${MAKE_AVG_NPZ:-1}"

if ! [[ "${K}" =~ ^[0-9]+$ ]] || (( K != 4 )); then
  echo "[ERROR] Pressure split-CP baseline is expected to use K=4. Got K=${K}." >&2
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
printf '[INFO] Ordinary pressure split CP folds: %s .. %s, K=%s\n' "${START_FOLD}" "$((K - 1))" "${K}"
printf '[INFO] RUN_TRAIN=%s RUN_INFER=%s RUN_CP=%s RUN_AGG=%s RUN_VTP=%s MAKE_AVG_NPZ=%s\n' \
  "${RUN_TRAIN}" "${RUN_INFER}" "${RUN_CP}" "${RUN_AGG}" "${RUN_VTP}" "${MAKE_AVG_NPZ}"
printf '[INFO] This is NOT CV+: no OOF merge and no final full-400 refit.\n\n'

for ((FOLD = START_FOLD; FOLD < K; FOLD++)); do
  echo ""
  echo "==================== Pressure Split CP fold ${FOLD}/${K} ===================="

  FOLD_RUN_ID="splitcp_4fold/fold_${FOLD}"
  FOLD_RESULT_DIR="${PROJECT_ROOT}/results/pressure_splitcp_4fold/fold_${FOLD}"
  FOLD_NORM_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/normalization"
  mkdir -p "${FOLD_RESULT_DIR}" "${FOLD_NORM_DIR}"

  echo "[INFO] Generating AutoCFD official split with fold ${FOLD} as calibration holdout"
  SPLIT_ARGS=(--n-splits 4 --calib-fold "${FOLD}" --symlink --force)
  if [[ -n "${DATASET_DIR:-}" ]]; then
    SPLIT_ARGS+=(--dataset-dir "${DATASET_DIR}")
  fi
  python "${PROJECT_ROOT}/split_dataset.py" "${SPLIT_ARGS[@]}"

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    echo "[INFO] Training fold ${FOLD}: AutoCFD train 400 -> 300 train + 100 calib"
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
    echo "[INFO] Skip training fold ${FOLD} (RUN_TRAIN=${RUN_TRAIN})"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    CKPT_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/checkpoints_best"
    if [[ ! -d "${CKPT_DIR}" ]]; then
      echo "[ERROR] Missing checkpoint directory for fold ${FOLD}: ${CKPT_DIR}" >&2
      exit 1
    fi
    if [[ ! -f "${FOLD_NORM_DIR}/surface_fields_normalization.npz" ]]; then
      echo "[WARN] Fold normalization file missing: ${FOLD_NORM_DIR}/surface_fields_normalization.npz"
      echo "[WARN] Inference will fall back to current data.normalization_dir default if not overridden correctly."
    fi

    echo "[INFO] Inference on fold ${FOLD} calibration set only; this qhat will NOT be merged with other folds."
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" \
    MASTER_PORT="$((29600 + FOLD))" \
    bash "${PROJECT_ROOT}/scripts/test.sh" \
      "run_id=${FOLD_RUN_ID}" \
      "checkpoint_dir=${CKPT_DIR}" \
      "inference.phase=val" \
      "data.val.data_path=data_splits/calib" \
      "data.normalization_dir=${FOLD_NORM_DIR}" \
      "cp_output.save_pointwise_npz=true" \
      "cp_output.subdir=splitcp_calib/fold_${FOLD}" \
      "data.resolution=8192"

    echo "[INFO] Inference on AutoCFD official test set for fold ${FOLD}."
    NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}" \
    MASTER_PORT="$((29700 + FOLD))" \
    bash "${PROJECT_ROOT}/scripts/test.sh" \
      "run_id=${FOLD_RUN_ID}" \
      "checkpoint_dir=${CKPT_DIR}" \
      "inference.phase=val" \
      "data.val.data_path=data_splits/test" \
      "data.normalization_dir=${FOLD_NORM_DIR}" \
      "cp_output.save_pointwise_npz=true" \
      "cp_output.subdir=splitcp_test/fold_${FOLD}" \
      "data.resolution=8192"
  else
    echo "[INFO] Skip inference fold ${FOLD} (RUN_INFER=${RUN_INFER})"
  fi

  if [[ "${RUN_CP}" == "1" ]]; then
    CALIB_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/splitcp_calib/fold_${FOLD}"
    TEST_DIR="${PROJECT_ROOT}/runs/${FOLD_RUN_ID}/splitcp_test/fold_${FOLD}"
    if [[ ! -d "${CALIB_DIR}" || ! -d "${TEST_DIR}" ]]; then
      echo "[ERROR] Missing calibration/test npz directory for fold ${FOLD}." >&2
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
    echo "[INFO] Computing fold-local qhat and test metrics for fold ${FOLD}."
    python "${PROJECT_ROOT}/cp_compare_global_point_case.py" "${CP_ARGS[@]}"
  else
    echo "[INFO] Skip CP comparison fold ${FOLD} (RUN_CP=${RUN_CP})"
  fi
done

if [[ "${RUN_AGG}" == "1" ]]; then
  echo ""
  echo "[INFO] Aggregating 4-fold ordinary split CP results ..."
  AGG_ARGS=(
    --root "${PROJECT_ROOT}/results/pressure_splitcp_4fold"
    --runs-root "${PROJECT_ROOT}/runs/splitcp_4fold"
    --n-folds 4
  )
  if [[ "${MAKE_AVG_NPZ}" == "1" ]]; then
    AGG_ARGS+=(--make-average-npz)
  fi
  python "${PROJECT_ROOT}/aggregate_splitcp4_pressure_results.py" "${AGG_ARGS[@]}"
else
  echo "[INFO] Skip aggregation (RUN_AGG=${RUN_AGG})"
fi

if [[ "${RUN_VTP}" == "1" ]]; then
  echo ""
  echo "[INFO] Writing VTP visualization files from averaged 4-fold outputs ..."
  bash "${PROJECT_ROOT}/scripts/write_splitcp_4fold_vtp.sh"
fi

echo "[OK] Done. Summary: ${PROJECT_ROOT}/results/pressure_splitcp_4fold/splitcp_4fold_summary.json"
