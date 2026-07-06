#!/usr/bin/env bash
# Generate final calibrated intervals for a trained Transolver transfer run.
#
# The script writes three interval sets:
#   global_abs  : global absolute-width CP
#   point_sigma : point-wise sigma-scaled CP
#   case_sigma  : case-wise sigma-scaled CP
#
# Default paths match scripts/run_transolver_single_cp.sh for split 0.
#
# Usage:
#   cd /path/to/Transolver_Pressure_transfer
#   bash scripts/generate_final_intervals_global_point_case.sh
#
# Common overrides:
#   SPLIT=2 bash scripts/generate_final_intervals_global_point_case.sh
#   QHAT_JSON=results/pressure_transolver_single_cp/split_0/qhat.json bash scripts/generate_final_intervals_global_point_case.sh
#   WRITE_VTP=1 bash scripts/generate_final_intervals_global_point_case.sh
set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  PROJECT_ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
fi
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

SPLIT="${SPLIT:-${SPLITCP_START_SPLIT:-0}}"
RUN_ID="${RUN_ID:-transolver_single_cp/split_${SPLIT}}"
CALIB_DIR="${CALIB_DIR:-${PROJECT_ROOT}/runs/${RUN_ID}/single_cp_calib/split_${SPLIT}}"
TEST_DIR="${TEST_DIR:-${PROJECT_ROOT}/runs/${RUN_ID}/single_cp_test/split_${SPLIT}}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/results/pressure_transolver_single_cp/split_${SPLIT}/intervals}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/results/pressure_transolver_single_cp/split_${SPLIT}}"
QHAT_JSON="${QHAT_JSON:-}"
ALPHA="${CP_ALPHA:-0.1}"
CASE_SCORE="${CASE_SCORE:-quantile}"
SCORE_SAMPLE_PER_FILE="${SCORE_SAMPLE_PER_FILE:-}"
WRITE_VTP="${WRITE_VTP:-0}"
MODES="${MODES:-global_abs point_sigma case_sigma}"

mkdir -p "${OUT_DIR}" "${RESULT_DIR}"

INTERVAL_ARGS=(
  --test-dir "${TEST_DIR}"
  --out "${OUT_DIR}"
  --alpha "${ALPHA}"
  --case-score "${CASE_SCORE}"
  --modes ${MODES}
)

if [[ -n "${QHAT_JSON}" ]]; then
  INTERVAL_ARGS+=(--qhat-json "${QHAT_JSON}")
else
  INTERVAL_ARGS+=(--calib-dir "${CALIB_DIR}")
fi

if [[ -n "${SCORE_SAMPLE_PER_FILE}" ]]; then
  INTERVAL_ARGS+=(--score-sample-per-file "${SCORE_SAMPLE_PER_FILE}")
fi

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] CALIB_DIR=${CALIB_DIR}"
echo "[INFO] TEST_DIR=${TEST_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] MODES=${MODES}"
python "${PROJECT_ROOT}/cp_generate_intervals_global_point_case.py" "${INTERVAL_ARGS[@]}"

if [[ "${WRITE_VTP}" == "1" ]]; then
  if [[ -n "${QHAT_JSON}" ]]; then
    VTP_QHAT="${QHAT_JSON}"
  else
    VTP_QHAT="${OUT_DIR}/qhat.json"
  fi
  python "${PROJECT_ROOT}/cp_write_vtp_global_point_case.py" \
    --test-dir "${TEST_DIR}" \
    --qhat-json "${VTP_QHAT}" \
    --out "${OUT_DIR}/vtp" \
    --modes ${MODES}
fi

echo "[OK] Final calibrated intervals are under: ${OUT_DIR}"
