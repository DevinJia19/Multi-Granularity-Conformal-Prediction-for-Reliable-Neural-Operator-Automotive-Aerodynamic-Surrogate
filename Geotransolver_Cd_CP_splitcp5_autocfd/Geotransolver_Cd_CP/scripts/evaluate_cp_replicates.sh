#!/usr/bin/env bash
# Slurm entrypoint for split-CP 5-fold Monte Carlo evaluation.
#
# This script reads existing per-fold prediction CSV files under
# results/splitcp_5fold/fold_*/. It does not load checkpoints, does not run
# model inference, and does not use final-model in-sample predictions.
#
# Run from the project root:
#   sbatch scripts/evaluate_cp_replicates.sh
#
# Optional overrides:
#   CP_EVAL_N_CAL=40 CP_EVAL_R=1000 sbatch scripts/evaluate_cp_replicates.sh
#SBATCH --job-name=geo-cp-mc
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/cp_mc_%j.out
#SBATCH --error=logs/cp_mc_%j.err

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    submit_dir="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
    if [[ "$(basename "${submit_dir}")" == "scripts" ]]; then
      PROJECT_ROOT="$(cd "${submit_dir}/.." && pwd)"
    else
      PROJECT_ROOT="${submit_dir}"
    fi
  else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
    PROJECT_ROOT="$(cd "${script_dir}/.." && pwd)"
  fi
fi

export PROJECT_ROOT
cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/splitcp_5fold"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

CONDA_ENV="${CP_EVAL_CONDA_ENV:-geotrans_py311}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  source "${CONDA_SH}"
else
  echo "[ERROR] conda not found. Load conda first or set CONDA_SH=/path/to/conda.sh" >&2
  exit 1
fi
conda activate "${CONDA_ENV}"

ROOT="${CP_EVAL_SPLITCP_ROOT:-${PROJECT_ROOT}/results/splitcp_5fold}"
K="${SPLITCP_N_SPLITS:-${SPLIT_N_SPLITS:-5}}"
N_CAL="${CP_EVAL_N_CAL:-40}"
R="${CP_EVAL_R:-500}"
ALPHA="${CP_EVAL_ALPHA:-0.1}"

echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] K=${K}"
echo "[INFO] CP_EVAL_N_CAL=${N_CAL} CP_EVAL_R=${R} CP_EVAL_ALPHA=${ALPHA}"
echo "[INFO] Reading existing fold prediction CSV files only."

for ((FOLD = 0; FOLD < K; FOLD++)); do
  fold_dir="${ROOT}/fold_${FOLD}"
  pred_csv="${fold_dir}/predictions.csv"
  if [[ ! -f "${pred_csv}" ]]; then
    pred_csv="${fold_dir}/per_sample_cd_cp_intervals.csv"
  fi
  if [[ ! -f "${pred_csv}" ]]; then
    echo "[ERROR] Missing prediction CSV for fold ${FOLD}: ${fold_dir}/predictions.csv" >&2
    echo "[ERROR] Also tried: ${fold_dir}/per_sample_cd_cp_intervals.csv" >&2
    exit 1
  fi

  cache_npz="${fold_dir}/mc_eval_prediction_csv_cache.npz"
  out_json="${fold_dir}/mc_replicates_summary.json"

  echo "[INFO] Fold ${FOLD}: predictions_csv=${pred_csv}"
  python "${PROJECT_ROOT}/evaluate_cp_replicates.py" \
    --predictions-csv "${pred_csv}" \
    --n-cal "${N_CAL}" \
    --R "${R}" \
    --alpha "${ALPHA}" \
    --cache "${cache_npz}" \
    --out-json "${out_json}"
done

echo "[INFO] Aggregating split-CP 5-fold Monte Carlo summaries..."
python "${PROJECT_ROOT}/aggregate_splitcp_results.py" \
  --root "${ROOT}" \
  --n-folds "${K}"

echo "[OK] Done: ${ROOT}/splitcp_5fold_summary.json"
