#!/usr/bin/env bash
# Merge 4-fold CV+ OOF pointwise .npz and compute channel-wise conformal q_hat.
#
# Submit from repository root:
#   sbatch scripts/calibrate_cvplus_only.sh
#
# Optional env: CP_ALPHA=0.1

#SBATCH --job-name=pressure-cpmerge
#SBATCH --partition=alvis
#SBATCH --account=naiss2025-22-1747
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/pressure_cpmerge_%j.out
#SBATCH --error=logs/pressure_cpmerge_%j.err

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export PROJECT_ROOT
cd "${PROJECT_ROOT}"

mkdir -p logs
mkdir -p results/pressure_cvplus_cp

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source "/cephyr/users/chaoxi/Alvis/Desktop/mimer_naiss2025-23-604/Liyujing/Anaconda3/etc/profile.d/conda.sh"
fi

conda activate geotran_pressure

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

# 这个脚本本身不需要 GPU，即使申请了 1 张，也强制不用 CUDA
export CUDA_VISIBLE_DEVICES=""

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "Running CV+ calibration merge only..."
echo "Start time: $(date)"
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"

python "${PROJECT_ROOT}/cp_calibrate_cvplus_normalized.py" \
  --calib-glob "runs/cvplus/fold_*/cvplus_oof/fold_*" \
  --alpha "${CP_ALPHA:-0.1}" \
  --out "${PROJECT_ROOT}/results/pressure_cvplus_cp"

echo "Finished at: $(date)"
