#!/bin/bash
#SBATCH --job-name=disk_eval
#SBATCH --partition=v100g32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=disk_eval_%j.out
#SBATCH --error=disk_eval_%j.err

set -euo pipefail

# Slurm 会复制脚本，项目路径不从复制后的脚本位置推断。
PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zyh/work/jaqmc_merge}
PYTHON_BIN=${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/disk-eval-mpl-${SLURM_JOB_ID:-$$}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/analyzing/evaluateDisk.py" "$@"
