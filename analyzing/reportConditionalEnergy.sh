#!/bin/bash
#SBATCH --job-name=conditional-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=p1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=conditional_report_%j.out
#SBATCH --error=conditional_report_%j.errs

set -euo pipefail
PROJECT_ROOT=/data/home/zyh/work/jaqmc_merge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/conditional-report-${SLURM_JOB_ID:-local}"
PLOT_PYTHON="${PLOT_PYTHON:-/data/home/zyh/work/jaqmc/.venv/bin/python}"
exec "$PLOT_PYTHON" -u analyzing/reportConditionalEnergy.py "$@"
