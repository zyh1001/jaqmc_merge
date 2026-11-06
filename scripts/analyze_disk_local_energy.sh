#!/bin/bash
#SBATCH --job-name=disk-local-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=disk_local_energy_%j.out
#SBATCH --error=disk_local_energy_%j.errs

set -euo pipefail
PROJECT_ROOT=/data/home/zyh/work/jaqmc_merge
PLOT_PYTHON=/data/home/zyh/work/jaqmc/.venv/bin/python
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/disk-local-energy-matplotlib-${SLURM_JOB_ID:-local}"
PROGRAM="$PROJECT_ROOT/scripts/analyze_disk_local_energy.py"
for argument in "$@"; do
    if [[ "$argument" == --dry-run || "$argument" == --self-test ]]; then
        exec "$PROJECT_ROOT/.venv/bin/python" -u "$PROGRAM" "$@"
    fi
    if [[ "$argument" == --plot-only ]]; then
        exec "$PLOT_PYTHON" -u "$PROGRAM" "$@"
    fi
done
"$PROJECT_ROOT/.venv/bin/python" -u "$PROGRAM" "$@"
"$PLOT_PYTHON" -u "$PROGRAM" "$@" --plot-only
