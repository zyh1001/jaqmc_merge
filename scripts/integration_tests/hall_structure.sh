#!/usr/bin/env bash
#SBATCH --job-name=hall-structure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=hall-structure-%j.out
#SBATCH --error=hall-structure-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; all runtime checks use a compute node." >&2
    exit 2
fi
cd /data/home/zyh/work/jaqmc_merge
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec .venv/bin/python scripts/integration_tests/hall_structure.py "$@"
