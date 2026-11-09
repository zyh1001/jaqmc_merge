#!/bin/bash
#SBATCH --job-name=conditional-energy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=conditional_energy_%j.out
#SBATCH --error=conditional_energy_%j.errs

set -euo pipefail
PROJECT_ROOT=/data/home/zyh/work/jaqmc_merge
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for argument in "$@"; do
    if [[ "$argument" == --dry-run ]]; then
        exec "$PROJECT_ROOT/.venv/bin/python" -u analyzing/evaluateConditionalEnergy.py "$@"
    fi
done
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "计算请使用 sbatch analyzing/evaluateConditionalEnergy.sh；输入检查可用 bash 加 --dry-run。" >&2
    exit 2
fi
exec "$PROJECT_ROOT/.venv/bin/python" -u analyzing/evaluateConditionalEnergy.py "$@"
