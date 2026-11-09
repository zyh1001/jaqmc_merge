#!/bin/bash
#SBATCH --job-name=disk_eval_config
#SBATCH --partition=p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=disk_eval_config_%j.out
#SBATCH --error=disk_eval_config_%j.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/home/zyh/work/jaqmc_merge}
RUN_DIR=${1:-/data/home/zyh/work/data/disk/n6_Q18_kapa1_8192_final}
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "${PROJECT_ROOT}"

"${PROJECT_ROOT}/.venv/bin/python" -m pytest tests/analyzing/disk_evaluation_test.py -q
for quantity in energy density pair_correlation wave_scan; do
    bash evaluate_disk.sh "${quantity}" "${RUN_DIR}" --dry-run
done
