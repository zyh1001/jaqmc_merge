#!/usr/bin/env bash
#SBATCH --job-name=hall-integration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=v100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=integration_test_results/slurm-%j.out
#SBATCH --error=integration_test_results/slurm-%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; numerical tests require a compute node." >&2
    exit 2
fi
suite="${1:-all}"
trap 'status=$?; if (( status == 0 )); then echo "PASS: $suite"; else echo "FAIL: $suite (exit $status)"; fi; exit "$status"' EXIT
cd /data/home/zyh/work/jaqmc_merge
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-3}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cuda
result_dir="$PWD/integration_test_results/${SLURM_JOB_ID}-${suite}"
mkdir -p "$result_dir"
exec > >(tee "$result_dir/stdout.log") 2> >(tee "$result_dir/stderr.log" >&2)
printf 'Suite: %s\nDirectory: %s\nPython: %s\n' "$suite" "$PWD" "$(command -v python)"
python -m pytest -o addopts= -q -s tests/app/hall/multigeometry_test.py::test_environment_report

run_tests() {
    local label="$1"
    shift
    python -m pytest -o addopts= -q -s --basetemp="$result_dir/$label" \
        --junitxml="$result_dir/$label.xml" "$@"
}

run_suite() {
    case "$1" in
        sphere)
            run_tests sphere tests/app/hall/multigeometry_test.py -k 'sphere or geometry_config'
            run_tests sphere-upstream tests/app/hall/workflow_test.py \
                tests/app/hall/wavefunction_test.py tests/app/hall/hamiltonian_test.py
            ;;
        torus)
            run_tests torus tests/app/hall/multigeometry_test.py -k torus
            run_tests torus-source tests/app/hall/torus_mhpo_test.py \
                tests/app/hall/torus_sigma_mhpo_test.py tests/app/hall/torus_one_rdm_test.py \
                tests/geometry/torus_test.py tests/estimator/fubini_test.py
            ;;
        disk)
            run_tests disk tests/app/hall/multigeometry_test.py -k 'source_wavefunctions_and_hamiltonians and disk or train_restore_evaluate and disk'
            ;;
        estimators)
            run_tests estimators tests/app/hall/multigeometry_test.py -k 'histograms or observable_workflow or overlap or estimator_source or scan_selection or penalty'
            ;;
        other)
            run_tests shared tests/app/hall/multigeometry_test.py::test_split_channel_defaults_match_upstream
            run_tests other tests/hydrogen/molecule_test.py::test_without_pretrain \
                tests/app/solid/workflow_test.py tests/utils/module_resolver_test.py \
                tests/utils/checkpoint_test.py
            ;;
        parallel)
            # Two CPU devices exercise collectives within the compute allocation.
            JAX_PLATFORMS=cpu run_tests parallel tests/app/hall/multigeometry_test.py \
                -k 'histograms or scan_selection or overlap_analytic' --n-cpu-devices=2
            ;;
        *) echo "Unknown suite: $1" >&2; return 2 ;;
    esac
}
if [[ "$suite" == all ]]; then
    for part in sphere torus disk estimators other parallel; do
        run_suite "$part"
    done
else
    run_suite "$suite"
fi
