#!/bin/bash
#SBATCH --job-name=sphere_n6_flux15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --partition=v100
#SBATCH --gres=gpu:8
#SBATCH --output=sphere_%j.out
#SBATCH --error=sphere_%j.err

set -euo pipefail

# 进入工作目录
run_dir="${SLURM_SUBMIT_DIR}/data/mhpo-1det"
mkdir -p "$run_dir"
cd "$run_dir"

# 公共配置
SHARED=(
    "system.nspins=[6,0]"
    system.flux=15
    wf.module=mhpo
    wf.num_layers=2
    wf.ndets=1
    workflow.batch_size=8192
    sampler.steps=10
)

# 1. 神经网络训练
echo "===== Training ====="

uv run --extra cuda12 jaqmc hall train \
    "${SHARED[@]}" \
    train.run.iterations=200000 \
    workflow.save_path=train

# 2. 常规评估
echo "===== Evaluation ====="

uv run --extra cuda12 jaqmc hall evaluate \
    "${SHARED[@]}" \
    run.iterations=10000 \
    workflow.source_path=train \
    workflow.save_path=eval

# 3. Pair Correlation 评估
echo "===== Pair Correlation ====="

uv run --extra cuda12 jaqmc hall evaluate \
    "${SHARED[@]}" \
    run.iterations=200000 \
    estimators.enabled.energy=false \
    estimators.enabled.angular_momentum=false \
    estimators.enabled.pair_correlation=true \
    workflow.source_path=train \
    workflow.save_path=pair_corr

echo "===== All stages completed ====="