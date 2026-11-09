#!/bin/bash
#SBATCH -J disk_n4_kapa3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=v100
#SBATCH --gres=gpu:4
#SBATCH --chdir=/data/home/zyh/work/jaqmc_merge
#SBATCH --output=%j.out
#SBATCH --error=%j.errs

set -euo pipefail
# 进入工作目录
cd "${PROJECT_ROOT:-/data/home/zyh/work/jaqmc_merge}"

# 加载 Python 环境
source .venv/bin/activate
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

export JAX_DEBUG_NANS=True

# 训练磁盘波函数，末尾参数可覆盖默认配置
jaqmc hall train system.module=jaqmc.app.hall.config:HallDiskConfig \
    workflow.save_path=/data/home/zyh/work/data/disk/n6_Q18_kapa1_8192_final_fixed \
    system.flux=18 system.nspins="[6, 0]" \
    system.d=0.1 \
    wf.envelope_type=m_learnable2 \
    system.interaction_strength=1 \
    system.num_orbitals=18 \
    train.run.iterations=150000 \
    workflow.batch_size=8192 \
    train.optim.module=jaqmc.optimizer.kfac \
    train.optim.learning_rate.rate=0.05 \
    train.optim.learning_rate.delay=200000 \
    "$@"
