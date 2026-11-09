# 圆盘训练结果的物理量评估

`evaluate_disk.sh` 使用当前项目的代码恢复训练检查点，每次提交一项独立评估。
脚本申请一个进程、三个 CPU 核、16 GB 主机内存和一张 `v100g32` 队列的 GPU。
计算结束后调用已有分析程序，原训练结果和已有绘图代码保持不变。

## 提交计算

```bash
cd /data/home/zyh/work/jaqmc_merge
RUN_DIR=/data/home/zyh/work/data/disk/n6_Q18_kapa1_8192_final
sbatch evaluate_disk.sh energy "$RUN_DIR"
sbatch evaluate_disk.sh density "$RUN_DIR"
sbatch evaluate_disk.sh pair_correlation "$RUN_DIR" --reference-r 3.0
sbatch evaluate_disk.sh wave_scan "$RUN_DIR"
```

默认使用 `train_ckpt_149999.npz`。脚本读取 `train_config.yaml` 中的物理参数、网络设置和采样设置，
实际采样配置数固定为检查点中的 8192，不继承训练优化器或旧评估状态。

| 物理量 | 正式评估步数 | 默认设置 | 输出目录，相对于训练目录 |
|---|---:|---|---|
| `energy` | 10000 | 同时计算角动量，动能按64个配置分块 | `eval/energy` |
| `density` | 3000 | 400×400网格，两个方向均为−10至10 | `eval/density` |
| `pair_correlation` | 4000 | 参考位置为(3,0)，800×800网格，两个方向均为−15至15 | `eval/pair_correlation_r3.0` |
| `wave_scan` | 1 | 从8192个配置选取背景，301×301网格，范围−8至8 | `eval/wavefunction_scan` |

共同采用2000步预热，随机种子为1。每次评估之间的采样移动步数从训练配置继承。
波函数扫描另保存0.001至3之间的200点径向扫描。
这些默认值是初始计算设置，误差和收敛性需要根据正式结果检查。

## 参数

```bash
sbatch evaluate_disk.sh energy "$RUN_DIR" --iterations 20000 --burn-in 2000
sbatch evaluate_disk.sh pair_correlation "$RUN_DIR" --reference-r 5.0
sbatch evaluate_disk.sh density "$RUN_DIR" --save-path "$RUN_DIR/eval/density_repeat"
sbatch evaluate_disk.sh wave_scan "$RUN_DIR" estimators.wavefunction_scan.n_x=401 estimators.wavefunction_scan.n_y=401
```

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 来源检查点，绝对路径或相对于训练目录的路径 |
| `--iterations` | 覆盖所选评估的默认步数 |
| `--burn-in` | 预热步数，默认2000 |
| `--seed` | 随机种子，默认1 |
| `--reference-r` | 对关联参考半径，默认3 |
| `--save-path` | 新输出目录；已存在的目录不会覆盖或续算 |
| `--dry-run` | 检查配置，不运行采样，也不创建正式输出目录 |
| `--skip-analysis` | 保存计算结果，不调用分析程序 |

末尾 `key=value` 可以覆盖 `run`、`sampler`、`jax`、`logging` 及估计器设置。
物理模型、来源检查点、实际配置数和启用的物理量通过固定配置或专用参数确定，不能用末尾参数更改。
需要改变批量或物理模型时，应先确认检查点和评估流程相容。

脚本默认项目位置是 `/data/home/zyh/work/jaqmc_merge`，可以通过 `PROJECT_ROOT` 修改。
分析程序默认来自 `/data/home/zyh/work/jaqmc/analyzing`，使用该项目原有虚拟环境。
`DISK_ANALYSIS_ROOT` 和 `DISK_ANALYSIS_PYTHON` 可指定其他分析程序目录及解释器。
波函数分析带 `--analysis-only`，不会再次开展物理量计算。

## 配置检查

下面的脚本向 `p1` 队列申请三个 CPU 核和4 GB内存，执行定向测试，
并解析四种默认评估配置。它不运行采样或 GPU 计算。

```bash
sbatch analyzing/checkDiskEvaluation.sh "$RUN_DIR"
```

## 输出与检查

输出包含 `requested_evaluation.yaml`、程序解析后的 `evaluation_config.yaml`、
`evaluation_digest.npz`、逐步统计、最终评估检查点，以及 `run_manifest.json`。
成功计算后的 `validation.json` 记录产物检查结果。
`analysis_outputs` 保存现有分析程序生成的表格、摘要和图片。

脚本核对评估参数与来源检查点一致、实际配置数为8192、网络参数没有更新。
密度二维和径向积分应接近六电子，当前检查容差为0.01。
对关联检查参考位置和实际累计配置次数，波函数检查网格形状。
局域能量方差不是均值标准误差，均值误差沿用现有分析程序另行处理。

已有输出目录会导致程序停止，可通过新的 `--save-path` 重新计算。
失败时保留输出与来源记录，便于检查原因。
本脚本不会自动将结果写入已有统一数据集。

## 从完整检查点恢复对关联结果

对关联的最终归一化在 CPU 上使用 NumPy 完成。无密度支持的网格保留为
`NaN`，有效网格仍按一体密度与二体密度定义归一化。采样阶段可以保持
`jax.debug_nans=true`，原始计数中的无效值会使汇总停止。

若全部采样已经完成，但汇总文件尚未生成，下面的正式作业申请三个 CPU 核和
一张 GPU。作业先在 CPU 上运行单设备和双设备测试，再从最后的检查点生成
对关联汇总文件并调用原有分析程序，不重新采样。随后在 GPU 上运行完整的波函数扫描，
结果保存在 `eval/wavefunction_scan_fixed`。

```bash
sbatch analyzing/repairDiskEvaluation.sh "$RUN_DIR"
```

程序核对来源文件校验值、实际配置数、完成步数、累计次数和网络参数，
仅在检查点已完成全部请求步数时恢复。
`run_manifest.before_recovery.json` 保留原始失败记录，当前说明文件记录恢复来源。
已有汇总文件或恢复记录不会被覆盖。

波函数扫描保留设备分布一致性检查。各设备使用共同选出的背景，
扫描结果在写回共享状态前按设备取平均，保证条件分支返回一致的设备分布。
失败扫描可以使用新的输出目录重新提交：

```bash
sbatch evaluate_disk.sh wave_scan "$RUN_DIR" --save-path "$RUN_DIR/eval/wavefunction_scan_fixed"
```
