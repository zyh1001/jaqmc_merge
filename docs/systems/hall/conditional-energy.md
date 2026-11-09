# 固定五电子背景下的局域能量

本程序读取合并数据所关联的六电子训练检查点。固定五个电子的位置，让剩余电子围绕一个固定电子移动，计算完整六电子局域能量。动能仍对全部电子坐标求导，固定位置仅限定所考察的构型。

## 输入检查与作业提交

在项目根目录检查三种几何的数据来源、文件校验值和扫描距离范围：

```bash
bash analyzing/evaluateConditionalEnergy.sh --geometry all --dry-run
```

默认数据目录是 `/data/home/zyh/work/data/merge`。程序从 `unified.h5` 读取训练配置，通过 `manifest.json` 的来源文件编号定位原始训练检查点。清单中的相对路径以 `source_root` 为基准。合并数据仅作为输入，扫描结果另行保存。

也可以直接指定训练目录中的配置和检查点，例如读取六电子圆盘模型：

```bash
bash analyzing/evaluateConditionalEnergy.sh --geometry disk \
  --config /data/home/zyh/work/data/disk/n6_Q18_kapa1_8192_final/train_config.yaml \
  --checkpoint /data/home/zyh/work/data/disk/n6_Q18_kapa1_8192_final/train_ckpt_149999.npz \
  --dry-run
```

`--config` 和 `--checkpoint` 必须一起提供，并明确选择一种几何。此时程序读取该训练配置和检查点，不使用合并文件中的模型参数或固定背景。计算仍通过同一个提交脚本运行，去掉 `--dry-run` 后可设置所需的扫描分辨率。

输入检查通过后，先提交一次参数恢复与单点能量检查：

```bash
sbatch analyzing/evaluateConditionalEnergy.sh --geometry all --validate-only
```

检查通过后，可以提交低分辨率径向计算：

```bash
sbatch analyzing/evaluateConditionalEnergy.sh --geometry all --radial-only --radial-bins 8 --angles 8 --batch-size 1
```

正式计算默认增加方向数复核曲线，并为圆盘计算完整条件积分：

```bash
sbatch analyzing/evaluateConditionalEnergy.sh --geometry all
```

计算脚本默认使用 `v100` 的一张显卡。可以在脚本名前指定 `--partition=v100g32`。所有程序参数通过脚本原样传入。每次作业默认写入 `runs/conditional_energy_作业编号`，其中按 `disk`、`sphere`、`torus` 分目录保存。也可以显式指定 `--output-root`，已有非空结果目录不能覆盖。

## 背景与坐标

默认读取训练检查点的第零个保存构型，固定电子一至五，让电子零围绕电子一移动。构型编号由 `--walker-index` 指定，移动电子与参考电子由 `--mobile-electron`、`--probe-electron` 指定。

圆盘可以复用合并数据中已有的扫描背景：

```bash
sbatch analyzing/evaluateConditionalEnergy.sh --geometry disk --merged-background
```

`--background-manifest` 可以读取 JSON 背景清单，既支持包含 `fixed_electrons` 的对象，也支持计算结果中的 `run_manifest.json`。前者的五个位置按除移动电子外的电子编号排列。后者保存完整六电子构型，程序忽略其中移动电子的位置。

背景坐标必须使用对应几何的原生定义。圆盘是平面坐标，球面是两个球面角，圆环面是晶胞分数坐标。清单可同时保存 `mobile_electron`、`probe_electron` 和来源检查点校验值，程序会检查其一致性。

所有距离均以磁长为单位。圆盘使用平面距离，球面使用弦长，圆环面使用最短周期距离。圆环面的扫描半径必须小于最短非零周期长度的一半，球面扫描半径必须小于直径。

程序只允许两项旧检查点兼容：圆盘缺少的轨道虚部设为零，圆环面新增质心特征的输入权重设为零并保留原自旋权重位置。其他参数差异直接报错。计算采用双精度，但将原单精度参数转换为双精度不会恢复已丢失的有效数字。

## 输出与绘图

计算保存逐点坐标、波函数对数、复数局域总能量、动能和势能。圆盘额外保存电子间势能和背景项。非有限值会保留在结果文件中并记录错误位置，该次扫描不标记为完成。

径向曲线采用概率加权方向平均：

$$
\mu(r)=\frac{\sum_\theta |\Psi(r,\theta)|^2\operatorname{Re}E_{\mathrm{loc}}(r,\theta)}{\sum_\theta |\Psi(r,\theta)|^2}.
$$

`radial_grid.h5` 和 `radial_refined_grid.h5` 保存两个方向网格的逐点结果，其中 `electrons` 保留全部六电子坐标，`radius` 和 `theta` 标明扫描位置。`curves.npz` 保存初始方向网格的加权平均，`curves_refined.npz` 保存方向数加倍后的平均，绘图使用后者。

`run_manifest.json` 记录输入文件校验值、来源检查点、构型编号、固定背景、配置、参数兼容前后的结构、原始精度及计算设置。`validation.json` 记录参数兼容检查、能量恒等式、结果文件校验值和方向加密检查。方向加密检查不是蒙特卡洛标准误差。小网格未通过加密检查时，结果可以用于核对计算流程，但需要提高分辨率后才能判断曲线是否收敛。

圆盘完整积分另外保存 `base_grid.h5`、`refined_grid.h5` 和 `tail_grid.h5`，对应原网格、加密网格和延长的尾部区域。`base.npz`、`refined.npz` 和 `integrals.npz` 保存相应积分结果。最后一个文件使用加密网格与尾部的共同归一化。

计算完成后，在终端把以下示例中的 `123456` 替换为实际计算作业编号，再提交绘图：

```bash
sbatch analyzing/reportConditionalEnergy.sh --root runs/conditional_energy_123456
```

绘图使用 `p1`，默认输出包含作业编号和时间戳的报告目录。当前计算环境没有绘图库，绘图脚本默认使用同工作区中已安装绘图库的 `jaqmc/.venv/bin/python`，并始终加载本项目的分析脚本。可通过环境变量 `PLOT_PYTHON` 指定其他已安装绘图库的解释器。

每种几何分别输出中文报告、逐距离表格，以及以下四个区间的独立 PNG 和 PDF：

- $10^{-3}$ 至 $10^{-2}$，对数横轴。
- $10^{-2}$ 至 $10^{-1}$，对数横轴。
- $10^{-1}$ 至 $1$，对数横轴。
- $1$ 至 $2$，线性横轴。

分段只改变显示范围。相同几何、体系参数和固定背景的多组模型结果可以通过 `--root` 后列出多个目录一起绘图。不同几何分别生成报告。

圆盘完整积分结果中的概率和累计能量贡献采用全计算区域归一化。累计末端是此固定背景下的条件平均能量，不能替代全体系平均能量。球面和圆环面的径向结果不包含完整空间积分。局域能量的大值需要结合概率权重解释，不能仅凭曲线幅值判断平均能量优劣。
