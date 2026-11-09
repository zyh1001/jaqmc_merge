# JaQMC 多几何融合说明

## 仓库与来源

融合仓库为 `/data/home/zyh/work/jaqmc_merge`，本地分支为
`integration/hall-multigeometry`。

| 来源 | 提交 |
| --- | --- |
| 上游 `bytedance/jaqmc` 的 `main` | `d5f50497e2601dce4b0cde9eb96aa3a42281414b` |
| `Usercyk/jaqmc` 的 `hall/torus` | `8c418a4c4835c1e185e5bd4d3afcb46e168a97ba` |
| 本地 `/data/home/zyh/work/jaqmc` | `204264eb43920be8e6363337871122ed3ee82705`，同时读取当前未提交的源码 |

本地迁移范围只有 `src`。来源文件清单、原始文件摘要及原仓库状态保存在
`integration_sources.json`。数值对照使用的三个来源压缩包位于
`tests/app/hall/reference/`，对应路径与摘要记录在来源清单中。
计算节点直接读取这些快照，不需要安装 Git。
整个操作不向远程仓库推送。

## 实现内容

球面沿用上游波函数、动能、势能、角动量评估和训练默认配置。
配置和势能模块改为同名包，原有导入入口继续有效。

环面采用参考分支的两种波函数、周期特征、采样器、动能、周期库仑势、密度、
对关联和一体约化密度矩阵。二维周期求和直接复用上游 `EwaldSum2D`。
默认波函数支持一般环面参数；另一种使用椭圆函数的波函数保留原有的矩形环面限制
及弃用提示。其已验证的几何是正方形环面。

为保留环面系数的原始初始化方式，公共 `SplitChannelDense` 增加可选的
`kernel_init` 参数。其默认初始化和参数路径保持上游行为，并准备了与上游原函数
使用同一参数树的输出对照测试。

圆盘迁入本地神经网络、拉夫林态和全部符号波函数，并保留复数振幅、包络模式和
原参数名称。动能与圆盘背景势独立保留。背景势仍不计常数背景自能，电子运动没有
新增的硬边界。

新增观测量包括圆盘密度、对关联、一体约化密度矩阵、结构因子、边缘格林函数、
条件波函数扫描、角动量惩罚、波函数重叠和短距离能量统计。
短距离统计已接入配置及评估流程，其相互作用强度与系统配置保持一致。
边缘格林函数继续采用固定第一个电子的条件采样。

环面参考分支中尚未接入的圆盘初始化和工作流分支由本地实现补齐。
抽象基类中的未实现方法仍保持抽象定义。

## 接口与兼容

| 用途 | 入口 |
| --- | --- |
| 球面默认配置 | 不需要指定几何，原命令继续使用 |
| 圆盘配置 | `system.module=jaqmc.app.hall.config:HallDiskConfig` |
| 环面配置 | `system.module=jaqmc.app.hall.config:HallTorusConfig` |
| 圆盘重叠评估 | `jaqmc hall overlap` |
| 环面检查点重叠比较 | 普通评估中设置 `estimators.enabled.fubini=true` |
| 圆盘短距离统计 | `estimators.enabled.short_range=true` |

圆盘重叠评估的当前参数从 `workflow.source_path` 读取，参考参数从
`workflow.reference_path` 读取。两者必须使用相同波函数结构。
省略参考路径即计算自重叠。它使用原有的对称重叠采样分布，独立生成
`overlap_digest.npz`。

环面重叠评估仍采用参考分支的检查点比较算法，不替换圆盘算法。
具体调用示例位于 `docs/systems/hall/geometries.md`。

圆盘及环面迁入的直方图、结构因子和扫描状态已适配上游当前的设备间汇总方式，
保留物理公式、结果名称及归一化定义。使用旧训练检查点进行评估时，应建立新的
评估输出目录，不恢复旧版本已经累积的观测量状态。
早于 `amplitude_imag` 参数的旧圆盘检查点不会自动补参数。

## 测试环境与提交

本仓库已建立独立的 `.venv`，使用上游锁定依赖和 CUDA 12 可选依赖。
需要重建环境时执行下列准备脚本。它安装环境、建立日志目录并记录当前版本，
不运行数值计算。

```bash
cd /data/home/zyh/work/jaqmc_merge
bash scripts/integration_tests/setup.sh
```

测试脚本默认申请 `v100` 分区、单节点、一个任务、三个处理器核心、一张显卡，
时限一小时，内存采用分区默认值。分区及资源配置来自服务器配置与现有脚本，
没有核实当前空闲资源。

提交前在登录节点记录当前提交和工作区改动。该步骤需要登录节点上的 Git；
计算节点只读取生成的版本记录。修改代码后重新提交时，也应更新这份记录。
推荐分别提交，便于查看每项结果和重新运行：

```bash
cd /data/home/zyh/work/jaqmc_merge
.venv/bin/python scripts/integration_tests/prepare.py
sbatch scripts/integration_tests/run.sh sphere
sbatch scripts/integration_tests/run.sh torus
sbatch scripts/integration_tests/run.sh disk
sbatch scripts/integration_tests/run.sh estimators
sbatch scripts/integration_tests/run.sh other
sbatch scripts/integration_tests/run.sh parallel
```

也可以通过 `sbatch scripts/integration_tests/run.sh all` 顺序运行全部测试。
如果集群编译耗时较长，建议分项提交，或由用户调整提交命令中的时限。
脚本不能直接在登录节点执行。

| 测试参数 | 主要内容 | 当前状态 |
| --- | --- | --- |
| `sphere` | 原配置、短训练与恢复评估、上游球面波函数及哈密顿量测试 | 待用户运行 |
| `torus` | 两种波函数、独立能量、磁平移边界、短训练与恢复评估、检查点重叠比较 | 待用户运行 |
| `disk` | 全部圆盘波函数与来源对照、导数、独立能量、短训练及恢复评估 | 待用户运行 |
| `estimators` | 观测量配置和实际计算、原始模块对照、分箱求和与重叠计算 | 待用户运行 |
| `other` | 上游小分子短运行、固体工作流以及配置解析和检查点测试 | 待用户运行 |
| `parallel` | 在计算节点上使用两个处理器虚拟设备，检查跨设备计数及扫描选择 | 待用户运行 |

新编写的短运行采用 32 个采样点和两次迭代。限定六电子的符号波函数仍使用六电子。
上游及来源分支的已有测试保留自身参数与容差。
新增来源对照采用相同坐标、同一参数树和同一当前运行环境。
波函数振幅与相位的比较容差为 `2e-6`，导数及能量通常为 `2e-5`。
相位通过复指数比较，避免对数相位相差整周造成误判。
这些对照验证迁入函数在当前公共框架中的行为，不代替历史训练环境的完整对照。

## 结果与验收

Slurm 原始日志为 `integration_test_results/slurm-<作业编号>.out` 和 `.err`。
每次测试的详细日志、测试报告及数值输出保存在
`integration_test_results/<作业编号>-<测试参数>/`。

日志包含实际目录、准备时记录的提交与工作区改动、运行环境、设备、主要配置、
波函数与能量模块类型、
关键数值和测试结论。所有必要测试成功时，脚本以零退出码结束并打印 `PASS`。
任何必要测试失败时，返回非零退出码并打印 `FAIL`。

验收时检查局域能量及方差是否有限、计数是否等于实际采样数量、短距离分箱之和
是否等于总和，以及自重叠是否为一、重叠距离是否为零。
圆盘对关联在没有密度支持的网格上保留未定义值，测试只对有定义的网格要求有限。

静态检查结果记录在 `integration_validation.json`。
用户报告的并行测试作业 `3842921` 在测试开始前因计算节点找不到 Git 退出。
相关调用已移至登录节点的准备步骤，修正后的测试待用户重新提交。
本助手未提交作业，也未在登录节点执行训练、采样或能量测试。
目前不能声称数值结果已经等价或集群运行已经通过。


## 三几何接口与短训练检查

配置统一在单个 `config.py` 中，通过 `HallSphereConfig`、`HallDiskConfig`、
`HallTorusConfig` 选择几何。默认采用球面。旧的 `config.disk` 等子模块入口
需要替换为 `jaqmc.app.hall.config:HallDiskConfig` 等显式入口。

新增脚本 `scripts/integration_tests/hall_structure.sh` 支持 `--mode dry-run`、
`--mode train` 和 `--mode all`，覆盖球面、圆盘、方形环面及六角晶格环面。
默认均为两个电子、三分之一填充对应配置、相互作用强度 1、100 步训练。

```bash
sbatch scripts/integration_tests/hall_structure.sh --mode dry-run
sbatch scripts/integration_tests/hall_structure.sh --mode train
sbatch scripts/integration_tests/hall_structure.sh --mode all
```

指定几何及覆盖配置：

```bash
sbatch scripts/integration_tests/hall_structure.sh --mode all --geometry torus-hex workflow.batch_size=64
```

脚本会检查训练更新、数值有限性、检查点保存及恢复评估。结果按作业编号写入
`integration_test_results`，具体入口替换表和结果说明见
`docs/systems/hall/structure-tests.md`。

本次仅完成静态检查，没有提交作业。配置试运行、训练、恢复评估和数值回归
均待用户提交运行。
