# H24 2 Å RHF→DMRG 64 sweeps 与 MPS→JaQMC 蒸馏模块化实施方案

版本：2026-10-29；状态：**方案与部分基础模块已提交；服务器已验证 RHF→DMRG 源码尚未导入，H24 端到端未验证**。

## 1. 总结与不可破坏的约束

**将服务器实际完成 2 Å / 64 sweeps 的 RHF→DMRG Python 源码作为唯一数值基准；不要从旧版单文件重新推导或重写 DMRG。** Reference Producer、Teacher Builder 和 JaQMC Distiller 必须能独立运行、保存 manifest 和恢复。原 RHF、MP2 自然轨道、occupied/virtual PM 局域化、Fiedler 排序、block2 SU(2) 四态 SA-DMRG、M250×4 + M250×16 + M320×44 的 64 步调度、持久化和 restart 行为均保持原样。64 步预算完成与能量收敛是两个不同指标。

注意：目前 GitHub jaqmc/main 未包含服务器使用的 fixed64 源文件；此前上传的 20261028 版本与 fixed64_argfix 版本并不相同。**未经源文件 SHA256 对照，严禁声称完成“原版复制”或通过生产级回归。** 服务器上的最终脚本、对应 YAML、Slurm 文件、restart manifest、block2/JAX/PySCF 版本及成功日志须一起归档。若原始几千行脚本是唯一成功版本，第一阶段可以先完整保留为 `reference_producer/vendor/verified_server_fixed64.py`，经输出等价验证后再逐函数机械拆分，避免同时改动科学算法。

## 2. 最终文件架构及长度预算

```text
reference_producer/                      # 服务器 RHF→DMRG，独立 CPU 环境
  vendor/verified_server_fixed64.py      # 原版只读备份（临时过渡）
  geometry.py                            # ≤180 行；周期 H24/2 Å/bohr 转换
  rhf.py                                 # ≤260 行；PySCF reference
  orbitals.py                            # ≤350 行；MP2 NO / PM / Fiedler
  dmrg.py                                # ≤450 行；block2 SU2 多态求解
  restart.py                             # ≤220 行；固定预算和恢复
  artifacts.py                           # ≤220 行；MPS/轨道/manifest
  cli.py                                 # ≤150 行；prepare / resume / status
teacher_builder/                         # 独立 CPU 任务，按 root 导出
  su2_sz.py                              # ≤240 行
  selected_ci.py                         # ≤350 行
  spin_grouping.py                       # ≤180 行
  validation.py                          # ≤260 行
  manifest.py                            # ≤180 行
  cli.py                                 # ≤150 行
src/jaqmc/extensions/mps_distill/
  __init__.py
  schemas.py                             # Teacher 数据协议
  grouping.py                            # α/β 唯一占据和稀疏系数索引
  losses.py                              # 相对幅度、相位、fidelity
  teachers/selected_ci.py                # JAX 连续坐标 teacher（后续接入）
  teachers/cache.py                      # 离线标签读取（后续接入）
  dataset.py                             # 数据切分/刷新（后续接入）
  estimator.py                           # JaQMC FunctionEstimator（后续接入）
  trainer.py                             # 4 root 训练调度（后续接入）
  validation.py                          # overlap/ESS/held-out（后续接入）
  nes_bridge.py                          # checkpoint→NES（后续接入）
scripts/h24_refactor/                    # Slurm wrappers / source-freeze
tests/extensions/                        # 数值测试
```

行数是设计上限，不是通过牺牲可读性强制压缩。大文件按物理职责拆分；CLI 只做参数和调度，不能内嵌物理内核。

## 3. 两个阶段之间的契约

### Reference Producer 输出（不得以 CI 导出结果判定 DMRG 失败）
```text
reference/
  source_manifest.json      # source_sha256, config_sha256, dependencies
  orbitals/orbital_reference.npz
  dmrg/root_000/ ... root_003/   # block2 原始 MPS checkpoint
  dmrg/summary.json         # sweeps_completed, bond_schedule, energies, dw
```

必须包含体系 lattice、单位（Å 与 bohr 明确）、原子排列、PBC/twist/k-point、电子数和自旋、MO 系数与 AO 基、轨道排序置换、各 root 的 MPS checkpoint 路径、spin symmetry、能量、discarded weight、source hash、checkpoint hash。DMRG 状态 `completed_64` 与 `physically_converged` 独立。

### Teacher Builder 输出
```text
teacher/
  source_manifest.json
  root_000/selected_ci.npz
  root_000/teacher_manifest.json
  ... root_003/
  summary.json
```

每个 `selected_ci.npz` 存储唯一 alpha/beta 占据矩阵、alpha_index、beta_index、复数 coefficients。**不要误把 `sum(abs(c)^2)` 的局部归一化结果当作相对完整 MPS 的 retained weight**；记录 full-MPS normalization convention、候选集 norm²、K-budget norm²、归一化因子、SU2→SZ 误差和独立能量。对不同 cutoff 的候选集合必须以配置 hash / 占据集合 hash 去重，不能只凭 size 判断同一集合。

### Distiller 输出
```text
distill/
  root_000/trainckpt_*.npz
  ... root_003/
  distillation_manifest.json
  jaqmc_resolved.yml
```

每个 root 记录训练/验证 loss、fidelity、ESS、phase consistency、原始 Teacher SHA、坐标 convention 和训练 seed。训练完成 ≠ fit validated ≠ NES ready。

## 4. 生产主线：Selected CI→连续 Teacher→JaQMC

1. 保留服务器原始四态 SU(2) DMRG，固定 64 sweeps，不允许提前按能量阈值结束。原脚本曾发生 `sweeps_only` 包装器参数不匹配和 SU2→SZ MPO 导出 OOM；须以服务器实际修复后的版本为准。
2. 每个 root 在**独立进程**中 SU(2)→SZ；避免四态共享内存峰值。导出阶段失败只重试该 root，不重跑 DMRG。
3. 自适应 Selected CI：cutoff 网格可沿用 `1e-2` 至 `1e-8`；候选预算从 20k、40k、80k、160k 到 200k；分别记录全候选 `W_all`、预算内 `W_K` 和 `ΔE(K)`。CI 能量 MPO 验证单独作业，不能在所有 cutoff 循环中反复构建 MPO。200k 为当前配置，不是普适足够预算。
4. 将 `K` 个 determinant 转为 `Uα` 和 `Uβ` 个唯一自旋行列式以及索引 `(a_k,b_k,c_k)`。复杂度从约 `2K` 个 determinant 评估下降到 `Uα+Uβ`；实际收益必须测量 `Uα,Uβ`，不可预设。
5. 连续坐标 Teacher 采用 PySCF PBC AO→MO 和 `Σ c_k detΦα detΦβ`，使用 complex128、分块 logdet、数值稳定的复数缩放求和；特别测试节点附近、全零项、相位约定、轨道顺序、反对称性、周期性及 electron ordering。**严禁将零振幅或浮点下溢静默变为错误的有限 logpsi。**
6. H24 优先离线标签缓存：保存坐标 `[N,24,3]`、四态 complex logpsi、采样分布和 split；数据生成与 GPU 训练解耦。只缓存坐标和标签，不缓存每个样本所有 Slater 矩阵。加入 Student/Teacher 混合样本、刷新、重要性权重、ESS 检查。初始可先用原有在线 Teacher 小体系建立数值基线。
7. 使用 JaQMC 已有 `SolidWavefunction`、`VMCWorkStage`、`FunctionEstimator`、Adam 与 checkpoint 格式；**不要为离线数据伪装成在线 MCMC**，需单独 minibatch stage 或经过验证的批数据适配器。现有 `phase_logpsi` 返回 `(unit_phase, logabs)`，而 `logpsi` 返回 complex logpsi，必须区分。
8. 每 root 独立预训练并保存 `trainckpt_*.npz`，保留 NES 能读取的参数树与命名约定；进入四态 NES 前检查全部 root 的物理体系哈希、checkpoint、波函数结构和重叠矩阵。

## 5. 损失与训练策略

原 loss 为 `(log|ψS|-log|ψT|)^2 + λ(1-cos(θS-θT))`，对全局尺度与相位不变性不足。推荐主线先用去除每 root 全局常数的 log-amplitude 方差和 `1-|mean(exp(i Δθ))|`；然后逐步混入归一化 overlap / infidelity 目标。Teacher 分布采样下：
```math
F = |E_T[ψ_S/ψ_T]|² / E_T[|ψ_S/ψ_T|²].
```
只有样本确实来自 `|ψT|²` 时可直接使用；混合分布必须带重要性权重，低 ESS 时不能把该估计作为可靠质量门槛。多设备求均值必须全局归约，不能只平均各 GPU 上的局部 fidelity。

建议训练阶段：A 原损失基线；B 去规范 centered loss；C overlap/fidelity fine-tune；D Teacher+Student 混合采样和在线刷新。相同网络、batch、计算预算、随机种子下比较；原配置 `ndets:1` 可作为基线，并与 `ndets:4/8` 做容量消融。不要将提高训练迭代数误认成蒸馏算法提升。

## 6. 候选方案优先级

| 级别 | Teacher / 目标 | 优势 | 风险 | 结论 |
|---|---|---|---|---|
| P0 | 原逐项 Selected CI + 原损失 | 复现基线 | K×两次 det，高成本 | 保留比较 |
| P1 | 自旋分组 Selected CI + centered loss + cache | 可落地、复用 block2/JaQMC | CI 截断 | 主线 |
| P2 | 相同 Teacher + overlap loss | 目标更物理 | ESS / 节点不稳定 | 对照 |
| P3 | MPS 采样 / 自适应 CI 扩充 | 提高 W_K | 200k 预算可能不足 | CPU 研究 |
| P4 | 稀疏 CI 系数矩阵低秩近似 | 压缩收缩成本 | 额外截断误差 | 测 rank 后决定 |
| P5 | CI+RDM/observable/subspace 正则 | 更贴近 NES 物理量 | 优化目标竞争 | 第二阶段 |
| P6 | 直接 MPS→连续坐标算符收缩 | 绕开 CI 截断 | 复杂且昂贵 | H4/H6 原型 |

**43% retained weight** 若确实指原始归一化 MPS 的投影范数²，则完美拟合归一化截断 CI 的原 MPS squared fidelity 上限为 0.43；这不是 NES 优化后物理精度的上限。先测四态 `W_K(K)` 与能量误差，不得把 43% 当成 Student 训练问题。

## 7. 验证矩阵与通过条件

| 级别 | 测试 | 判据 |
|---|---|---|
| T0 | 源文件归档 | server script/YAML/Slurm SHA256 与 manifest 完整 |
| T1 | RHF→DMRG 原版复制 | H24 2 Å 四态、轨道排序、64 sweep、checkpoint 与原版一致；容忍度需预先定义 |
| T2 | H4/H6 FCI | SU2→SZ / CI 占据符号、归一化、反对称性、周期相位正确 |
| T3 | 自旋分组 | 对随机复数系数与占据，逐项和分组 CI 复振幅一致（float64 `rtol=1e-10`，节点附近用绝对误差） |
| T4 | Teacher 验证 | 各 root `W_K`、`ΔE(K)`、有效相位、有限值比例、单点时间与显存 |
| T5 | 蒸馏 | held-out fidelity、ESS、相位、训练/验证曲线、节点附近测试 |
| T6 | 四态 | `S_ij=<ψ_i|ψ_j>`、root 对应、参数结构、四态能量及 NES 重启 |
| T7 | HPC | CPU 导出峰值内存、OOM 后按 root 恢复、GPU JIT/吞吐、Slurm 依赖 |

H24 生产门槛必须先明确 `W_K`、`ΔE`、overlap 和 ESS 的定量阈值；不允许只因 `train.run.iterations` 完成就自动通过。

## 8. 部署与迁移顺序

**Phase 0：服务器冻结**。从服务器复制真正成功的 fixed64 Python、YAML、Slurm、日志和 checkpoint manifest；对源文件生成 SHA256；将原文件原封不动归档到 `reference_producer/vendor`。若无法取得，停止 Reference Producer 的“已验证”标记。

**Phase 1：无数值改动拆分**。将原脚本按 `geometry/rhf/orbitals/dmrg/restart/artifacts/cli` 提取，先用 wrapper 保留调用和所有默认值；对 H4/H6 做回归，然后在相同服务器条件下比对 H24 2 Å 的参考结果。绝不把 2 Å 与 `2.0 bohr` 混淆（2 Å≈3.77945 bohr）。

**Phase 2：Teacher 导出**。每 root 独立 SU2→SZ、Selected CI；生成 `W_K` 和 unique spin determinant 统计；分离 MPO 验证。

**Phase 3：蒸馏主线**。实现连续 AO/MO evaluator、离线缓存和新 loss；H4/H6 小体系验证，H12 吞吐基准，H24 四态训练。

**Phase 4：NES 接口**。保持原 JaQMC `trainckpt_*.npz` 与参数树；检查四态 checkpoint 与 lattice/单位/root/phase；进入四态 NES 与 DMD。

建议 Slurm DAG：`reference` → 四个 `export_root` array job → `teacher_validate` → `label_cache` → 四个 `distill_root` → `subspace_validate` → `NES`。每个任务独立 manifest、原子写入、失败重试，禁止在导出失败时重新跑 RHF/DMRG。

## 9. 本次仓库变更的真实范围

本分支新增 Markdown 设计、`mps_distill` 的可测试基础模块、单元测试和服务器源文件 SHA256 冻结工具。**没有拿到服务器已验证的 RHF→DMRG 原文件，也没有声称已完成它的逐函数拆分。** 连续坐标 Teacher 的 PBC AO→MO 和完整训练适配器仍须以原始 2 Å 源码和运行环境为依据迁移，随后进行端到端验证。
