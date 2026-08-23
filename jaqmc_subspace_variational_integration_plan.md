# JaQMC 子空间变分移植与实现方案

> 目标：以当前开源 JaQMC 的连续电子 NNQMC 架构为主体，参考 NeuralQXLab `variational-subspace-methods` 中 determinant-state / Rayleigh-matrix / determinant-sampler 的已验证实现思路，在 **不引入 NetKet 运行时依赖、不破坏 JaQMC 现有 wavefunction / sampler / estimator / workflow 分层** 的前提下，加入可用于低能多态联合优化的子空间变分功能，并为后续 RDMD 与 Bridge 共用同一套子空间基础设施。
>
> 本文以 2026-08-14 可见的 JaQMC `main` 分支与 NeuralQXLab `variational-subspace-methods` `main` 分支为参考。文中给出的类名与文件名属于推荐实现方案，除特别说明外并非 JaQMC 当前已存在接口。

---

## 1. 设计目标

实现的第一目标不是“把 NetKet/Bridge 代码复制进 JaQMC”，而是把论文中的核心算法重新表达成 JaQMC 原生组件：

$$
\mathcal V_m
=
\operatorname{span}
\left\{
|\phi_{\theta_0}\rangle,\ldots,|\phi_{\theta_{m-1}}\rangle
\right\}
$$

通过 determinant-state 映射

$$
|\Phi_A\rangle
=
|\phi_{\theta_0}\rangle\wedge\cdots\wedge|\phi_{\theta_{m-1}}\rangle
$$

并最小化

$$
E_{\mathrm{sub}}
=
\mathrm{Tr}\!\left(G^{-1}G^{(H)}\right),
$$

其中

$$
G_{ij}
=
\langle\phi_i|\phi_j\rangle,
\qquad
G^{(H)}_{ij}
=
\langle\phi_i|H|\phi_j\rangle.
$$

根据 Ky Fan 变分原理，最优 \(m\) 维子空间为最低 \(m\) 个本征态张成的空间。实际 VMC 使用 determinant-state 分布采样：

$$
p(\mathcal R)
\propto
\left|\det\Phi(\mathcal R)\right|^2,
$$

其中

$$
\Phi_{rs}
=
\phi_s(\mathbf R_r).
$$

对应 local Rayleigh matrix：

$$
R_L(\mathcal R)
=
\Phi^{-1}\Phi^{(H)},
$$

并且

$$
E_L^{\mathrm{sub}}
=
\mathrm{Re}\,\mathrm{Tr}\,R_L.
$$

最终目标是使 JaQMC 获得：

1. `m=1` 时严格退化回普通 ground-state VMC；
2. `m=2,4,8,...` 时可联合优化低能子空间；
3. 保留 FermiNet / Psiformer / LapNet 等现有物理波函数实现；
4. 复用 JaQMC 现有 local-energy、MCMC、JAX、multi-device、optimizer 与 workflow 基础设施；
5. 为 RDMD 输出 Ritz energies、Ritz vectors、1/2-RDM 或一般 operator matrix；
6. 后续 Bridge 只作为另一个 workflow 复用 Rayleigh-matrix 基础层，而不与静态子空间训练耦合。

---

## 2. 当前两套代码中应复用的部分

### 2.1 JaQMC 应继续负责的内容

JaQMC 当前源码结构的核心分层为：

```text
src/jaqmc/
├── app/
├── estimator/
├── geometry/
├── laplacian/
├── optimizer/
├── sampler/
├── utils/
├── wavefunction/
├── workflow/
├── writer/
├── array_types.py
└── data.py
```

推荐继续让 JaQMC 独占以下职责：

- 真实空间电子坐标与 system `Data`；
- FermiNet / Psiformer / LapNet；
- 分子、晶体、PBC、Ewald、ECP 等电子结构逻辑；
- 单态 local kinetic / potential / total energy；
- JAX `jit/vmap` 与多 GPU walker sharding；
- MCMC sampler 基础设施；
- `Estimator` / `PerWalkerEstimator`；
- `LossAndGrad`；
- Optax / SR / KFAC；
- `VMCWorkflow` / `VMCWorkStage`；
- checkpoint / writer / config / CLI。

**原则：子空间层不得重新实现电子 Hamiltonian。**

### 2.2 NeuralQXLab 代码中应借鉴的核心思想

参考：

```text
libraries/bridge/bridge/_src/
├── bridge.py
├── bridge_tools.py
├── determinant_sampler.py
├── determinant_state.py
├── models.py
├── numpy_tools.py
└── sum_of_states.py
```

真正应迁移的是算法，而不是 NetKet 类型：

- `DeterminantModel` 的 \(m\times m\) amplitude matrix 构造；
- state axis 参数矢量化；
- determinant-state log amplitude；
- determinant-state MCMC；
- 每次只更新一个 Hilbert-space copy；
- amplitude matrix / inverse / logdet cache；
- Sherman-Morrison 与 matrix determinant lemma 快速更新；
- sampled Rayleigh estimator；
- `solve(Φ, ΦH)` 而不是显式 `inv(Φ) @ ΦH`；
- SOS estimator 仅作为辅助/验证方案。

**不要迁移：**

- `nk.vqs.MCState`；
- `nk.hilbert.TensorHilbert`；
- `operator.get_conn_padded()`；
- NetKet 的离散 spin Hilbert 假设；
- NetKet sampler 生命周期；
- 与 JaQMC 已有功能重复的 config/writer/optimizer。

---

## 3. 推荐最终架构

不要建立一个平行的 `jaqmc/subspace/` 大目录复制所有 JaQMC 层级。推荐把子空间功能按现有责任边界分散到原模块：

```text
src/jaqmc/
├── estimator/
│   ├── base.py
│   ├── loss_grad.py
│   ├── ...
│   ├── rayleigh.py                  # 新增
│   └── operator_rayleigh.py         # 第二阶段新增，可合并到 rayleigh.py
│
├── sampler/
│   ├── base.py
│   ├── mcmc.py
│   └── determinant.py               # 新增
│
├── wavefunction/
│   ├── base.py
│   ├── ...
│   └── determinant_state.py         # 新增
│
├── workflow/
│   ├── vmc.py
│   ├── ...
│   ├── subspace_vmc.py              # 新增
│   └── bridge.py                     # 后续新增，不在第一阶段实现
│
├── utils/
│   └── subspace_linalg.py           # 新增，纯线性代数
│
└── data.py                           # 尽量不改或仅小幅扩展
```

测试目录对应增加：

```text
tests/
├── estimator/
│   └── test_rayleigh.py
├── sampler/
│   └── test_determinant_sampler.py
├── wavefunction/
│   └── test_determinant_state.py
├── workflow/
│   └── test_subspace_vmc.py
└── utils/
    └── test_subspace_linalg.py
```

---

## 4. 数据模型：replica axis 绝不能当成电子 axis

### 4.1 普通 JaQMC

普通电子 walker：
$$

\mathbf R
=
(\mathbf r_1,\ldots,\mathbf r_{N_e}).

$$
典型 batched shape：

```text
electrons: [B, Ne, 3]
```

其中：

- `B`：walker batch；
- `Ne`：物理电子数。

### 4.2 子空间 determinant walker

一个 determinant-state walker 包含 \(m\) 个完整物理构型：

$$
\mathcal R
=
(\mathbf R_0,\ldots,\mathbf R_{m-1}).
$$

推荐 shape：

```text
electrons: [B, M, Ne, 3]
```

其中：

- `B`：Monte Carlo walker；
- `M`：子空间 replica/state-copy 维；
- `Ne`：一个真实物理体系的电子数。

**严禁将其展平成 `M*Ne` 个真实电子。**

否则 JaQMC 的 Coulomb/Ewald 逻辑会错误加入不同 replica 间的：

$$
\frac{1}{|\mathbf r_i^{(r)}-\mathbf r_j^{(s)}|},
\qquad r\neq s
$$

虚假相互作用。

正确扩展 Hamiltonian 为：

$$
\mathbb H
=
\sum_{r=0}^{M-1}
H^{(r)},
$$

不同 replica 之间不相互作用。

---

## 5. 尽量保持 `Data` / `BatchedData` 不变

当前 JaQMC `BatchedData` 的核心假设是：

> 指定字段具有额外的 leading walker axis，并且只沿该 leading axis 做 `vmap` / sharding。

因此 `[B, M, Ne, 3]` 可以自然工作：

- `BatchedData` 仍只认识 `B`；
- `M` 只是单 walker 内部 shape；
- 多 GPU 默认仍沿 walker axis 分布；
- state/replica axis 在每张卡完整保留。

建议仅新增一个轻量描述对象：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class SubspaceSpec:
    n_states: int
    replica_fields: tuple[str, ...] = ("electrons",)
```

以及纯 helper：

```python
def take_replica(data: Data, replica_index: int, spec: SubspaceSpec) -> Data:
    ...
```

其行为：

```text
input:
    electrons [M, Ne, 3]

output:
    electrons [Ne, 3]
```

base wavefunction 永远只收到普通物理 `Data`，因此 FermiNet/Psiformer 不需要理解子空间概念。

---

## 6. `DeterminantStateWavefunction`

建议新增：

```text
src/jaqmc/wavefunction/determinant_state.py
```

### 6.1 责任

只做以下工作：

1. 持有一个 JaQMC `base_wavefunction`；
2. 管理 \(M\) 套参数；
3. 计算 cross log-amplitude matrix；
4. 计算 determinant-state `logpsi`；
5. 提供单行 amplitude evaluation，供 sampler 快速更新；
6. 不计算 Hamiltonian；
7. 不负责 MCMC；
8. 不负责 Rayleigh matrix。

### 6.2 推荐接口

```python
class DeterminantStateWavefunction:
    base_wavefunction: WavefunctionLike
    spec: SubspaceSpec

    def component_logpsi_matrix(
        self,
        params,
        replica_data: Data,
    ) -> jax.Array:
        """
        Return L[r, s] = log psi_s(R_r).
        Shape: [M, M]
        """

    def component_amplitude_row(
        self,
        params,
        physical_data: Data,
    ) -> jax.Array:
        """
        Return psi_s(R) for all s.
        Shape: [M]
        """

    def logpsi(
        self,
        params,
        replica_data: Data,
    ) -> jax.Array:
        """
        Return log det Phi for one determinant walker.
        """
```

### 6.3 参数树

推荐把 \(M\) 个 state parameters 组织为一个带 state axis 的 PyTree：

```text
params:
  ...leaf shape [M, ...]
```

不要使用 Python `list[Params]` 进入 JIT 主路径。

初始化可采用：

```python
params_list = [
    base_wf.init_params(data0, rng_i)
    for i in range(M)
]
stacked_params = jax.tree.map(
    lambda *xs: jnp.stack(xs, axis=0),
    *params_list,
)
```

后续 state evaluation：

```python
jax.vmap(base_wf.evaluate, in_axes=(0, None))
```

如果 data 也矢量化，则形成两层映射：

```text
replica r × state s
```

---

## 7. 连续电子体系下必须加入 amplitude rescaling

NeuralQXLab 参考代码适用于其离散网络时可以直接将 log amplitude exponentiate，但连续电子 NNQMC 中：

$$
\log |\Psi|
$$

动态范围可能很大。

不能简单：

```python
phi = jnp.exp(logpsi_matrix)
```

然后直接 `det(phi)`。

建议对每个 replica row 做缩放：

$$
a_r
=
\max_s \operatorname{Re} L_{rs},
$$
$$
A_{rs}
=
\exp(L_{rs}-a_r).
$$

则：

$$
\Phi = D A,
\qquad
D=\operatorname{diag}(e^{a_r}),
$$

且

$$
\log\det\Phi
=
\sum_r a_r
+
\log\det A.
$$

实现建议：

```python
row_shift = jax.lax.stop_gradient(
    jnp.max(jnp.real(logpsi_matrix), axis=-1)
)

scaled = jnp.exp(logpsi_matrix - row_shift[:, None])
sign, logabsdet = jnp.linalg.slogdet(scaled)

logdet = (
    jnp.sum(row_shift)
    + jnp.log(sign)
    + logabsdet
)
```

对 complex determinant 需确认 JAX `slogdet` 返回格式；也可以写统一 complex logdet helper。

同样的 row scaling 可用于 Rayleigh：

$$
(D^{-1}\Phi)^{-1}
(D^{-1}\Phi^{(H)})
=
\Phi^{-1}\Phi^{(H)},
$$

所以数值缩放不改变 local Rayleigh matrix。

---

## 8. `RayleighMatrixEstimator`

新增：

```text
src/jaqmc/estimator/rayleigh.py
```

建议继承当前：

```python
PerWalkerEstimator
```

原因：

- 每个 JaQMC walker 对应一个 determinant configuration；
- 单 walker 输出天然是一个 `[M, M]` matrix；
- JaQMC 已有 `chunked_vmap`；
- `vmap_chunk_size` 可控制 walker 维显存。

### 8.1 核心公式

构造：

$$
\Phi_{rs}
=
\psi_s(\mathbf R_r).
$$

对每一对 \((r,s)\) 调用已有 JaQMC physical local-energy：

$$
E_{L,s}(\mathbf R_r)
=
\frac{H\psi_s(\mathbf R_r)}
{\psi_s(\mathbf R_r)}.
$$
因此：

$$
\Phi^{(H)}_{rs}
=
\Phi_{rs}
E_{L,s}(\mathbf R_r).
$$

最后：

$$
R_L
=
\operatorname{solve}(\Phi,\Phi^{(H)}).
$$

不要写：

```python
jnp.linalg.inv(phi) @ phi_h
```

必须写：

```python
jnp.linalg.solve(phi, phi_h)
```

### 8.2 推荐接口

```python
@configurable_dataclass
class RayleighMatrixEstimator(PerWalkerEstimator):
    f_component_amplitudes: ...
    f_cross_local_energy: ...
    matrix_dtype: str = "complex128"
    pair_chunk_size: int | None = None
    condition_warning: float = 1e10

    def evaluate_single_walker(...):
        ...
```

返回至少：

```python
{
    "local_rayleigh": rayleigh,
    "subspace_energy": jnp.real(jnp.trace(rayleigh)),
    "subspace_energy_imag": jnp.imag(jnp.trace(rayleigh)),
    "rayleigh_solve_residual": residual,
}
```

evaluation 模式额外输出：

```text
rayleigh_mean
ritz_energies
ritz_vectors
max_ritz_imag
```

---

## 9. 不要重写 Hamiltonian：建立 `CrossLocalEnergyEvaluator`

这是整个移植能否保持干净架构的关键。

`rayleigh.py` 不应该 import：

```text
EuclideanKinetic
Ewald
ECP
TotalEnergy
...
```

建议引入一个轻量协议：

```python
class CrossLocalEnergyEvaluate(Protocol):
    def __call__(
        self,
        state_params,
        physical_data,
    ) -> jax.Array:
        ...
```

或者直接使用 callable type alias：

```python
LocalEnergyEvaluate = Callable[[Params, Data], jax.Array]
```

`SubspaceVMCWorkflow` 在装配阶段负责提供系统对应的：

```python
local_energy_fn(params_s, data_r)
```

然后 Rayleigh estimator 只关心数学关系：

$$
\Phi^{(H)}=\Phi\odot E_L.
$$

这样未来 JaQMC 修改：

- kinetic backend；
- Laplacian；
- Ewald；
- ECP；
- PBC；
- molecule/solid system；

子空间代码无需同步复制修改。

---

## 10. `M^2` cross local-energy 是主要性能热点

对一个 determinant walker：

$$
r=0,\ldots,M-1,
\qquad
s=0,\ldots,M-1.
$$

需要：

$$
M^2
$$

个：

$$
E_{L,s}(\mathbf R_r).
$$

对于 real-space neural QMC，local kinetic energy 需要电子坐标梯度/Laplacian，成本远高于：

$$
M\times M
$$

的小矩阵 `solve`。

因此优化优先级应为：

```text
cross local energy
    >>
amplitude matrix
    >>
small-matrix linear algebra
```

而不是过早优化 `det` 的 \(O(M^3)\)。

### 10.1 pair flattening

把：

$$
(r,s)
$$

展平成：

```text
pair_id = r * M + s
```

形成：

```text
[M*M, ...]
```

后再：

```python
chunked_vmap(...)
```

建议配置：

```yaml
subspace:
  evaluation:
    pair_chunk_size: 8
```

例如：

- `M=4`：16 pairs；
- `M=8`：64 pairs；
- `M=16`：256 pairs。

避免一次保存全部二阶 autodiff activation。

### 10.2 可共享 potential

固定 replica 构型 \(\mathbf R_r\) 下：

$$
V(\mathbf R_r)
$$

与 state index \(s\) 无关。

如果 current JaQMC estimator 可以安全拆分：

$$
E_{L,s}(R_r)
=
T_{L,s}(R_r)
+
V(R_r),
$$

则：

- potential：只算 \(M\) 次；
- kinetic：仍算 \(M^2\) 次。

该优化应为 optional fast path，不要硬编码到核心 API，因为 nonlocal ECP 等项未必能完全按 state-independent potential 处理。

---

## 11. `DeterminantMCMCSampler`

新增：

```text
src/jaqmc/sampler/determinant.py
```

### 11.1 第一版 correctness sampler

第一版可以最大程度复用当前 `MCMCSampler`：

$$
\log p
=
2\operatorname{Re}\log\det\Phi.
$$

但普通 Gaussian proposal 会同时移动所有 replica，生产环境效率会较差。

该版本只用于：

```text
M=1,2
small system
correctness tests
```

### 11.2 production sampler

参考 NeuralQXLab：

每次随机选一个 replica：

$$
r\in\{0,\ldots,M-1\},
$$

只更新：

$$
\mathbf R_r
\rightarrow
\mathbf R'_r.
$$

此时 amplitude matrix 只有一行变化：

$$
\Phi_{r,:}
\rightarrow
\Phi'_{r,:}.
$$

只需要重新计算：

$$
\psi_s(\mathbf R'_r),
\qquad s=0,\ldots,M-1.
$$

因此每 proposal 的 NN amplitude evaluation 从重新构造整个 \(M^2\) matrix 降到：

$$
O(M)
$$

个 state evaluations。

---

## 12. sampler cache

推荐：

```python
@dataclass
class DeterminantSamplerState:
    stddev: jax.Array
    pmoves: jax.Array
    counter: jax.Array

    logdet: jax.Array
    amplitude_matrix: jax.Array
    inverse_matrix: jax.Array | None

    refresh_counter: jax.Array
```

每个 walker 可缓存：

```text
Φ
Φ^{-1}
log det Φ
```

第一版：

```yaml
determinant_update: full
```

即每次 proposal 后重新：

```text
slogdet
solve
```

production 再加入：

```yaml
determinant_update: rank1
refresh_frequency: 32
```

---

## 13. Sherman-Morrison fast path

若只有第 \(r\) 行改变：

$$
\Phi'
=
\Phi
+
e_r u^T.
$$

matrix determinant lemma：

$$
\det(\Phi')
=
\det(\Phi)
\left(1+u^T\Phi^{-1}e_r\right).
$$

Sherman-Morrison：

$$
(\Phi+e_ru^T)^{-1}
=
\Phi^{-1}
-
\frac{
\Phi^{-1}e_r u^T\Phi^{-1}
}{
1+u^T\Phi^{-1}e_r
}.
$$

NeuralQXLab 已采用该思路做 \(O(M^2)\) determinant probability update。

JaQMC 版本必须增加安全机制：

```text
if |factor| < threshold:
    full recompute
if refresh_counter >= refresh_frequency:
    full recompute
if inverse residual > threshold:
    full recompute
```

例如：

```yaml
rank1_factor_min: 1e-8
inverse_residual_max: 1e-7
refresh_frequency: 32
```

避免长期 rank-1 update 累积 floating-point drift。

---

## 14. 子空间 energy gradient：尽量复用 `LossAndGrad`

当前 JaQMC `LossAndGrad` 已实现标准 VMC energy gradient：

$$
\nabla_\theta E
=
2
\left\langle
(E_L-\langle E_L\rangle)
\nabla_\theta\log|\psi_\theta|
\right\rangle.
$$

determinant-state 本身就是一个合法的 variational wavefunction：

$$
\Psi_A(\Theta;\mathcal R)
=
\det\Phi.
$$

其 local energy 为：

$$
E_L^{A}
=
\mathrm{Tr}\,R_L.
$$

因此第一版应优先尝试：

```python
LossAndGrad(
    loss_key="subspace_energy",
    f_log_psi=determinant_wf.logpsi,
)
```

而不是新增一套独立 gradient engine。

如果该路径通过：

- finite difference；
- autodiff/reference；
- `M=1` regression；

则可继续使用 JaQMC 原来的 estimator lifecycle。

---

## 15. 优化器策略

### 第一阶段

仅使用：

```text
Optax Adam
```

原因：

- 参数 PyTree 已经带 state axis；
- Adam 不需要重新定义 Fisher metric；
- 最适合确认 determinant-state 梯度正确。

### 第二阶段

尝试：

```text
SR / natural gradient
```

### 第三阶段

才评估：

```text
KFAC
```

不要默认将现有单态 KFAC 原封不动用于 determinant-state。

原因是：

$$
\log\det\Phi
$$

把所有 state parameters 耦合，正确 Fisher metric 不再等价于 \(M\) 个完全独立的单态 Fisher block。

可评估两种方案：

1. **exact subspace SR/QGT**；
2. **block-diagonal approximate KFAC**，每个 state 一个 block，加适量跨态修正或忽略跨态块。

在没有验证前，KFAC 不作为初版默认优化器。

---

## 16. `SubspaceVMCWorkflow`

新增：

```text
src/jaqmc/workflow/subspace_vmc.py
```

### 16.1 设计原则

优先复用：

```text
VMCWorkflow
VMCWorkStage
```

而不是复制其训练循环。

推荐：

```python
class SubspaceVMCWorkflow(VMCWorkflow):
    config_namespace = "subspace_train"

    def __init__(self, cfg):
        super().__init__(cfg)

        base_wf = build_base_wavefunction(cfg)
        det_wf = DeterminantStateWavefunction(
            base_wavefunction=base_wf,
            spec=SubspaceSpec(
                n_states=cfg.subspace.n_states,
            ),
        )

        train = VMCWorkStage.builder(
            cfg.scoped("subspace_train"),
            det_wf,
        )

        train.configure_sample_plan(
            det_wf.logpsi,
            {"electrons": DeterminantMCMCSampler(...)},
        )

        train.configure_estimators(
            RayleighMatrixEstimator(...),
            ...
        )

        train.configure_loss_grads(
            f_log_psi=det_wf.logpsi,
        )

        train.configure_optimizer(
            default="jaqmc.optimizer.optax",
            ...
        )

        self.train_stage = train.build()
```

具体代码需以 JaQMC 当前 builder 的真实 API 为准，不应为了匹配此伪代码而反向修改 `VMCWorkStage`。

---

## 17. CLI 与配置

普通接口必须保持不变：

```bash
jaqmc molecule train ...
jaqmc solid train ...
```

新增子空间命令建议：

```bash
jaqmc molecule subspace-train ...
jaqmc solid subspace-train ...
```

不要通过：

```text
train.subspace=true
```

把大量条件分支塞进普通 ground-state workflow。

推荐配置：

```yaml
subspace:
  n_states: 4

  initialization:
    mode: independent
    perturbation_scale: 1.0e-3

  sampling:
    sampler: jaqmc.sampler.determinant.DeterminantMCMCSampler
    steps: 10
    initial_width: 0.02
    update_mode: full
    refresh_frequency: 32

  evaluation:
    estimator: jaqmc.estimator.rayleigh.RayleighMatrixEstimator
    walker_chunk_size: 16
    pair_chunk_size: 4
    matrix_dtype: complex128

  optimizer:
    module: jaqmc.optimizer.optax
    method: adam
    learning_rate: 1.0e-4

  diagnostics:
    condition_warning: 1.0e10
    solve_residual_warning: 1.0e-6
    max_imag_eigenvalue_warning: 1.0e-6
```

---

## 18. 初始化策略

不能简单复制同一个 ground-state 参数：

$$
\theta_0=\theta_1=\cdots=\theta_{M-1}.
$$

否则：

$$
\det\Phi=0.
$$

### 18.1 最小实现

分别随机初始化：

```python
theta_s = init(rng_s)
```

或者：

$$
\theta_s
=
\theta_{\mathrm{base}}
+
\epsilon_s.
$$

### 18.2 更好的 production 初始化

使用具有不同物理特征的初始态：

- HF ground determinant；
- single/double excitation；
- CASSCF low roots；
- 不同 \(S_z\) sector；
- 不同 momentum / parity；
- 不同 twist；
- selected-CI trial vectors。

但注意：不同严格 symmetry sector 通常应分 block 训练，而不是强行放入同一 determinant subspace。

---

## 19. 对称性分块

对氢链推荐：

```text
block 1: N, Sz=0, k=0, parity=+
block 2: N, Sz=1, ...
block 3: N±1 charge sectors
...
```

每个 block 独立：

$$
M=2,4,8,16
$$

子空间优化。

不要试图一次训练原 RDMD 文献中的 400 个态。

对于 neural QMC，合理目标是：

```text
M = 2 -> 4 -> 8 -> 16
```

并通过 `M` 收敛性检查确定 RDMD 所需低能空间是否已经稳定。

---

## 20. Rayleigh matrix 的数值诊断

理论上：

$$
R
=
G^{-1}G^{(H)}
$$

虽然不一定满足普通：

$$
R^\dagger=R,
$$

但其与 Hermitian matrix 相似，理想无噪声情况下 eigenvalues 为实数。

训练/eval 应记录：

```text
subspace_energy
subspace_energy_imag

ritz_energy[i]
ritz_energy_imag[i]

max_ritz_imag
local_rayleigh_variance
solve_residual

amplitude_sigma_min
amplitude_sigma_max
amplitude_condition
determinant_acceptance
```

尤其要监控：

$$
\max_i|\operatorname{Im}\lambda_i|.
$$

该量可作为：

- Monte Carlo noise；
- amplitude matrix ill-conditioning；
- linear solve 误差；
- 实现错误；

的敏感诊断。

---

## 21. complex matrix 统计不能直接套普通 scalar variance

`RayleighMatrixEstimator.reduce()` 建议覆盖默认 reduce。

对：

$$
R_{ij}\in\mathbb C
$$

可统计：

$$
\mathrm{Var}_c(R_{ij})
=
\left\langle
|R_{ij}-\bar R_{ij}|^2
\right\rangle.
$$

或者分别：

```text
variance_real
variance_imag
```

而：

$$
E_L^{\mathrm{sub}}
=
\mathrm{ReTr}R_L
$$

仍作为普通实 scalar loss 进入 `LossAndGrad`。

---

## 22. multi-GPU 设计

### 默认策略：只 shard walkers

数据：

```text
[B, M, Ne, 3]
```

默认：

```text
shard B
replicate M
```

理由：

每个 determinant row 必须同时得到：

$$
[\psi_0(R_r),\ldots,\psi_{M-1}(R_r)].
$$

如果把 state axis 拆到不同 GPU，则每次 MCMC proposal 都需要 all-gather amplitude row，通信成本会极高。

因此第一版和中等 \(M\) 生产版本都应：

$$
\boxed{
\text{walker parallelism first}
}
$$

只有 \(M\) 大到模型参数本身放不下一张卡时，才考虑 2D mesh：

```text
walker_axis × state_axis
```

这属于后续高级优化，不进入第一版。

---

## 23. 显存控制

主要显存风险不是小矩阵：

$$
[M,M],
$$

而是：

$$
M^2
$$

个 local-energy autodiff graph。

建议：

1. walker chunk；
2. pair chunk；
3. `lax.scan` 替代一次性 materialize 所有 pair activation；
4. 必要时重计算/gradient checkpoint；
5. local-energy evaluation 与 amplitude evaluation 分开编译；
6. state-independent potential 缓存；
7. eval 时 matrix dtype 可用 `complex128`，network 可保留原 precision；
8. 训练时先测试 `complex64`，但必须监控 conditioning。

推荐默认：

```text
network dtype: follow existing JaQMC
small matrix solve: complex128 if x64 available
```

如果 GPU 双精度代价过高，可配置为：

```text
complex64 + iterative refinement / diagnostics
```

但不能静默降低精度。

---

## 24. `subspace_linalg.py`

新增纯函数，避免线性代数散落在 sampler/estimator：

```python
def stable_complex_logdet(matrix): ...
def row_scaled_matrix(logpsi_matrix): ...
def rayleigh_solve(phi, phi_h): ...
def generalized_ritz(G, GH): ...
def hermitianized_rayleigh(G, GH): ...
def complex_variance(x, axis=0): ...
def matrix_condition_proxy(matrix): ...
def solve_residual(A, X, B): ...
```

这一层必须：

- 不 import JaQMC estimator；
- 不 import system；
- 不 import NetKet；
- 能单独 NumPy/JAX 单测。

---

## 25. 是否需要显式构造 `G`？

训练 determinant-state VMC 时：

$$
R
=
\mathbb E_{|\det\Phi|^2}
[
\Phi^{-1}\Phi^{(H)}
]
$$

不要求先估计 \(G\)。

因此默认训练使用 determinant estimator。

但 RDMD / observable / Bridge 后处理中仍可能需要：

$$
G,
\qquad
G^{(O)}.
$$

所以第二阶段可以增加：

```text
SOS / cross-overlap estimator
```

但它不应替代 determinant estimator 成为默认，因为近线性相关时：

$$
G^{-1}
$$

会放大采样误差。

---

## 26. RDMD 输出接口

RDMD 不应直接依赖 training workflow 内部对象。

子空间训练结束时导出标准 artifact：

```text
subspace_checkpoint/
├── component_params/
│   ├── state_000/
│   ├── state_001/
│   └── ...
├── metadata.json
├── rayleigh_matrix.npy
├── ritz_energies.npy
├── ritz_vectors.npy
├── diagnostics.json
└── config_resolved.yaml
```

后续加入：

```text
operator_matrices/
├── one_rdm.npy
├── two_rdm.npy
├── hopping_l1.npy
├── hopping_l2.npy
├── hopping_l3.npy
├── double_occupancy.npy
└── spin_correlation.npy
```

或者更通用地保存：

$$
G^{(O)}
$$

与：

$$
R_O = G^{-1}G^{(O)}.
$$

RDMD 只消费这些物理结果，不 import sampler/optimizer。

---

## 27. Bridge 的未来接口

不要在第一版子空间训练中加入 Bridge。

推荐未来：

```text
workflow/bridge.py
```

输入：

```text
existing time-dependent checkpoints
```

而不是训练 \(M\) 个新态。

Bridge workflow 复用：

```text
MultiStateEvaluator
RayleighMatrixEstimator
OperatorMatrixEstimator
subspace_linalg
```

但不复用：

```text
DeterminantState LossAndGrad
SubspaceVMC optimizer
```

其核心：

$$
i\dot\alpha
=
R\alpha,
$$

时间无关时：

$$
\alpha(t)=e^{-iRt}\alpha(0).
$$

这样 Section 2 静态低能子空间与 Section 3 Bridge 在软件上共享底层数学，而 workflow 保持独立。

---

## 28. 具体开发阶段

### P0：线性代数与 reference

新增：

```text
utils/subspace_linalg.py
tests/utils/test_subspace_linalg.py
```

验证：

- complex logdet；
- `solve`；
- basis transformation invariance；
- generalized eigenvalue；
- Hermitian similarity；
- complex variance。

**通过标准：**

与 NumPy/SciPy 小矩阵 reference 在 `float64/complex128` 下达到机器精度。

---

### P1：`M=1` determinant wavefunction

新增：

```text
wavefunction/determinant_state.py
```

只支持 `M=1`。

必须满足：

$$
\det[\psi(R)]=\psi(R),
$$

因此：

```text
determinant logpsi == base logpsi
```

**通过标准：**

- amplitude；
- log amplitude；
- gradient；
- MCMC probability；

与普通 JaQMC 完全一致。

---

### P2：`M=2` amplitude matrix，不训练

加入：

```text
component_logpsi_matrix
component_amplitude_row
```

随机两套参数。

验证：

- permutation 只改变 determinant sign/phase；
- \(|\det\Phi|^2\) 不变；
- `M=2` matrix 与直接 Python loop 一致。

---

### P3：Rayleigh estimator

新增：

```text
estimator/rayleigh.py
```

先不做 determinant sampler，只用固定 samples。

验证：

$$
R_L=\Phi^{-1}\Phi^{(H)}
$$

与手工 reference 一致。

然后验证：

$$
\mathrm{Tr}R_L
$$

与扩展 determinant wavefunction 直接 Hamiltonian local energy 在 tiny toy system 中一致。

---

### P4：full-recompute determinant sampler

新增：

```text
sampler/determinant.py
```

每次只移动一个 replica，但：

```text
Φ
logdet
inverse
```

全部重新计算。

**通过标准：**

- detailed balance；
- sampled observable 与 brute/reference 一致；
- `M=1` 与普通 MCMC 一致；
- 接受率合理。

---

### P5：联合训练

新增：

```text
workflow/subspace_vmc.py
```

先：

```text
M=2
Adam
```

目标：

$$
\min \mathrm{ReTr}R.
$$

选择可与 FCI 精确 benchmark 的小体系。

建议顺序：

```text
H2
He-like toy
LiH / very small molecule
```

**通过标准：**

- ground Ritz energy ≥ exact ground；
- first Ritz energy ≥ exact first target state；
- 随训练下降并稳定；
- `max Im Ritz energy` 接近统计噪声；
- 两 Ritz states 对应能级正确。

---

### P6：rank-1 sampler 优化

实现：

```text
Sherman-Morrison
matrix determinant lemma
periodic full refresh
```

要求：

```text
full update
vs
rank1 update
```

统计结果在误差条内一致。

---

### P7：chunking + multi-GPU

增加：

```text
walker_chunk_size
pair_chunk_size
```

测试：

```text
1 GPU
2 GPU
4 GPU
```

要求：

- energy 统计一致；
- walker 数扩展后吞吐接近合理线性增长；
- 不发生 state-axis all-gather 热点。

---

### P8：PBC 氢链

进入真实目标体系：

```text
H4 -> H6 -> H8
M = 2 -> 4 -> 8
OBC first
PBC/twist second
```

检查：

- Ritz spectrum；
- subspace convergence with M；
- 1-RDM / 2-RDM；
- RDMD descriptors；
- 不同 symmetry block。

---

## 29. 必须加入的不变量测试

### 29.1 `M=1` reduction

必须严格满足：

$$
\Psi_A=\psi,
$$

$$
R_L=E_L,
$$

$$
E_{\mathrm{sub}}=E_{\mathrm{VMC}}.
$$

这是最重要的回归测试。

### 29.2 basis permutation

交换：

$$
\phi_i\leftrightarrow\phi_j
$$

determinant state 只差符号，但：

$$
|\Psi_A|^2
$$

不变，

$$
\mathrm{Tr}R
$$

不变，

$$
\operatorname{eig}R
$$

不变。

### 29.3 一般 basis transform

对于小型显式线性组合测试：

$$
\phi'_i
=
\sum_j B_{ji}\phi_j,
\qquad
\det B\neq0,
$$

验证：

$$
R'
=
B^{-1}RB,
$$

因此：

$$
\operatorname{Tr}R'=\operatorname{Tr}R,
$$

$$
\operatorname{eig}R'=\operatorname{eig}R.
$$

### 29.4 near-linear-dependence

构造：

$$
\phi_2
=
\phi_1+\epsilon\chi
$$

扫描：

```text
epsilon = 1e-1 ... 1e-10
```

检查：

- condition number；
- solve residual；
- MCMC acceptance；
- rank-1 refresh；
- warning 是否触发；
- 不得 silent NaN。

---

## 30. FCI / exact benchmark

对可精确求解的小系统，应验证 generalized variational bound：

$$
E_k^{\mathrm{exact}}
\le
\mu_k.
$$

至少记录：

```text
E0 exact vs Ritz
E1 exact vs Ritz
...
sum exact low-M energies vs Tr(R)
```

同时做：

```text
M=1,2,4
```

收敛性。

---

## 31. 性能基准

每次性能测试拆分：

```text
t_component_logpsi_matrix
t_cross_local_energy
t_rayleigh_solve
t_sampler_proposal
t_rank1_update
t_full_refresh
t_loss_grad
t_optimizer
```

并记录：

```text
samples/s
walker-steps/s
GPU memory peak
GPU utilization
compile time
```

### 31.1 复杂度预期

amplitude matrix：

$$
O(M^2 C_\psi)
$$

完整 cross local energy：

$$
O(M^2 C_{EL})
$$

small solve：

$$
O(M^3)
$$

MCMC 单 replica row proposal：

$$
O(M C_\psi + M^2)
$$

其中通常：

$$
C_{EL} \gg M
$$

所以 real-space QMC 下主要热点应是 cross local energy，而不是 `solve`。

---

## 32. 许可证与代码复用策略

建议采取：

```text
algorithm/reference reuse
not source-copy reuse
```

即：

1. 论文和 NeuralQXLab 代码作为行为/reference；
2. JaQMC 内重新实现符合自身 API 的版本；
3. 保留来源说明与论文引用；
4. 如果要逐段复制 NeuralQXLab 具体代码，必须先核实其仓库当前许可证是否明确允许；
5. JaQMC 自身为 Apache-2.0，提交上游前需遵守其贡献规范。

这样既减少许可证风险，也避免 NetKet-specific abstraction 污染 JaQMC。

---

## 33. 推荐的第一版 public API

只公开四个核心概念：

```python
DeterminantStateWavefunction
DeterminantMCMCSampler
RayleighMatrixEstimator
SubspaceVMCWorkflow
```

内部 helper 不作为稳定 API：

```text
SubspaceSpec
row scaling
matrix cache
rank1 update
cross local-energy batching
```

后续再增加：

```python
OperatorRayleighEstimator
BridgeWorkflow
```

不要加入：

```text
RDMDWavefunction
ExcitedPsiformer
HubbardSpecificEstimator
```

RDMD 是消费者，不应该反向污染通用 QMC 核心。

---

## 34. 推荐调用链

```text
                         ┌─────────────────────┐
                         │ FermiNet/Psiformer  │
                         │ physical ψ_s(R)     │
                         └──────────┬──────────┘
                                    │
                            explicit params
                                    │
                         ┌──────────▼──────────┐
                         │ MultiStateEvaluator │
                         │ L_rs=log ψ_s(R_r)   │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴────────────────┐
                    │                                │
          ┌─────────▼──────────┐          ┌──────────▼─────────┐
          │ DeterminantSampler│          │ CrossLocalEnergy   │
          │ |det Φ|²          │          │ E_L,s(R_r)         │
          └─────────┬──────────┘          └──────────┬─────────┘
                    │                                │
                    └───────────────┬────────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │ RayleighEstimator     │
                         │ R_L=solve(Φ, ΦH)      │
                         └──────────┬────────────┘
                                    │
               ┌────────────────────┴────────────────────┐
               │                                         │
     ┌─────────▼─────────┐                     ┌─────────▼─────────┐
     │ Re Tr(R_L)        │                     │ Ritz / Operator   │
     │ subspace loss     │                     │ outputs           │
     └─────────┬─────────┘                     └─────────┬─────────┘
               │                                         │
     ┌─────────▼─────────┐                     ┌─────────▼─────────┐
     │ existing          │                     │ RDMD / Bridge     │
     │ LossAndGrad       │                     │ post-processing   │
     └─────────┬─────────┘                     └───────────────────┘
               │
     ┌─────────▼─────────┐
     │ Optax / future SR│
     └───────────────────┘
```

---

## 35. 建议的最小代码骨架

### `wavefunction/determinant_state.py`

```python
@dataclass(frozen=True)
class SubspaceSpec:
    n_states: int


class DeterminantStateWavefunction:
    def __init__(self, base_wavefunction, spec: SubspaceSpec):
        self.base_wavefunction = base_wavefunction
        self.spec = spec

    def component_logpsi_matrix(self, params, replica_data):
        # replica_data contains M physical configurations
        # output [M, M]
        ...

    def stable_amplitude_matrix(self, params, replica_data):
        logs = self.component_logpsi_matrix(params, replica_data)
        shift = jax.lax.stop_gradient(
            jnp.max(jnp.real(logs), axis=-1)
        )
        return jnp.exp(logs - shift[:, None]), shift

    def logpsi(self, params, replica_data):
        phi, shift = self.stable_amplitude_matrix(
            params, replica_data
        )
        sign, logabsdet = jnp.linalg.slogdet(phi)
        return (
            jnp.sum(shift)
            + jnp.log(sign)
            + logabsdet
        )
```

### `estimator/rayleigh.py`

```python
@configurable_dataclass
class RayleighMatrixEstimator(PerWalkerEstimator):
    f_amplitude_matrix: Any = runtime_dep(...)
    f_cross_local_energy: Any = runtime_dep(...)
    pair_chunk_size: int | None = None

    def evaluate_single_walker(
        self,
        params,
        data,
        prev_walker_stats,
        state,
        rngs,
    ):
        phi = self.f_amplitude_matrix(params, data)
        e_local = self.f_cross_local_energy(
            params,
            data,
            chunk_size=self.pair_chunk_size,
        )

        phi_h = phi * e_local
        rayleigh = jnp.linalg.solve(phi, phi_h)

        residual = (
            jnp.linalg.norm(phi @ rayleigh - phi_h)
            / jnp.maximum(jnp.linalg.norm(phi_h), 1e-30)
        )

        return {
            "local_rayleigh": rayleigh,
            "subspace_energy": jnp.real(jnp.trace(rayleigh)),
            "subspace_energy_imag": jnp.imag(jnp.trace(rayleigh)),
            "rayleigh_solve_residual": residual,
        }, state
```

### `sampler/determinant.py`

```python
@configurable_dataclass
class DeterminantMCMCSampler(...):
    steps: int = 10
    initial_width: float = 0.02
    update_mode: str = "full"
    refresh_frequency: int = 32

    def propose_single_replica(...):
        # choose r
        # move physical configuration R_r
        # evaluate new amplitude row
        ...

    def update_full(...):
        ...

    def update_rank1(...):
        ...
```

### `workflow/subspace_vmc.py`

```python
class SubspaceVMCWorkflow(VMCWorkflow):
    def __init__(self, cfg):
        super().__init__(cfg)

        base_wf = ...
        det_wf = DeterminantStateWavefunction(...)

        train = VMCWorkStage.builder(
            cfg.scoped("subspace_train"),
            det_wf,
        )

        ...
```

以上均为架构骨架，实际签名应以当前 JaQMC builder/runtime dependency API 为准。

---

## 36. 氢链应用建议

第一阶段不要直接做大 PBC 氢链。

推荐：

```text
Step 1: H2 / tiny molecule
Step 2: H4 OBC
Step 3: H6/H8 OBC
Step 4: H4/H6 PBC
Step 5: H8 PBC/twist
```

每个体系：

```text
M=1 -> 2 -> 4 -> 8
```

输出：

```text
Ritz spectrum
1-RDM
2-RDM
dt_l
dU
dJ
```

检查 RDMD effective Hamiltonian 随：

```text
M
walker count
network size
boundary condition
twist
```

的稳定性。

---

## 37. 验收标准

功能合并到主线前至少满足：

### 正确性

- `M=1` 与现有 JaQMC 结果回归一致；
- tiny system Rayleigh matrix 与 exact/reference 一致；
- Ritz energies 满足变分上界；
- basis permutation / basis transform invariance 通过；
- near-singular case 不 silent failure；
- full-update 与 rank1-update 统计一致。

### 数值稳定性

- `solve_residual` 可控；
- `max Im Ritz energy` 小于设定阈值；
- condition warning 可触发；
- periodic full refresh 后无长期 drift。

### 性能

- determinant proposal 单步只重算一个 amplitude row；
- `pair_chunk_size` 能有效限制显存；
- 多 GPU 默认只 shard walkers；
- `M=4/8` 在目标 GPU 上可运行且吞吐可接受。

### 架构

- 普通 `jaqmc ... train` 不受影响；
- NetKet 不成为 JaQMC dependency；
- physical Hamiltonian 无重复实现；
- RDMD/Bridge 不侵入 core VMC；
- 新功能可通过独立 workflow/config 使用。

---

## 38. 最终推荐结论

最合适的迁移方式不是：

```text
copy NeuralQXLab bridge package into JaQMC
```

而是：

```text
JaQMC physical electronic backend
        +
determinant-state wavefunction adapter
        +
determinant sampler
        +
Rayleigh estimator
        +
existing LossAndGrad / optimizer / VMC workflow
```

核心新增层只有：

$$
\boxed{
\texttt{DeterminantStateWavefunction}
+
\texttt{DeterminantMCMCSampler}
+
\texttt{RayleighMatrixEstimator}
+
\texttt{SubspaceVMCWorkflow}
}
$$

其中性能重点应放在：

$$
\boxed{
M^2\text{ cross local-energy evaluation}
}
$$

的 batching/chunking，而不是过早优化小型 `M×M` 线性代数。

完成这一基础层后：

- 静态低能子空间变分使用 `SubspaceVMCWorkflow`；
- RDMD 消费 Ritz/operator outputs；
- 动力学 Bridge 未来使用独立 `BridgeWorkflow`；
- 两者共享 determinant/Rayleigh 基础设施。

这样既保留 JaQMC 作为“连续电子 NNQMC 框架”的清晰边界，也充分利用 NeuralQXLab 已验证的 determinant-state 数值设计，同时避免引入 NetKet 特有的离散 Hilbert-space 抽象。

---

## 39. 参考代码与论文

### JaQMC

- Repository: https://github.com/bytedance/jaqmc
- `src/jaqmc/data.py`
- `src/jaqmc/wavefunction/base.py`
- `src/jaqmc/sampler/mcmc.py`
- `src/jaqmc/estimator/base.py`
- `src/jaqmc/estimator/loss_grad.py`
- `src/jaqmc/workflow/vmc.py`

### NeuralQXLab

- Repository: https://github.com/NeuralQXLab/variational-subspace-methods
- `libraries/bridge/bridge/_src/determinant_state.py`
- `libraries/bridge/bridge/_src/determinant_sampler.py`
- `libraries/bridge/bridge/_src/bridge.py`
- `libraries/bridge/bridge/_src/sum_of_states.py`

### 理论

- Adrien Kahn, Luca Gravina, Filippo Vicentini, *Variational subspace methods and application to improving variational Monte Carlo dynamics*, arXiv:2507.08930v2 / Quantum (accepted 2026).
- David Pfau et al., *Accurate computation of quantum excited states with neural networks*, Science 385 (2024).
