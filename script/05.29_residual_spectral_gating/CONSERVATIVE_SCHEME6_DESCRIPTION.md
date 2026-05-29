# Scheme 6 最保守 Adapter 方案说明

## 1. 方案目的

这份说明对应当前 `2026-05-29-residual-spectral-gating` 分支里最简单、
最保守的 Scheme 6 实现。它基本就是 Word 技术报告里建议优先采用的
保守主线：

```text
rFFT direct + reference spectrum Fr + 单向 Gy gate + global gate + residual add
```

当前第一阶段的目标不是一次性测试所有可能的设计，而是先回答一个最核心
的问题：

```text
在 backbone 已经给出预测 y0 之后，插入一个很小的 residual-aware
spectral adapter，是否能稳定改善预测？
```

所以现在的 runner 只围绕 adapter 本身做 ablation。normalization、
PAAS、dataloader tuning、复杂训练策略这些都先不写进脚本里，避免第一轮
实验被太多非核心变量干扰。

## 2. 当前最保守主线

当前核心流程如下：

```text
base forecast y0
  -> rFFT(y0) = Fy
  -> 从历史 residual buffer 得到 reference spectrum Fr
  -> 用 Fr 生成单向 local gate Gy
  -> 得到 gated current spectrum: Gy * Fy
  -> 可选轻量 diagonal spectral projection
  -> irFFT(..., n=L) 得到 delta
  -> global gate g 控制整体修正强度
  -> residual add: y_hat = y0 + g * delta
```

对应公式可以写成：

```text
Fy = rfft(y0, n=L)
Fr = rfft(window(Rbuf), n=L)
Gy = sigmoid(a_y * |Fr| + b_y)
Z  = Gy * Fy
delta = irfft(Wo(Z), n=L)
y_hat = y0 + g * delta
```

其中 `Rbuf` 来自历史 residual profile，`g` 是一个小的 sample-level
global gate。

这个版本保守的关键点是：`Fr` 只负责告诉模型当前预测频谱 `Fy` 哪些频率
更可信，哪些频率应该压低。它不直接把历史 residual 的内容注入进预测。

## 3. 当前包含什么

当前 core adapter 包含：

- 使用 `torch.fft.rfft` 做 one-sided real FFT。
- 使用 `torch.fft.irfft(..., n=L)`，显式传入原始长度 `L`。
- 使用历史 `residual_profile` 构造 reference buffer。
- 对 reference residual 使用 Hann window。
- 使用单向 local spectral gate：

```text
Gy = sigmoid(a_y * |Fr| + b_y)
```

- 默认使用轻量 diagonal spectral output projection。
- 默认开启 global gate：

```text
g = sigmoid(MLP(mean(y0), ||Fr||2, low_freq_ratio))
```

- 最后通过 residual add 输出：

```text
y_hat = y0 + g * delta
```

## 4. 第一轮故意不包含什么

第一轮最保守版本先不使用：

- bidirectional reference content injection：

```text
Gr * Fr
```

- dense spectral output projection。
- spectral path 上的 RevIN。
- 新的 normalization 策略。
- PEFT backend routing。
- multi-window spectral encoder。
- 大规模实验矩阵参数。

这些以后都可以作为扩展 ablation，但不应该混进第一轮。第一轮应该先隔离
核心 adapter 是否有效。

## 5. 当前代码位置

当前 adapter 实现在：

```text
Spectral-master/tta/cosa_scheme6.py
```

当前 runner 在：

```text
script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

结果 family 是：

```text
COSA_SCHEME6
```

当前配置走的是：

```text
TTA.COSA.SPECTRAL.*
```

不是旧的：

```text
TTA.COSA.SELECT.*
```

这样以后和 `main` merge 时也更稳定，不会依赖已经被 main 替换掉的
`cosa_select.py` 路径。

## 6. 默认最小实验

现在 runner 的默认实验非常小：

```text
model:   DLinear
dataset: ETTh1
horizon: 96
seed:    0
```

默认 ablation 是：

```text
core
no_out_proj
no_global_gate
```

其中主方案是：

```text
core
```

它对应：

```text
SPECTRAL.USAGE = direct
GATE_MODE = unidirectional
USE_GLOBAL_GATE = True
OUT_PROJ = diagonal
WINDOW_TYPE = hann
ENABLE_REFERENCE_CONTENT_INJECTION = False
FFT_PRECISION = fp32
FFT_LENGTH_MODE = exact
```

## 7. 为什么第一轮只做这些 ablation

第一轮三个 ablation 分别回答最基础的问题：

```text
core
```

测试完整的最保守 adapter。

```text
no_out_proj
```

测试 diagonal spectral projection 是否真的必要。如果它和 `core` 差不多，
后面可以考虑进一步简化。

```text
no_global_gate
```

测试 global gate 是否真的在防止过度修正。这个版本可能更不稳定，但它能
帮助判断 global gate 是不是负迁移控制的关键。

后面可以再加：

```text
context_plus_core
```

它会把旧的 spectral context adapter 叠回 direct spectral correction
上面。但这应该是第二阶段，不应该作为第一轮默认测试。

## 8. 运行方式

只看命令是否展开正确：

```bash
DRY_RUN=True bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

只跑最保守 core adapter：

```bash
ABLATIONS="core" bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

跑默认小 ablation：

```bash
bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

## 9. 第一轮结果应该看什么

第一轮不要只看一个 MSE。建议同时看：

- `test_mse`
- `test_mae`
- `direct_global_gate_mean`
- `direct_local_gate_mean`
- `direct_delta_norm`
- `direct_low_freq_ratio_mean`
- `no_global_gate` 是否平均更好但修正更激进
- `no_out_proj` 是否和 `core` 接近

如果 `no_global_gate` 容易出现更大的 correction norm 或更明显负迁移，
说明 global gate 确实有保护作用。

如果 `no_out_proj` 和 `core` 几乎一样，说明 diagonal projection 可能不是
必要组件，后面可以把主方法进一步简化。

## 10. 保守决策规则

如果 `core` 相比 base 或当前 spectral baseline 有稳定提升，并且没有明显
不稳定，就把 `core` 作为主线。

如果 global gate 能减少负迁移或过大修正，即使平均收益略低，也优先保留
global gate。

如果 `no_out_proj` 与 `core` 基本持平，就可以考虑去掉 projection，让方法
更简单。

如果 bidirectional reference content injection 没有跨 seed、dataset、
horizon 稳定胜出，就不要把它放进主方法，只作为 ablation。

这就是当前最保守方案的核心逻辑：先证明 guarded direct spectral adapter
有效，再根据 ablation 决定是否扩展。不要一开始就把复杂组件都加上去。
