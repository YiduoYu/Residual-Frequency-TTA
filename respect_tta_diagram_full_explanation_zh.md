# ReSpect-TTA 结构图完整解释

这个文件专门用来解释你图里的每一个模块：**是什么、为什么加、加了有什么作用、代码里怎么加、参数怎么传、老师追问时怎么回答**。

当前展示名称：**ReSpect-TTA: Spectral Residual Correction**

代码中的历史名称：`Scheme6` / `COSA_SCHEME6`

说明：代码里还保留 `Scheme6` 是因为这个方法最初是第六个实验方案，后来正式展示时改名为 ReSpect-TTA。

核心代码位置：

- `Spectral-master/tta/cosa_scheme6.py`
- `Scheme6ResidualSpectralAdapter`
- `Scheme6DirectSpectralCorrector`
- `script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh`

## 1. 一句话讲清楚

ReSpect-TTA 冻结原来的 forecasting backbone，只在测试阶段记录最近的预测残差，把基础预测和残差记忆都变到频域，然后用残差频谱生成一个 complex modulator 去直接修正预测频谱，最后通过 learned gate 控制修正强度并加回原预测。

更短版本：

> ReSpect-TTA uses recent residual spectra to directly correct the prediction spectrum while keeping the forecasting backbone frozen.

## 2. 图里的核心公式

```text
F_y = rFFT(Y_t^(0))
F_r = rFFT(R_buf)
Delta Y_t = iFFT(M(F_r) odot F_y)
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

每个符号含义：

- `X_t`：输入历史窗口。
- `Y_t^(0)`：frozen backbone 的原始预测。
- `R_buf`：最近 residual memory。
- `F_y`：base prediction 的频谱。
- `F_r`：recent residual memory 的频谱。
- `M(F_r)`：由 residual spectrum 生成的 complex modulator。
- `odot`：element-wise multiplication。
- `Delta Y_t`：从频域修正变回来的时间域 correction。
- `g_t`：learned correction gate。
- `Y_hat_t`：最终预测结果。

## 3. 总体 Pipeline

1. 输入历史窗口 `X_t`。
2. Frozen backbone 输出基础预测 `Y_t^(0)`。
3. 当前预测 `Y_t^(0)` 做 `rFFT`，得到 prediction spectrum `F_y`。
4. 代码从过去 batches 的 residual memory 里取最近的 residual profile。
5. residual profile 先平均、去均值、加 Hann window。
6. 处理后的 residual memory 做 `rFFT`，得到 residual spectrum `F_r`。
7. `F_r` 输入 complex modulator，生成 `M(F_r)`。
8. 用 `M(F_r)` 直接乘到 `F_y` 上：

```text
M(F_r) odot F_y
```

9. 对修正频谱做 `iFFT`，得到时间域修正项 `Delta Y_t`。
10. learned gate `g_t` 控制修正强度。
11. 最终预测：

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

12. 等当前 batch 的 ground truth 到达之后，用 MSE 更新 adapter。
13. 同时计算当前 residual，更新 residual memory 给后面的 batch 用。

最重要的答辩点：

> 当前 batch 的预测只使用过去 batch 的 residual，不使用当前 ground truth，所以没有 future leakage。

## 4. 模块逐个解释

## 4.1 Input Historical Window `X_t`

是什么：

- 输入给 forecasting backbone 的历史时间序列窗口。
- 形状可以理解成：

```text
[batch_size, seq_len, n_variables]
```

为什么需要：

- 这是原始 forecasting task 的标准输入。
- ReSpect-TTA 不改变 backbone 的输入方式，只在 backbone 输出之后做 test-time correction。

作用：

- 让原模型基于历史窗口预测未来 horizon。

代码里怎么传：

实验脚本里传入：

```text
DATA.SEQ_LEN = 96
DATA.LABEL_LEN = 48
DATA.PRED_LEN = 96 / 192 / 336 / 720
MODEL.seq_len = 96
MODEL.label_len = 48
MODEL.pred_len = pred_len
```

代码位置：

- `script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh`
- `Spectral-master/tta/cosa_scheme6.py` 里的 `adapt_simple`

答辩怎么说：

> The input is the same as the original forecasting setup. My method does not change the backbone input; it adds an online correction after the backbone prediction.

## 4.2 Frozen Backbone

是什么：

- 已经训练好的 forecasting model。
- 实验里包括 DLinear、FreTS、iTransformer、MICN、OLS、PatchTST。

为什么要 frozen：

- Test-time adaptation 如果更新整个大模型，容易不稳定。
- 冻结 backbone 可以保持原模型的基本预测能力。
- 只训练小 adapter，计算成本更低，也更适合在线场景。

作用：

- 提供稳定的基础预测 `Y_t^(0)`。
- ReSpect-TTA 只学习 correction，而不是重新训练预测模型。

代码里怎么做：

脚本中关闭训练，只开启 TTA：

```text
TRAIN.ENABLE False
TEST.ENABLE False
TTA.ENABLE True
TRAIN.CHECKPOINT_DIR = checkpoint path
```

adapter 初始化时：

```text
self._freeze_all_model_params()
self._unfreeze_adapter_params()
```

也就是：

- backbone 参数 `requires_grad=False`
- adapter 参数 `requires_grad=True`

代码位置：

- `Spectral-master/tta/cosa_scheme6.py`
- `Scheme6ResidualSpectralAdapter.__init__`

答辩怎么说：

> The forecasting backbone is frozen. I only update the lightweight spectral adapter, which makes the method more stable and efficient for test-time adaptation.

## 4.3 Base Prediction `Y_t^(0)`

是什么：

- frozen backbone 的原始预测。
- 没经过 ReSpect-TTA 修正之前的输出。

为什么需要：

- 它是最终预测的基础。
- ReSpect-TTA 的目标不是替换 backbone，而是学习一个 residual correction。

作用：

- 一方面作为最终公式里的 base term。
- 另一方面被转换到频域，得到 `F_y`，用于频谱修正。

代码里怎么做：

```text
pred, ground_truth = forecast(...)
original_pred = pred.clone()
```

这里 `original_pred` 保存了 backbone 的原始预测，后面 residual memory 也是基于它计算。

代码位置：

- `Spectral-master/tta/cosa_scheme6.py`
- `adapt_simple`

答辩怎么说：

> `Y_t^(0)` is the original frozen model prediction. ReSpect-TTA learns a correction on top of this prediction.

## 4.4 Prediction Spectrum `F_y`

是什么：

- base prediction 的 real FFT：

```text
F_y = rFFT(Y_t^(0))
```

为什么加：

- 时间序列误差经常有频率结构。
- 例如 trend 偏差、seasonality 错位、周期幅度不准、phase shift。
- 在频域修正可以更直接地控制不同频率成分。

作用：

- 把预测从时间域变到频域。
- 让 adapter 能对不同 frequency bins 分别修正。

代码里怎么做：

```text
fy = torch.fft.rfft(y.float(), n=self.pred_len, dim=1)
```

形状：

```text
y:   [B, pred_len, V]
F_y: [B, pred_len//2 + 1, V]
```

相关参数：

```text
FFT_PRECISION = fp32
FFT_LENGTH_MODE = exact
```

为什么用 `rFFT`：

- 输入预测是实数序列。
- `rFFT` 只保留非冗余频率，比完整 FFT 更高效。

答辩怎么说：

> I convert the base prediction into the frequency domain, so the adapter can correct frequency components instead of only adjusting individual time steps.

## 4.5 Recent Residual Memory `R_buf`

是什么：

- 最近几个 test batches 的 signed residual memory。
- residual 定义为：

```text
R_t = Y_t - Y_t^(0)
```

注意这里是 signed residual，不是 MSE。

为什么加：

- Test-time distribution shift 下，backbone 的错误模式会变化。
- 最近 residual 可以告诉我们：模型现在主要错在哪里。
- 如果模型近期一直低估、周期幅度不够、或者趋势偏移，residual memory 会记录这种模式。

作用：

- 提供 online adaptation signal。
- 让方法根据最近错误动态调整预测。

代码里怎么做：

ground truth 到达之后：

```text
signed_residual = targets - prediction
residual_profile = signed_residual.mean(dim=0)
self.sample_history.append(batch_info)
```

其中 `prediction` 是 `original_pred`，也就是 frozen backbone 的原始预测。

为什么 residual memory 用 `original_pred` 算：

- 因为我们想记录 backbone 的原始错误。
- 如果用已经修正后的 prediction，会把 adapter 自己的影响混进去。
- 用 `original_pred` 更清楚地表示 backbone 在当前 test distribution 下的 failure pattern。

相关参数：

```text
TTA.COSA.BUFFER_CONTEXT_SIZE = 10
TTA.COSA.SPECTRAL.HISTORY_SIZE = 8
```

两个参数区别：

- `BUFFER_CONTEXT_SIZE=10`：给 context adapter 用。
- `HISTORY_SIZE=8`：给 direct spectral corrector 用，用最近 8 个 residual profile 的平均。

代码位置：

- `Spectral-master/tta/cosa_scheme6.py`
- `update_memory_buffer`
- `_reference_profile`

答辩怎么说：

> The residual memory stores how the frozen backbone recently failed. It provides online feedback for the spectral correction module.

## 4.6 Residual Spectrum `F_r`

是什么：

- recent residual memory 的频域表示：

```text
F_r = rFFT(R_buf)
```

为什么加：

- time-domain residual 只能看到每个时间点错多少。
- frequency-domain residual 可以看到错误的频率模式。
- 例如低频部分强，说明 trend / mean shift 可能比较明显。
- 高频部分强，说明短期波动或 noise mismatch 比较明显。

作用：

- 给 complex modulator 提供 conditioning signal。
- 让 adapter 根据近期 residual 的频谱结构生成 correction。

代码里怎么做：

先从 memory 里拿最近的 residual profile：

```text
profile = average(last HISTORY_SIZE residual_profile entries)
```

然后去均值：

```text
ref = ref - ref.mean(dim=1, keepdim=True)
```

再加 Hann window：

```text
ref = ref * self.window
```

最后做 `rFFT`：

```text
fr = torch.fft.rfft(ref.float(), n=self.pred_len, dim=1)
```

相关参数：

```text
HISTORY_SIZE = 8
WINDOW_TYPE = hann
FFT_PRECISION = fp32
FFT_LENGTH_MODE = exact
```

为什么去均值：

- 避免 DC component 过强。
- 让频谱更多反映 residual 的变化结构，而不是只反映整体偏置。

为什么加 Hann window：

- FFT 默认序列是周期延拓。
- 短序列边界不连续会造成 spectral leakage。
- Hann window 可以减弱边界突变，让 residual spectrum 更稳定。

答辩怎么说：

> The residual spectrum summarizes recent errors by frequency. It tells the adapter whether the current error pattern is more like trend shift, seasonal mismatch, or high-frequency variation.

## 4.7 Complex Modulator `M(F_r)`

是什么：

- 一个 trainable complex-valued modulator。
- 它根据 residual spectrum `F_r` 生成一个复数乘子 `M(F_r)`。

公式：

```text
M(F_r) = W [Re(F_r), Im(F_r), |F_r|, freq] + b
```

为什么加：

- 如果只把 residual spectrum 当作 context，模型只是“看见” residual。
- ReSpect-TTA 希望 residual spectrum 直接决定如何修正 prediction spectrum。
- complex modulator 可以同时修正 amplitude 和 phase。

作用：

- 给每个 frequency bin 和每个 variable 生成一个 complex correction operator。
- 可以调节预测频谱的强度和相位。

代码里怎么做：

可训练参数：

```text
complex_mod_weight: [n_freq, n_vars, 4, 2]
complex_mod_bias:   [n_freq, n_vars, 2]
```

输入特征：

```text
fr_norm.real
fr_norm.imag
abs(fr_norm)
freq_coord
```

代码逻辑：

```text
features = torch.stack(
    [fr_norm.real, fr_norm.imag, torch.abs(fr_norm), freq_coord],
    dim=-1
)
raw = torch.einsum("bfvc,fvco->bfvo", features, self.complex_mod_weight)
raw = raw + self.complex_mod_bias.unsqueeze(0)
modulation = torch.complex(raw[..., 0], raw[..., 1])
```

形状：

```text
F_r:      [B, n_freq, V]
features: [B, n_freq, V, 4]
M(F_r):   [B, n_freq, V]
```

相关参数：

```text
CORRECTION_SOURCE = complex_modulation
COMPLEX_MOD_FREQ_CUTOFF_RATIO = 1.0
```

为什么输入里有 `freq_coord`：

- 同样的 residual magnitude 在低频和高频里的含义不一样。
- `freq_coord` 让 modulator 知道当前处理的是哪个频率位置。

为什么要 normalize residual spectrum：

- 代码里用 residual energy 对 `F_r` 做 normalization。
- 这样 modulator 不会只被 residual 的绝对大小支配，而能更多关注 residual 的频谱形状。

为什么初始化为 0：

- `complex_mod_weight` 和 `complex_mod_bias` 初始都是 0。
- 初始 correction 也接近 0。
- 所以一开始模型基本等于 frozen backbone，不会刚开始就产生很大的错误修正。

答辩怎么说：

> The complex modulator maps recent residual spectra into a complex correction operator. Because it is complex-valued, it can adjust both amplitude and phase of the prediction spectrum.

## 4.8 Element-wise Spectral Modulation `M(F_r) odot F_y`

是什么：

- 用 residual-conditioned modulator 直接乘 prediction spectrum：

```text
Z = M(F_r) odot F_y
```

为什么加：

- 这是 ReSpect-TTA 的核心改动。
- residual spectrum 不只是 context，而是直接变成 correction operator。
- 它直接作用在 `F_y` 上。

作用：

- 对每个 frequency 和 variable 进行独立修正。
- 可以改变 prediction spectrum 的 amplitude 和 phase。

代码里怎么做：

```text
modulation = self._complex_modulation(fr)
modulation = modulation * self.complex_mod_freq_mask
z = z + modulation * fy
```

相关参数：

```text
COMPLEX_MOD_FREQ_CUTOFF_RATIO = 1.0
```

参数含义：

- `1.0` 表示所有频率都允许被修正。
- `0.05` / `0.10` / `0.20` 是 low-pass ablation，只允许低频修正。

为什么用乘法，不是直接加 residual：

- 直接加 residual spectrum：只是把过去的错误形状加回来。
- 乘 prediction spectrum：是根据 residual pattern 去变换当前预测。
- 这更像一个 operator，而不是一个普通 residual feature。

答辩怎么说：

> This is the key difference. The residual spectrum directly modulates the prediction spectrum, so it acts as a correction operator rather than only a context feature.

## 4.9 iFFT and Spectral Correction `Delta Y_t`

是什么：

- 把修正后的频谱变回时间域：

```text
Delta Y_t = iFFT(M(F_r) odot F_y)
```

为什么加：

- 最终 forecast 必须是 time-domain sequence。
- 频域里得到的 correction 要通过 inverse FFT 回到时间域。

作用：

- 得到可直接加到 base prediction 上的 correction term。

代码里怎么做：

```text
delta = torch.fft.irfft(z, n=self.pred_len, dim=1)
```

形状：

```text
z:         [B, n_freq, V]
Delta Y_t: [B, pred_len, V]
```

答辩怎么说：

> After spectral modulation, I use inverse FFT to convert the correction back to the time domain.

## 4.10 Learned Correction Gate `g_t`

是什么：

- 一个 learned gate，控制 correction 的强度。

最终公式：

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

为什么加：

- 不是所有 residual memory 都可靠。
- 有些 batch 的 residual 可能很 noisy。
- 如果 correction 太强，可能 over-correction。
- gate 让模型自己判断 correction 该开多大。

作用：

- 控制修正幅度。
- 降低错误修正风险。
- 让 online adaptation 更稳。

代码里怎么做：

gate 输入特征：

```text
hs              = current prediction mean
l2              = residual spectral energy
low_freq_ratio  = low-frequency residual power ratio
```

代码：

```text
gate_features = self._global_gate_features(y, low_freq_ratio, spectral_l2)
raw_global_gate = torch.sigmoid(self.global_gate(gate_features).squeeze(-1))
y_hat = y + global_gate.unsqueeze(1) * delta
```

gate 网络结构：

```text
Linear(input_dim, hidden_dim)
ReLU
Linear(hidden_dim, 1)
Sigmoid
```

主实验参数：

```text
USE_GLOBAL_GATE = True
GLOBAL_GATE_FEATURES = hs,l2,low_freq_ratio
GLOBAL_GATE_INIT_BIAS = -2.0
GLOBAL_GATE_LR_MULTIPLIER = 10.0
GLOBAL_GATE_FLOOR = 0.0
CONSTANT_GLOBAL_GATE = -1.0
```

这些参数是什么意思：

- `USE_GLOBAL_GATE=True`：使用 learned gate。
- `GLOBAL_GATE_INIT_BIAS=-2.0`：初始 gate 约等于 `sigmoid(-2)=0.119`。
- `GLOBAL_GATE_LR_MULTIPLIER=10.0`：gate 的学习率是基础 adapter learning rate 的 10 倍。
- `GLOBAL_GATE_FLOOR=0.0`：没有强制最小 gate。
- `CONSTANT_GLOBAL_GATE=-1.0`：不使用固定 gate，使用 learned gate。

为什么初始 gate 不设成 1：

- 如果一开始 correction 全开，adapter 还没学好时可能伤害预测。
- 初始 gate 小一点更保守。

为什么 gate learning rate 更高：

- online setting 里 gate 需要很快判断 correction 要不要打开。
- 高一点的 gate LR 能更快适应不同 batch 的 residual reliability。

答辩怎么说：

> The gate is a safety mechanism. It controls how much spectral correction should be added, so the model can avoid over-correction when residual memory is unreliable.

## 4.11 Final Add / Fuse

是什么：

- 把 base prediction 和 gated correction 相加：

```text
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

为什么加：

- 这是 residual learning 的形式。
- 预测主干仍然来自 frozen backbone。
- adapter 只负责补偿当前 test distribution 下的误差。

作用：

- 保留原模型能力。
- 在需要时添加频域 residual correction。

代码里怎么做：

```text
y_hat = y + global_gate.unsqueeze(1) * delta
```

答辩怎么说：

> The final prediction is still anchored on the frozen backbone. ReSpect-TTA only adds a gated spectral residual correction.

## 4.12 Online Update After Ground Truth Arrives

是什么：

- 当前 batch 的真实值到达后，在线更新 adapter。

为什么加：

- Test-time adaptation 的关键是使用 test stream 中逐步到达的 feedback。
- 这样模型可以根据最近错误持续适应。

作用：

- 更新 adapter 参数。
- 更新 residual memory。
- backbone 仍然 frozen。

代码里怎么做：

预测阶段：

```text
with torch.no_grad():
    pred = self._apply_prediction_adapter(pred, raw_context)
```

训练 adapter 阶段：

```text
adapted_pred = self._apply_prediction_adapter(original_pred, raw_context)
loss = F.mse_loss(adapted_pred, ground_truth)
l2_reg = sum(param.pow(2).sum() for param in self._adapter_params)
loss = loss + 1e-4 * l2_reg
loss.backward()
torch.nn.utils.clip_grad_norm_(self._adapter_params, max_norm=0.1)
self.optimizer.step()
```

更新 residual memory：

```text
self.update_memory_buffer(ground_truth, original_pred)
```

相关参数：

```text
TTA.COSA.STEPS = 3
TTA.COSA.BATCH_SIZE = 48
```

答辩怎么说：

> The adapter is updated only after the ground truth arrives. The current prediction uses past residual memory, so there is no label leakage.

## 5. 代码里还有一个 Context Path

你图里主要画的是 direct spectral correction path。代码主实验其实使用：

```text
SPECTRAL.USAGE = context_plus_direct
```

这表示完整实现有两部分：

1. COSA-style spectral context adapter。
2. ReSpect-TTA direct spectral correction。

代码逻辑：

```text
adapted = self.output_adapter(prediction, projected_context)
final = self.direct_corrector(adapted, self.sample_history)
```

context 包含什么：

- recent target mean context。
- residual spectral features。
- top `K=8` non-DC frequency bins。
- `real_imag` representation。

相关参数：

```text
SPECTRAL.USAGE = context_plus_direct
SPECTRAL.MODE = concat
SPECTRAL.K = 8
SPECTRAL.HISTORY_SIZE = 8
SPECTRAL.REPR = real_imag
SPECTRAL.SOURCE = residual_profile
USE_CONTEXT_PROJECTION = True
CONTEXT_PROJECTION_DIM = 8
```

为什么图里没有详细画：

- 这一页 slide 的重点是新方法的核心：direct residual-conditioned spectral correction。
- context path 更接近前面 COSA-Spectral 的已有部分。
- 为了图简洁，主要突出 `F_r -> M(F_r) -> M(F_r) odot F_y -> Delta Y_t` 这一条新路径。

老师问为什么没画时可以说：

> The full implementation keeps the COSA-Spectral context adapter, but this slide focuses on the new direct correction path. The main novelty is that residual spectrum directly modulates the prediction spectrum.

## 6. 每个新增组件为什么加

| 新增组件 | 为什么加 | 作用 |
|---|---|---|
| residual memory `R_buf` | 记录最近 backbone 的错误模式 | 提供 online adaptation signal |
| residual spectrum `F_r` | 把错误模式转到频域 | 识别 trend / seasonality / high-frequency error |
| prediction spectrum `F_y` | 把预测转到频域 | 允许 frequency-wise correction |
| complex modulator `M(F_r)` | 用 residual spectrum 生成修正算子 | 同时调整 amplitude 和 phase |
| `M(F_r) odot F_y` | 让 residual 直接作用于 prediction spectrum | residual 不只是 context，而是 operator |
| `iFFT` | 把频域修正转回时间域 | 得到 `Delta Y_t` |
| learned gate `g_t` | 控制 correction strength | 减少 over-correction |
| Hann window | 减少 FFT 边界泄漏 | 让 residual spectrum 更平滑 |
| diagonal output projection | 给频率和变量一个轻量 scaling | 增加一点表达能力，参数量低 |
| high gate LR | gate 需要快速适应 | 让 correction 开关更灵活 |
| frozen backbone | 避免在线更新大模型不稳定 | 稳定、轻量、可控 |

## 7. 主实验参数完整表

主实验 variant：

```text
context_plus_complex_modulation_high_gate_lr
```

实验输出 tag：

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

参数表：

| 参数 | 值 | 作用 |
|---|---:|---|
| `TRAIN.ENABLE` | `False` | 不训练 backbone |
| `TEST.ENABLE` | `False` | 不走普通 test |
| `TTA.ENABLE` | `True` | 开启 test-time adaptation |
| `TTA.COSA.BATCH_SIZE` | `48` | 在线适应 batch size |
| `TTA.COSA.STEPS` | `3` | 每个 batch 更新 adapter 3 步 |
| `TTA.COSA.BUFFER_CONTEXT_SIZE` | `10` | context memory 长度 |
| `TTA.COSA.SPECTRAL.USAGE` | `context_plus_direct` | context adapter + direct correction |
| `TTA.COSA.SPECTRAL.MODE` | `concat` | 拼接 mean context 和 spectral context |
| `TTA.COSA.SPECTRAL.K` | `8` | 使用前 8 个 non-DC frequency bins |
| `TTA.COSA.SPECTRAL.HISTORY_SIZE` | `8` | residual profile 平均长度 |
| `TTA.COSA.SPECTRAL.ENABLE_DIRECT_CORRECTION` | `True` | 开启 direct spectral correction |
| `TTA.COSA.SPECTRAL.CORRECTION_SOURCE` | `complex_modulation` | 使用 complex modulator |
| `TTA.COSA.SPECTRAL.GATE_MODE` | `unidirectional` | residual-to-prediction direction |
| `TTA.COSA.SPECTRAL.USE_GLOBAL_GATE` | `True` | 使用 learned global gate |
| `TTA.COSA.SPECTRAL.GLOBAL_GATE_INIT_BIAS` | `-2.0` | 初始 gate 较小但不是关闭 |
| `TTA.COSA.SPECTRAL.GLOBAL_GATE_LR_MULTIPLIER` | `10.0` | gate 学得更快 |
| `TTA.COSA.SPECTRAL.GLOBAL_GATE_FLOOR` | `0.0` | 不强制 gate 下限 |
| `TTA.COSA.SPECTRAL.CONSTANT_GLOBAL_GATE` | `-1.0` | 不用固定 gate |
| `TTA.COSA.SPECTRAL.COMPLEX_MOD_FREQ_CUTOFF_RATIO` | `1.0` | 所有频率都允许修正 |
| `TTA.COSA.SPECTRAL.WINDOW_TYPE` | `hann` | residual FFT 前加 Hann window |
| `TTA.COSA.SPECTRAL.OUT_PROJ` | `diagonal` | 轻量 frequency-wise scaling |
| `TTA.COSA.SPECTRAL.FFT_PRECISION` | `fp32` | FFT 使用 float32 |
| `TTA.COSA.SPECTRAL.FFT_LENGTH_MODE` | `exact` | FFT 长度等于 prediction length |

实验范围：

```text
Models: DLinear, FreTS, iTransformer, MICN, OLS, PatchTST
Datasets: ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather, electricity, traffic
Prediction lengths: 96, 192, 336, 720
Seeds: 0, 1, 2
```

## 8. Shape 说明

设：

```text
B = batch size
L = prediction length
V = number of variables
F = L // 2 + 1
```

| 符号 | Shape | 含义 |
|---|---|---|
| `X_t` | `[B, seq_len, V]` | 输入历史窗口 |
| `Y_t^(0)` | `[B, L, V]` | backbone 原始预测 |
| `R_t` | `[B, L, V]` | signed residual |
| `residual_profile` | `[L, V]` | 一个 batch 平均后的 residual |
| `R_buf` | `[B, L, V]` | 最近 residual profile 平均后扩展到 batch |
| `F_y` | `[B, F, V]` complex | prediction spectrum |
| `F_r` | `[B, F, V]` complex | residual spectrum |
| `M(F_r)` | `[B, F, V]` complex | complex modulator |
| `Z` | `[B, F, V]` complex | correction spectrum |
| `Delta Y_t` | `[B, L, V]` | time-domain correction |
| `g_t` | `[B, V]` | learned correction gate |
| `Y_hat_t` | `[B, L, V]` | final forecast |

## 9. 和 COSA-Spectral 的区别

最短一句：

> COSA-Spectral uses residual spectrum as context; ReSpect-TTA uses it as a direct correction operator.

中文解释：

- COSA-Spectral：把 residual spectrum 作为 context feature，输入 adapter。
- ReSpect-TTA：用 residual spectrum 生成 `M(F_r)`，然后直接乘到 prediction spectrum `F_y` 上。

核心区别公式：

```text
COSA-Spectral: residual spectrum -> context -> adapter
ReSpect-TTA:   residual spectrum -> M(F_r) -> M(F_r) odot F_y
```

答辩怎么说：

> The main difference is that ReSpect-TTA does not only feed residual spectrum as context. It uses residual spectrum to directly construct a spectral correction operator.

## 10. “Residual spectrum acts as a correction operator” 是什么意思

这句话的意思不是说 residual spectrum 自己就是最终预测。

它的意思是：

1. `F_r` 先进入 complex modulator。
2. complex modulator 输出 `M(F_r)`。
3. `M(F_r)` 直接乘到 `F_y` 上。
4. 所以 residual spectrum 参与决定 prediction spectrum 应该怎么被改变。

通俗说：

> residual 不只是告诉模型“哪里错了”，而是进一步告诉模型“下一次应该怎么修正 prediction spectrum”。

## 11. 老师可能问的问题和回答

### Q1. 为什么不用 input `X_t`，而要用 residual？

因为 residual 直接反映当前模型在 test distribution 下的错误。`X_t` 只是输入信号，而 residual 告诉我们 frozen model 最近怎么错。

### Q2. 为什么要用 FFT？

很多时间序列误差有频率结构，例如趋势偏移、周期幅度错误、seasonality mismatch。FFT 让模型能按频率修正，而不是只在时间点上修正。

### Q3. 为什么要 complex modulation？

complex multiplier 可以同时影响 amplitude 和 phase。时间序列预测里，phase shift 或周期位置偏移也很重要，单纯 real-valued scaling 不够。

### Q4. 这个方法有没有偷看 future label？

没有。当前预测只用过去 residual memory。当前 batch 的 ground truth 只在预测完成后用来更新 adapter 和 residual buffer。

### Q5. 为什么 backbone 要 frozen？

因为 online test-time adaptation 更新整个模型容易不稳定，也更耗计算。冻结 backbone 可以保留原模型能力，只让轻量 adapter 适应当前分布。

### Q6. 为什么需要 gate？

residual memory 不一定总可靠。如果 correction 太强，可能 over-correct。gate 控制 correction 强度，让模型在不确定时少修正。

### Q7. 为什么 complex modulator 初始化为 0？

这样初始 correction 接近 0，模型刚开始等价于 frozen backbone。适应开始时更稳定。

### Q8. 为什么 `GLOBAL_GATE_INIT_BIAS=-2.0`？

因为 `sigmoid(-2.0)` 大约是 `0.119`，表示 correction 初始较小但不是完全关闭，适合 conservative adaptation。

### Q9. 为什么 gate learning rate 要乘 10？

gate 是控制 correction 是否可信的开关，online setting 中它需要快速适应，所以给它更高 learning rate。

### Q10. 为什么用 Hann window？

FFT 对有限序列边界比较敏感。Hann window 可以降低边界 discontinuity 导致的 spectral leakage。

### Q11. 为什么不是直接加 residual？

直接加 residual 是把过去的错误形状加回来；ReSpect-TTA 用 residual spectrum 去调制 prediction spectrum，更像一个 learned correction operator，表达能力更强。

### Q12. 具体训练哪些参数？

训练 adapter，不训练 backbone。包括：

- context adapter 参数。
- context projector 参数。
- complex modulator 的 weight 和 bias。
- global gate 参数。
- diagonal output projection 参数。

### Q13. loss 是什么？

```text
MSE(adapted_prediction, ground_truth) + 1e-4 * L2(adapter_parameters)
```

### Q14. 局限是什么？

如果 recent residual 很 noisy，或者 distribution shift 突然变化，residual memory 可能不可靠，correction 也可能不稳定。所以需要 gate 控制 correction strength。

## 12. 一页 slide 的讲法

可以直接这样讲：

> This slide shows ReSpect-TTA, a residual-conditioned spectral correction method. First, the frozen backbone gives a base prediction `Y_t^(0)`. After previous ground truth becomes available, we store recent signed residuals in a memory buffer. For the current prediction, we transform both the base prediction and the residual memory into the frequency domain. The key idea is that the residual spectrum is not only used as context. It generates a complex modulator `M(F_r)`, which directly corrects the prediction spectrum `F_y`. Then inverse FFT gives a time-domain correction `Delta Y_t`, and a learned gate `g_t` controls how much correction is added. The final output is the base prediction plus a gated spectral residual correction.

中文辅助理解：

> 这页图的故事就是：原模型先预测；我不动原模型；我看它过去最近怎么错；把过去错误变到频域；用错误频谱生成一个频域修正器；这个修正器直接改当前预测频谱；最后通过 gate 控制修正幅度，加回原预测。

## 13. 最简洁版本

如果时间非常短，只讲这个：

```text
Frozen backbone -> base prediction Y_t^(0)
Past residual memory -> residual spectrum F_r
Base prediction -> prediction spectrum F_y
F_r -> complex modulator M(F_r)
Correction = iFFT(M(F_r) odot F_y)
Final = Y_t^(0) + gated correction
```

## 14. 代码位置速查

| 内容 | 代码位置 |
|---|---|
| ReSpect-TTA adapter 入口 | `Spectral-master/main.py` |
| 识别 `COSA_SCHEME6` 并 build adapter | `Spectral-master/main.py` |
| 主 adapter 类 | `Spectral-master/tta/cosa_scheme6.py`, `Scheme6ResidualSpectralAdapter` |
| 直接频谱修正模块 | `Spectral-master/tta/cosa_scheme6.py`, `Scheme6DirectSpectralCorrector` |
| residual memory 更新 | `Spectral-master/tta/cosa_scheme6.py`, `update_memory_buffer` |
| residual reference profile | `Spectral-master/tta/cosa_scheme6.py`, `_reference_profile` |
| prediction spectrum / residual spectrum | `Spectral-master/tta/cosa_scheme6.py`, `forward` |
| complex modulator | `Spectral-master/tta/cosa_scheme6.py`, `_complex_modulation` |
| gate features | `Spectral-master/tta/cosa_scheme6.py`, `_global_gate_features` |
| 最终公式 | `Spectral-master/tta/cosa_scheme6.py`, `forward` |
| 在线训练 adapter | `Spectral-master/tta/cosa_scheme6.py`, `adapt_simple` |
| COSA-style context | `Spectral-master/tta/cosa_spectral.py`, `SpectralContextBuilder` |
| 实验参数传入 | `script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh` |

## 15. 不要过度声称的点

不要说：

- 这是一个全新的 forecasting backbone。
- 它保证所有 dataset 和 horizon 都提升。
- 它在当前预测前使用了当前 ground truth。
- 它完全替代 COSA-Spectral。

应该说：

- 它是一个 lightweight test-time residual correction module。
- 它加在 frozen pretrained backbone 之后。
- 它用 recent residual spectra 做 online adaptation。
- 它的核心 novelty 是 residual-conditioned direct spectral modulation。
- 效果要结合 dataset、horizon、model 分析，因为 residual 是否稳定会影响 correction 是否有效。

## 16. 老师问 “你到底加了什么？” 的标准回答

> I added a direct residual-conditioned spectral correction path. It stores recent residuals, transforms them into residual spectra, uses the residual spectrum to generate a complex modulator, applies this modulator to the prediction spectrum, converts the result back to the time domain, and adds it to the base prediction through a learned gate.

中文理解：

> 我加的是一条直接的 residual-conditioned spectral correction path。它不是只把 residual spectrum 当 context，而是用 residual spectrum 生成一个频域修正算子，直接修正当前 prediction spectrum。

## 17. 老师问 “为什么这个方法合理？” 的标准回答

> Under test-time distribution shift, recent residuals reveal systematic forecasting errors of the frozen model. If these errors have frequency structure, correcting the prediction in the spectral domain can be more targeted than only applying a time-domain adapter.

中文理解：

> test-time shift 下，最近 residual 反映原模型当前怎么错。如果这些错误有频率结构，那么在频域做修正比只在时间域修正更有针对性。

## 18. 老师问 “参数怎么设的？” 的标准回答

> The main variant uses `context_plus_direct`, `complex_modulation`, `K=8`, `HISTORY_SIZE=8`, Hann window, diagonal output projection, learned global gate, gate initialization bias `-2.0`, gate learning-rate multiplier `10.0`, `TTA_STEPS=3`, and batch size `48`.

中文理解：

> 主方法用 context plus direct correction，correction source 是 complex modulation，残差历史长度 8，取前 8 个频率 bin，Hann window，diagonal output projection，learned global gate，gate 初始 bias 是 -2，gate 学习率乘 10，每个 batch 适应 3 步，batch size 48。

