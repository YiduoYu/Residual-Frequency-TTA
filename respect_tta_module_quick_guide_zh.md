# ReSpect-TTA 模块一句话速查

这个文件用于快速理解结构图中每个模块的意义，以及 slide 上哪些模块可以合并显示。

## 主流程模块

| 模块 | 一句话解释 | 能不能从图上去掉 |
|---|---|---|
| `Input historical window X_t` | 当前历史输入窗口，用来让 frozen backbone 产生基础预测。 | 不能完全去掉，可简写为 `Input X_t` |
| `Frozen Backbone` | 已训练好的预测模型，保持冻结，只输出 base prediction。 | 不能去掉，这是 TTA adapter 的前提 |
| `Base Prediction Y_t^(0)` | backbone 的原始预测，也是最终 correction 的基础。 | 不能去掉 |
| 上方 `rFFT` | 把 base prediction 转成 prediction spectrum。 | 算法不能去，图上可合并到 `F_y` |
| `Prediction Spectrum F_y` | 原始预测的频域表示，是被修正的对象。 | 不能去掉 |
| `Recent residual memory R_buf` | 存储最近窗口的预测残差，用来表示模型当前怎么错。 | 不能去掉 |
| 下方 `rFFT` | 把 residual memory 转成 residual spectrum。 | 算法不能去，图上可合并到 `F_r` |
| `Residual Spectrum F_r` | 最近错误的频域模式，用来生成 spectral correction operator。 | 不能去掉 |
| `Complex Modulator M(F_r)` | 根据 residual spectrum 学出 complex-valued correction operator。 | 不能去掉，这是核心 |
| `M(F_r) odot F_y` | 用 residual-conditioned modulator 直接修正 prediction spectrum。 | 不能去掉，这是核心公式 |
| `iFFT` | 把频域 correction 转回时间域 correction。 | 算法不能去，图上可写进公式不单独画框 |
| `Spectral Correction Delta Y_t` | 从频域变回来的时间域修正项。 | 不能去掉 |
| `learned correction gate g_t` | 控制 correction 加多少，防止 over-correction。 | 主方法保留，ablation 可去掉 |
| `Y_hat_t = Y_t^(0) + g_t odot Delta Y_t` | 把 base prediction 和 gated correction 融合成最终预测。 | 不能去掉 |
| 右上角说明框 | 强调 residual spectrum 直接修正 prediction spectrum。 | 可以去掉，口头讲即可 |

## 最简图建议

如果想让 slide 更简洁，可以保留下面这条主线：

```text
Y_t^(0) -> F_y
R_buf -> F_r -> M(F_r)
Delta Y_t = iFFT(M(F_r) odot F_y)
Y_hat_t = Y_t^(0) + g_t odot Delta Y_t
```

## iFFT 为什么不能在算法里去掉

`M(F_r) odot F_y` 仍然是频域结果，而最终预测必须回到时间域，所以需要：

```text
Delta Y_t = iFFT(M(F_r) odot F_y)
```

因此 `iFFT` **算法上必须有**，但 **图上可以不单独画成一个框**。

## 和 COSA-Spectral 的一句话区别

```text
COSA-Spectral uses residual spectrum as context;
ReSpect-TTA uses residual spectrum as a direct correction operator.
```

## Complex Modulator 内部一句话

```text
F_r -> normalize -> [Re(F_r), Im(F_r), |F_r|, freq] -> W h + b -> M(F_r)
```

它的作用是把 residual spectrum 转成 complex-valued modulator，从而同时影响 prediction spectrum 的 amplitude 和 phase。

