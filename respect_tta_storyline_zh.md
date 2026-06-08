# ReSpect-TTA 故事线与实验路径

这个文件只讲 ReSpect-TTA / ReSpect-TTA 这条线，不展开前面�?spectral context reference 方法细节�?
核心问题�?
> 既然 residual spectrum 已经能作�?context，能不能进一步把 residual spectrum 当成一个直接修�?prediction spectrum �?operator�?
最终方法：

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

展示名：

```text
ReSpect-TTA: Spectral Residual Correction
```

一句话故事线：

> 我们不是一开始就得到 ReSpect-TTA，而是先发�?raw residual correction 没用，再发现 direct complex modulation 有一点信号但不稳定，接着发现 gate policy 很关键，最后把 residual-conditioned complex modulation �?context adapter 结合，才得到最终稳定提升的 ReSpect-TTA�?
## 1. 最终要讲的主线

可以按这个顺序讲�?
```text
Residual-only failed
-> Direct complex modulation showed signal
-> Gate policy was critical
-> Direct-only failed on long horizons
-> Context-plus-complex became the key design
-> Full 3-seed run confirmed ReSpect-TTA
-> Pred720 diagnostics showed remaining limitation is gate stability
```

中文口语版：

> 一开始我们只是想直接用最�?residual 去修正预测，�?residual-only 基本没有提升。然后我们把 residual 放到频域里，�?complex modulator 去修�?prediction spectrum，发现它�?no-TTA 有信号，但是还打不过强一点的 spectral context baseline。接着我们发现问题不只�?modulator，而是 correction 加多少非常关键，所以做�?gate 诊断。再往长预测窗口上测，direct-only 虽然能超�?no-TTA，但�?horizon 仍然不稳定。最后我们把 context path �?direct complex correction 结合，得�?context-plus-complex high-gate-lr，也就是现在�?ReSpect-TTA 主方法�?
## 2. 实验 1：Residual-only Smoke Test

目的�?
> 先测试最简单的想法：能不能直接�?recent residual / residual spectrum 加回预测�?
实验设置�?
- Model: `DLinear`
- Datasets: `ETTh2`, `ETTm2`, `electricity`, `exchange_rate`
- Horizon: `96`
- 4 �?averaged conditions
- 3 seeds

结果�?
| Variant | Baseline wins | Mean MSE improvement vs baseline | Mean MAE improvement vs baseline |
|---|---:|---:|---:|
| `adapter_residual_only` | 1/4 | -0.006% | +0.000% |
| `adapter_residual_force_global_gate_open` | 1/4 | -0.194% | -0.166% |

我们从这个实验学到什么：

> 直接 residual correction 太粗糙，基本没有用；强行打开 gate 也没有解决问题�?
设计决策�?
> 放弃 residual-only，转向更 expressive �?complex spectral modulation�?
答辩讲法�?
> The first experiment showed that simply injecting residuals is not enough, so we needed a more structured correction in the frequency domain.

## 3. 实验 2：Direct Complex Modulation Tier1

目的�?
> 测试 complex modulation 这条 direct correction path 本身有没有信号�?
核心改动�?
```text
Delta Y_t = iFFT(M(F_r) odot F_y)
```

也就是用 residual spectrum 生成 complex modulator，然后直接作用在 prediction spectrum 上�?
实验覆盖�?
- 16 averaged rows
- Models: `DLinear`, `PatchTST`, `iTransformer`
- Datasets: `ETTh2`, `ETTm2`, `electricity`, `exchange_rate`, `traffic`, `weather`
- Horizon: `96`

总体结果�?
| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 16 | 14/16 | +1.80% |
| spectral context reference | 16 | 5/16 | -0.99% |
| context adapter reference fixed/adaptive scheduling combined | 32 | 12/32 | -0.80% |

�?variant�?
| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| `adapter_complex_modulation` | baseline | 4 | 4/4 | +0.07% |
| `adapter_complex_modulation` | spectral context reference | 4 | 0/4 | -1.80% |
| `adapter_complex_modulation_force_global_gate_open` | baseline | 12 | 10/12 | +2.37% |
| `adapter_complex_modulation_force_global_gate_open` | spectral context reference | 12 | 5/12 | -0.73% |

我们从这个实验学到什么：

> Direct complex modulation �?residual-only 更有希望，因为它能超�?no-TTA；但它单独还不够强，尤其比不过更稳定�?spectral context 方法�?
设计决策�?
> 保留 complex modulation，但继续研究 gate �?context 结合�?
答辩讲法�?
> Direct complex modulation showed that residual spectrum can provide useful correction signal, but using the direct path alone was not stable enough.

## 4. 实验 3：Complex Gate Diagnostics

目的�?
> 判断 direct modulation 弱是不是因为 gate policy 不好，也就是 correction 加太多或太少�?
测试�?gate 方向�?
- constant gate
- desaturated gate
- high gate learning rate

实验覆盖�?
- 20 averaged rows
- Models: `DLinear`, `PatchTST`, `iTransformer`
- Horizon: `96`

总体结果�?
| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 20 | 19/20 | +5.23% |
| spectral context reference | 20 | 18/20 | +2.74% |
| context adapter reference fixed/adaptive scheduling combined | 40 | 36/40 | +3.08% |

Against spectral context reference by variant�?
| Variant | Conditions | Wins | Mean MSE improvement |
|---|---:|---:|---:|
| `adapter_complex_modulation_constant_gate` | 4 | 4/4 | +4.07% |
| `adapter_complex_modulation_desat_gate` | 4 | 4/4 | +1.84% |
| `adapter_complex_modulation_high_gate_lr` | 12 | 10/12 | +2.59% |

我们从这个实验学到什么：

> Modulator 本身有用，但 correction strength 很敏感；gate 是决定方法是否稳定的关键组件�?
设计决策�?
> 最终方法保�?learned gate，并使用 high gate learning rate，让 gate 更快适应当前 residual reliability�?
答辩讲法�?
> The gate diagnostics showed that the issue was not only the spectral operator itself. How much correction we apply is critical.

## 5. 实验 4：Long-Horizon Direct-Only Sweeps

目的�?
> 测试 direct complex modulation 在长预测 horizon 下能不能独立工作�?
测试内容�?
- 不加 context path，只�?direct correction�?- �?gate size sweep�?- �?low-pass frequency cutoff sweep�?
Gate sweep against no-TTA baseline�?
| Variant | Conditions | Wins | Mean MSE improvement vs baseline |
|---|---:|---:|---:|
| `constant_gate_005` | 42 | 42/42 | +4.55% |
| `constant_gate_010` | 42 | 42/42 | +5.71% |
| `constant_gate_020` | 42 | 39/42 | +6.29% |
| `high_gate_lr` | 42 | 40/42 | +6.01% |

但是 against spectral context reference�?
| Variant | Conditions | Wins | Mean MSE improvement vs spectral context reference |
|---|---:|---:|---:|
| `constant_gate_005` | 30 | 1/30 | -24.22% |
| `constant_gate_010` | 30 | 3/30 | -22.39% |
| `constant_gate_020` | 30 | 4/30 | -21.03% |
| `high_gate_lr` | 30 | 4/30 | -20.82% |

Low-pass sweep 也没有救回来�?
| Variant | Conditions | Wins vs spectral context reference | Mean MSE improvement vs spectral context reference |
|---|---:|---:|---:|
| `high_gate_lr_lowpass_005` | 30 | 3/30 | -21.16% |
| `high_gate_lr_lowpass_010` | 30 | 4/30 | -20.90% |
| `high_gate_lr_lowpass_020` | 30 | 4/30 | -20.85% |

我们从这个实验学到什么：

> Direct spectral correction 可以超过 no-TTA，但�?horizon 下单独使用不够稳定；问题不是简�?low-pass 就能解决的�?
设计决策�?
> direct complex path 需要和 context path 结合，不能只�?direct-only�?
答辩讲法�?
> Long-horizon sweeps showed that direct correction alone is not enough. It improves over no-TTA but fails against the stronger context-based adaptation, so we combined both.

## 6. 实验 5：Context-Plus-Complex Seed0 Breakthrough

目的�?
> 测试�?context path �?direct complex modulation 结合后，能不能解�?direct-only 的弱点�?
组合方式�?
```text
context adapter + residual-conditioned complex modulation
```

也就是：

- context path 负责提供稳定�?test-time adaptation context�?- direct complex path 负责直接修正 prediction spectrum�?
结果�?
| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| direct-only `high_gate_lr` | baseline | 42 | 40/42 | +6.01% |
| direct-only `high_gate_lr` | spectral context reference | 30 | 4/30 | -20.82% |
| context-plus-complex `high_gate_lr` | baseline | 42 | 42/42 | +21.63% |
| context-plus-complex `high_gate_lr` | context adapter reference fixed | 30 | 30/30 | +10.99% |
| context-plus-complex `high_gate_lr` | spectral context reference | 30 | 27/30 | +2.56% |

我们从这个实验学到什么：

> 这是 ReSpect-TTA 方向第一次真正跑通。complex modulation 单独不够，但�?context adapter 结合后，效果明显变强�?
设计决策�?
> �?`context_plus_complex_modulation_high_gate_lr` 提升为主候选方法�?
答辩讲法�?
> This was the breakthrough experiment. The direct complex operator becomes useful when combined with the context adapter.

## 7. 实验 6：Targeted Gate / Low-Pass Diagnostics

目的�?
> 在一�?hard long-horizon cases 上进一步看，是不是固定 gate �?low-pass 能提升稳定性�?
Seed0 targeted results�?
| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| `context_plus_constant_gate_010` | baseline | 12 | 12/12 | +21.04% |
| `context_plus_constant_gate_010` | spectral context reference | 12 | 8/12 | +1.28% |
| `context_plus_high_gate_lr_lowpass_010` | spectral context reference | 12 | 4/12 | -0.57% |
| `context_plus_high_gate_lr_lowpass_020` | spectral context reference | 12 | 4/12 | -0.57% |

3-seed constant gate 0.10 targeted check�?
| Reference | Conditions | Wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 12 | 12/12 | +20.95% |
| spectral context reference | 12 | 10/12 | +1.18% |
| context adapter reference fixed/adaptive scheduling combined | 24 | 24/24 | +14.51% |

我们从这个实验学到什么：

> 固定 gate 0.10 �?hard long-horizon cases 上更稳；low-pass restriction 不是主要解决方案�?
设计决策�?
> 主方法继续用 learned high-gate-lr；constant gate 0.10 作为 long-horizon stability diagnostic�?
答辩讲法�?
> The targeted diagnostics showed that stability is mostly a gate issue, not simply a frequency cutoff issue.

## 8. 实验 7：Full 3-Seed Main ReSpect-TTA Run

目的�?
> 在更大范围上验证主候选方法：

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

覆盖范围�?
- 171 averaged model-dataset-horizon conditions
- 每个 collected condition 使用 3 seeds
- Models: `DLinear`, `FreTS`, `MICN`, `OLS`, `PatchTST`, `iTransformer`
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`, `electricity`, `traffic`
- Horizons: `96`, `192`, `336`, `720`

说明�?
- 理论完整 grid �?192 conditions�?- 实际 collected result �?171 conditions�?- 缺失主要来自部分 traffic/electricity �?model-horizon 组合�?
总体结果�?
| Reference | Comparable conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement | Mean MAE improvement |
|---|---:|---:|---:|---:|---:|---:|
| No-TTA baseline | 171 | 161/171 | +14.37% | +11.13% | -3.48% | +6.81% |
| spectral context reference | 171 | 162/171 | +3.07% | +2.72% | -2.54% | +1.14% |
| context adapter reference fixed/adaptive scheduling combined | 342 | 332/342 | +9.43% | +6.35% | -1.67% | +3.89% |

�?backbone against spectral context reference�?
| Backbone | Conditions | MSE wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| DLinear | 32 | 30/32 | +2.70% | -1.79% |
| FreTS | 28 | 25/28 | +2.65% | -1.84% |
| MICN | 32 | 32/32 | +4.86% | +1.15% |
| OLS | 27 | 25/27 | +2.70% | -2.54% |
| PatchTST | 24 | 22/24 | +2.41% | -1.05% |
| iTransformer | 28 | 28/28 | +2.78% | +0.89% |

�?dataset against spectral context reference�?
| Dataset | Conditions | MSE wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| ETTh1 | 24 | 24/24 | +3.51% | +0.64% |
| ETTh2 | 24 | 23/24 | +2.32% | -0.02% |
| ETTm1 | 24 | 20/24 | +1.54% | -2.06% |
| ETTm2 | 24 | 20/24 | +2.07% | -2.54% |
| electricity | 19 | 19/19 | +3.70% | +1.52% |
| exchange_rate | 24 | 24/24 | +4.25% | +0.69% |
| traffic | 8 | 8/8 | +3.67% | +1.88% |
| weather | 24 | 24/24 | +4.04% | +0.88% |

�?horizon against spectral context reference�?
| Horizon | Conditions | MSE wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| 96 | 43 | 42/43 | +3.05% | -0.21% |
| 192 | 43 | 43/43 | +3.30% | +0.64% |
| 336 | 43 | 42/43 | +3.25% | -0.36% |
| 720 | 42 | 35/42 | +2.68% | -2.54% |

最�?cases�?
| Backbone | Dataset | Horizon | MSE improvement |
|---|---|---:|---:|
| MICN | ETTh1 | 336 | +8.59% |
| OLS | exchange_rate | 720 | +8.54% |
| MICN | ETTh1 | 720 | +8.27% |
| MICN | ETTh2 | 336 | +8.10% |
| DLinear | exchange_rate | 720 | +7.89% |

最�?cases�?
| Backbone | Dataset | Horizon | MSE improvement |
|---|---|---:|---:|
| OLS | ETTm2 | 720 | -2.54% |
| OLS | ETTm1 | 720 | -2.06% |
| FreTS | ETTm1 | 720 | -1.84% |
| DLinear | ETTm2 | 720 | -1.79% |
| DLinear | ETTm1 | 720 | -1.58% |
| PatchTST | ETTm1 | 720 | -1.05% |

我们从这个实验学到什么：

> ReSpect-TTA 的主方法是整�?archive 里最强的 broad result。它说明 direct frequency-domain correction �?context path 之上确实有额外价值�?
设计决策�?
> 最终报告和 presentation 采用 `adapter_context_plus_complex_modulation_high_gate_lr` 作为 ReSpect-TTA 主方法�?
答辩讲法�?
> The full 3-seed run confirms that the final context-plus-complex design improves over the references on most collected conditions, especially across different backbones, datasets, and horizons.

## 9. 实验 8：Pred720 Gate Stability Diagnostics

目的�?
> 主方法的负迁移主要集中在 horizon 720，尤�?ETTm1 / ETTm2，所以继续检�?long-horizon gate stability�?
### Constant Gate 0.10 Pred720

覆盖�?
- 42 averaged conditions
- Horizon: `720`
- 3 seeds

结果�?
| Reference | Conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|---:|
| No-TTA baseline | 42 | 42/42 | +28.78% | +23.76% | +2.03% |
| spectral context reference | 42 | 40/42 | +3.15% | +3.23% | -0.69% |
| context adapter reference fixed/adaptive scheduling combined | 84 | 81/84 | +16.35% | +13.45% | -3.02% |

结论�?
> 固定 gate 0.10 是一个很强的 long-horizon diagnostic，说�?pred720 稳定性确实和 gate 很相关�?
### Learned Gate Floor 0.005 Pred720

结果�?
| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 42 | 42/42 | +28.46% |
| spectral context reference | 42 | 35/42 | +2.74% |
| context adapter reference fixed/adaptive scheduling combined | 84 | 83/84 | +16.15% |

直接比较�?
| Comparison | Conditions | Wins | Mean improvement |
|---|---:|---:|---:|
| floor005 vs learned high-gate | 42 | 25/42 | +0.07% |
| floor005 vs constant gate 0.10 | 42 | 17/42 | -0.41% |

结论�?
> gate floor 有帮助，但还不足以替代主方法�?constant gate diagnostic�?
### MICN Pred720 Diagnostics

目的�?
> MICN 在主方法里表现特别好，所以测试是不是只是因为它需要更大的固定 gate�?
结果 against spectral context reference�?
| Variant | Conditions | Wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| constant gate 0.20 | 8 | 8/8 | +5.26% | +3.41% |
| learned gate floor 0.005 | 8 | 8/8 | +5.39% | +1.75% |
| learned gate floor 0.010 | 8 | 8/8 | +5.20% | +0.10% |

结论�?
> MICN 确实稳定受益，但不只是“gate 大一点”这么简单，learned adaptivity 仍然重要�?
## 10. 最终故事怎么�?
完整讲法�?
> Our goal was to make residual spectral information more active. Instead of only using it as context, we wanted it to directly correct the prediction spectrum. We first tried residual-only correction, but it did not improve the baseline, so raw residual injection was not enough. Then we introduced direct complex modulation, where the residual spectrum generates a complex operator `M(F_r)` and modulates the prediction spectrum. This showed useful signal against no-TTA, but direct-only correction was still unstable, especially against stronger spectral adaptation. Gate diagnostics showed that correction strength is critical, and long-horizon sweeps showed that direct-only correction is not enough. The breakthrough came when we combined the context adapter with the direct complex modulation path. In the full 3-seed run, this final ReSpect-TTA variant improved over spectral context reference on 162 out of 171 conditions with a mean MSE improvement of 3.07%. The remaining weakness is mainly long-horizon stability, so our follow-up diagnostics focused on fixed gates and gate floors.

中文讲法�?
> 我们的故事线不是一开始就提出最终结构，而是一步一步试出来的。最开�?residual-only 失败，说明不能简单把 residual 加回去；然后 complex modulation 证明 residual spectrum 作为频域修正器是有信号的，但 direct-only 还不稳定；再通过 gate diagnostics 发现 correction 加多少很关键；长 horizon 实验进一步说�?direct-only 不够，需要和 context path 结合；最�?context-plus-complex high-gate-lr �?3 seeds�?71 个条件上取得主要结果，说�?residual spectrum 作为 direct correction operator 是有额外价值的�?
## 11. 30 秒版�?
> We developed ReSpect-TTA through several ablations. Residual-only correction failed, so we moved to complex spectral modulation. Direct complex modulation improved over no-TTA, but was not stable enough by itself. Gate diagnostics showed that correction strength is critical, and long-horizon sweeps showed that direct-only correction cannot fully handle difficult horizons. The key improvement came from combining context-based adaptation with residual-conditioned complex modulation. The final context-plus-complex high-gate-lr variant achieved +3.07% mean MSE improvement over spectral context reference on 171 collected conditions.

## 12. 10 秒版�?
> We tried residual-only first, then direct complex modulation, then gate diagnostics, and finally context-plus-complex correction. The final ReSpect-TTA works because residual spectrum is not only context; it directly becomes a gated complex correction operator on the prediction spectrum.

## 13. 最终结�?
ReSpect-TTA 的实验逻辑是：

```text
不是 residual 本身有魔法，
而是 residual spectrum + complex operator + gate + context path 组合起来才稳定有效�?```

最�?claim 可以说：

> ReSpect-TTA adds value because it turns residual spectral information from a passive context feature into an active, gated, complex-valued correction operator.

