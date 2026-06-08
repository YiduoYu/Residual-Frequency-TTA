# ReSpect-TTA 极简实验故事�?
一句话主线�?
> 我们一步步验证：raw residual 不够，complex modulation 有信号，gate 很关键，direct-only �?horizon 不稳，最�?context-plus-complex 才成�?ReSpect-TTA�?
## 实验路径

| Step | 测试了什�?| 结果 | 得到的结�?|
|---|---|---|---|
| 1. Residual-only | 直接�?recent residual 修正预测 | 基本没提升：`adapter_residual_only` 只有 1/4 wins，MSE `-0.006%` | 不能简单把 residual 加回�?|
| 2. Direct complex modulation | �?`M(F_r) odot F_y` 直接修正 prediction spectrum | �?no-TTA 有信号：14/16 wins，MSE `+1.80%`；但输给 spectral context reference�?/16 wins，MSE `-0.99%` | complex correction 有用，但单独不够 |
| 3. Gate diagnostics | �?correction 加多少是否关�?| gate 改好�?against spectral context reference�?8/20 wins，MSE `+2.74%` | gate 是稳定性的关键 |
| 4. Long-horizon direct-only | �?direct complex path 能不能处理长 horizon | �?no-TTA 好：�?`+4.55%` �?`+6.29%`；对 spectral context reference 很差：约 `-20%` �?`-24%` | direct-only �?horizon 不够�?|
| 5. Low-pass sweep | 限制只修正低频能不能解决问题 | 仍然输给 spectral context reference：约 `-20.85%` �?`-21.16%` | 问题不是简单过滤高�?|
| 6. Context-plus-complex | �?context adapter �?complex modulation 结合 | seed0 long-horizon: 27/30 wins vs spectral context reference，MSE `+2.56%` | 这是最终方�?|
| 7. Full 3-seed main run | 大范围验�?`context_plus_complex_modulation_high_gate_lr` | 171 conditions；vs spectral context reference: 162/171 wins，MSE `+3.07%` | 确定�?ReSpect-TTA 主方�?|
| 8. Pred720 gate diagnostics | 检查长 horizon 720 的剩余问�?| constant gate 0.10: 40/42 wins，MSE `+3.15%`; floor005 只有小幅帮助 | 剩余问题主要�?long-horizon gate stability |

## 最终讲�?
> Residual-only failed, so we moved to complex spectral correction. Direct complex modulation showed useful signal but was unstable alone. Gate diagnostics showed that correction strength is critical. Long-horizon sweeps showed direct-only correction is not enough, and low-pass filtering does not solve it. The key improvement came from combining context adaptation with residual-conditioned complex modulation. The final ReSpect-TTA variant achieved 162/171 wins and +3.07% mean MSE improvement over spectral context reference.

## 最短版�?
```text
Residual-only: failed.
Complex modulation: useful but unstable alone.
Gate diagnostics: gate is critical.
Long horizon: direct-only not enough.
Low-pass: not the main fix.
Context-plus-complex: works.
Full run: +3.07% MSE vs spectral context reference on 171 conditions.
```

