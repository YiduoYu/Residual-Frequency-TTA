# 05.29 Residual Spectral Gating

- Created: `2026-05-29`.
- Previous version: legacy flat `script/` files from `3-versions-final-next-all-in-spectral`.
- Status: `active`.

## Change Overview

This version deploys Scheme 6 as the conservative residual spectral gating path.
It is wired through the main-compatible `COSA_SCHEME6` adapter path instead of
the older `cosa_select.py` path, so future merges into `main` do not depend on a
file that `main` has already replaced.
The default method uses one-sided rFFT on the current base prediction and the
historical residual profile, applies a unidirectional residual-spectrum gate
`Gy * Fy`, then applies a sample-level global gate before residual addition.

The bidirectional reference-content path remains available only as an explicit
ablation through environment variables. It is not the default.

## Runner

- Entrypoint: `run_residual_spectral_gating.sh`
- Result family: `COSA_SCHEME6`
- Adapter: `Spectral-master/tta/cosa_scheme6.py`
- Default scope: one small adapter ablation on `DLinear / ETTh1 / pred96 / seed0`.
- Default ablations: `core`, `no_out_proj`, `no_global_gate`.
- The runner intentionally leaves normalization, PAAS, dataloader tuning, and
  broader training/runtime knobs at config defaults. First-pass testing should
  isolate the core adapter behavior.

Useful conservative smoke test:

```bash
DRY_RUN=True bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run only the conservative adapter:

```bash
ABLATIONS="core" bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

## 2026-05-29 Conservative Formula Checklist

This is the minimal conservative Scheme 6 path currently implemented in
`Spectral-master/tta/cosa_scheme6.py`. The key change is replacing the old
time-domain correction generator `H_t` with a guarded frequency-domain
correction `DeltaY_t`, while keeping the base model, checkpoints, dataloader,
normalization, and main TTA flow unchanged.

| Formula | Meaning |
|---|---|
| $Y_t^{(0)} = f_\theta(X_t)$ | Frozen base prediction from the pretrained backbone. |
| $R_t = Y_t - Y_t^{(0)}$ | Residual profile / base prediction error. |
| $R_{buf} = \mathrm{MeanRecent}(R_t)$ | Reference residual profile built from recent residual history. |
| $F_y = \mathrm{rFFT}(Y_t^{(0)}, n=L)$ | Frequency representation of the current base prediction. |
| $F_r = \mathrm{rFFT}(w_h \odot R_{buf}, n=L)$ | Reference residual spectrum after Hann windowing. |
| $G_y = \sigma(a_y \odot \lvert F_r \rvert + b_y)$ | One-way local spectral gate generated from the reference spectrum. |
| $Z_t = G_y \odot F_y$ | Gated current prediction spectrum; this is the conservative correction core. |
| $Z_t' = W_o(Z_t)$ | Optional lightweight spectral projection; `core` uses diagonal `W_o`, while `no_out_proj` disables it. |
| $\Delta Y_t = \mathrm{irFFT}(Z_t', n=L)$ | Time-domain correction recovered from the gated spectrum. |
| $s_t = [\mathrm{mean}(Y_t^{(0)}), \lVert F_r \rVert_2, \rho_{low}(F_r)]$ | Global gate features: prediction summary, reference-spectrum energy, and low-frequency ratio. |
| $g_t = \sigma(\mathrm{MLP}_g(s_t))$ | Global correction gate controlling how much correction is applied. |
| $\hat{Y}_t = Y_t^{(0)} + g_t \odot \Delta Y_t$ | Final corrected output with residual addition. |
| $L_{adapt} = \mathrm{MSE}(\hat{Y}_t, Y_t)$ | Adaptation loss for updating only the Scheme 6 adapter parameters. |

Disabled by default:

| Formula | Meaning |
|---|---|
| $G_r \odot F_r$ | Bidirectional reference-content injection. It is disabled by default because it directly injects residual-spectrum content and has higher negative-transfer risk. |

Minimal correction-generator change:

| Version | Correction Generator |
|---|---|
| Before | $H_t = W [Y_t^{(0)} \mid C_t] + b$ |
| After | $\Delta Y_t = \mathrm{irFFT}(G_y \odot \mathrm{rFFT}(Y_t^{(0)}), n=L)$ |

## Collector

- Entrypoint: `collect_scheme6_results.py`
- Reads `Spectral-master/results/summary/`
- Understands `COSA_SCHEME6`, `COSA_SELECT`, and `SIMPLE` result families.

## Migration Notes

The older flat scripts remain for provenance but are no longer the active Scheme
6 entrypoint. Existing datasets, checkpoints, logs, and results are not renamed
or deleted.
