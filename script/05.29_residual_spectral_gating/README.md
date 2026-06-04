# 05.29 Scheme6-B Residual-Conditioned Complex Modulation

![COSA vs Scheme6-B architecture comparison](figures/0604.png)

- Created: `2026-05-29`.
- Previous version: legacy flat `script/` files from `3-versions-final-next-all-in-spectral`.
- Status: `active Scheme6-B probe; old residual-only Scheme 6 halted`.

## Current Direction

The active direction is now Scheme6-B: residual-conditioned complex frequency
modulation. It still reuses COSA's online TTA loop and residual/history buffer,
but the correction module is no longer the original time-domain COSA adapter and
is no longer the first residual-only Scheme 6 gate.

Current main candidate:

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

Mechanism:

| Step | Operation |
|---|---|
| Base prediction | $y = f_\theta(x)$ |
| Prediction spectrum | $F_y = \mathrm{FFT}(y)$ |
| Residual reference | $r = \mathrm{MeanRecent}(y_{true} - y)$ |
| Residual spectrum | $F_r = \mathrm{FFT}(r)$ |
| Complex modulator | $M = M_\theta(F_r)$ |
| Spectral correction | $Z = M(F_r) \odot F_y$ |
| Time-domain delta | $\Delta y = \mathrm{iFFT}(Z)$ |
| Final output | $\hat{y} = y + g \cdot \Delta y$ |

The key change is from `residual spectrum -> correction` to
`residual spectrum -> complex modulator -> prediction spectrum -> correction`.
Because `M` is complex, Scheme6-B can learn amplitude changes and phase shifts;
it is not limited to a sigmoid real-valued attenuation gate.

The current best variant combines the original Spectral-COSA context path with
the Scheme6-B direct complex modulation path:

```text
Spectral context adapter + residual-conditioned complex modulation
```

In the 2026-06-04 seed0 long-horizon check, this variant completed all 42
runnable `pred336/720` points. It beat no-TTA baseline on 42/42 points with
average MSE improvement `+21.63%`, beat COSA on 30/30 comparable points with
`+10.99%`, and beat Spectral-COSA on 27/30 comparable points with `+2.56%`.
The six skipped grid points were missing checkpoints, not runtime failures.

## Difference From COSA And Spectral

| Method | FFT Usage | Correction Path |
|---|---|---|
| COSA | No native FFT correction | Time-domain adapter maps prediction plus context/residual buffer to corrected output. |
| COSA-Spectral | FFT summarizes residual history into spectral context features | Frequency features are appended to context, but final correction is still produced by the time-domain `SimpleOutputAdapter`. |
| Scheme6-B | Uses both $F_y=\mathrm{FFT}(y)$ and $F_r=\mathrm{FFT}(r)$ | Residual spectrum generates a complex modulator $M(F_r)$ that directly modulates the prediction spectrum before inverse FFT reconstruction. |

In short, Spectral is spectral feature augmentation for COSA, while Scheme6-B is
frequency-native correction. The residual spectrum parameterizes a complex
operator applied to the forecast spectrum, instead of merely providing extra
features to a time-domain adapter.

## Old Residual-Only Shutdown Decision

The old residual-only Scheme 6 path is no longer active. Do not launch the full
`6 models x 8 datasets x 4 horizons x 3 seeds` grid for `adapter_residual_only`,
and do not use its partial outputs to fill the paper Excel.

Remote run state and result hygiene:

- No `tmux`, `main.py`, or Scheme 6 runner process was left running when the
  branch was halted.
- The earlier wrong-split `adapter_residual_only` outputs were isolated under
  `/home/normal/3830_FinalProject_COSA-to-Spectral_0604/Spectral-master/results_quarantine/wrong_split_adapter_residual_only_20260604_162541`
  so they should not enter Excel collection.
- The fixed runner protocol was verified against the 5.26 reference on
  `DLinear/ETTh2/pred96/seed0`: no-TTA produced `test_mse=0.1792` and
  `test_mae=0.2846`, matching the 5.26 reference CSV.

Smoke results after the split/protocol fix:

| Variant | Smoke Scope | Baseline Wins | Avg MSE Improvement vs 5.26 no-TTA | Decision |
|---|---:|---:|---:|---|
| `adapter_residual_only` | 4 DLinear `pred96` points | 1 / 4 | `-0.006%` | Failed gate |
| `adapter_residual_force_global_gate_open` | 4 DLinear `pred96` points | 1 / 4 | `-0.194%` | Worse than core |

The forced-open diagnostic confirms that the poor result is not merely caused by
an over-conservative global gate. The core run had `global_gate_mean` around
`0.003-0.011`; the forced-open run reached `0.96-0.99`, and
`direct_delta_norm` increased by roughly two orders of magnitude, but MSE became
worse. This indicates the residual-only correction direction is harmful or at
best too unstable, so it was abandoned in favor of Scheme6-B.

## Change Overview

This folder keeps the conservative residual-only Scheme 6 history and now hosts
the Scheme6-B complex-modulation probe.
It is wired through the main-compatible `COSA_SCHEME6` adapter path instead of
the older `cosa_select.py` path, so future merges into `main` do not depend on a
file that `main` has already replaced.
The historical residual-only method remains available for provenance, but the
active Scheme6-B ablations use `CORRECTION_SOURCE=complex_modulation`.

## Runner

- Entrypoint: `run_residual_spectral_gating.sh`
- Status: active for Scheme6-B smoke/diagnostic runs. Do not run the old
  residual-only full grid.
- Result family: `COSA_SCHEME6`
- Adapter: `Spectral-master/tta/cosa_scheme6.py`
- Default scope: the 5.26-style evaluation grid, `6 models x 8 datasets x 4 horizons x 3 seeds`.
  The runner is checkpoint-only and expects pretrained weights at
  `Spectral-master/checkpoints/seed<seed>/<model>/<dataset>_<pred_len>/checkpoint_best.pth`.
- Default ablations: `core`, `no_out_proj`, `no_global_gate`.
  `core` is residual-only and writes the variant `adapter_residual_only`.
  Optional diagnostics: `prediction_only`, `two_path`, `force_global_gate_open`,
  `complex_modulation`, `complex_modulation_force_open`,
  `complex_mod_desat_gate`, `complex_mod_high_gate_lr`,
  `complex_mod_constant_gate`,
  `context_plus_complex_modulation_high_gate_lr`.
  `force_global_gate_open` keeps the residual-only path but initializes the
  sample-level global gate open and writes
  `adapter_residual_force_global_gate_open`, so it cannot contaminate the main
  `adapter_residual_only` Excel rows.
  `complex_modulation` is the mechanism-B probe: it generates a complex,
  residual-conditioned multiplier `M(Fr)` and applies
  `irFFT(M(Fr) * Fy)`, with zero-output initialization.
  `complex_mod_desat_gate` keeps the gate learnable but initializes it away
  from saturation, `complex_mod_high_gate_lr` additionally gives global-gate
  parameters a higher LR multiplier, and `complex_mod_constant_gate` fixes the
  global gate at a moderate constant for diagnosis.
  `context_plus_complex_modulation_high_gate_lr` keeps the Spectral-COSA
  context adapter active while adding the high-LR complex modulation direct
  correction path; this is the current main Scheme6-B candidate.
- The runner intentionally leaves normalization, PAAS, dataloader tuning, and
  broader runtime knobs aligned with the 5.26 baseline protocol so the measured
  change isolates the Scheme 6 adapter behavior. In particular it explicitly
  sets `DATA.TRAIN_RATIO=0.7`, `DATA.TEST_RATIO=0.2`, `DATA.SCALE=standard`,
  `NORMALIZE=NST`, `NORM_MODULE.ENABLE=False`, and
  `MODEL.channel_independence=True`; do not remove these overrides because
  dataset-specific defaults otherwise reset ETT splits to `0.6`.
- Failed runs are appended to `Spectral-master/results/summary/scheme6_failures.tsv`.
  Set `CONTINUE_ON_ERROR=False` to stop at the first failure, or
  `SKIP_EXISTING=True` to resume without rerunning completed logs.

Useful conservative smoke test:

```bash
DRY_RUN=True MODELS="DLinear" DATASETS="ETTh1" PRED_LENS="96" SEEDS="0 1 2" \
  ABLATIONS="core" bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Gate the historical residual-only method before launching any residual-only full grid:

```bash
bash script/05.29_residual_spectral_gating/smoke_scheme6_vs_baseline.sh
```

By default this runs `DLinear` on `ETTh2 ETTm2 exchange_rate electricity`,
`pred96`, and seeds `0 1 2`, then compares averaged MSE against the official
5.26 `baseline_no_tta` rows. The script exits nonzero and prints
`DO_NOT_RUN_FULL_GRID` unless at least one case beats baseline and the average
MSE improvement is positive. Override `MIN_BASELINE_WINS`,
`MIN_AVG_IMPROVEMENT_PCT`, `MODELS`, `DATASETS`, `PRED_LENS`, or `SEEDS` for a
different gate.

Run only the conservative adapter:

```bash
ABLATIONS="core" bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run the forced-open diagnostic gate:

```bash
ABLATIONS="force_global_gate_open" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run the complex-modulation mechanism probe:

```bash
ABLATIONS="complex_modulation complex_modulation_force_open" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run the complex-modulation gate diagnostics:

```bash
ABLATIONS="complex_mod_desat_gate complex_mod_high_gate_lr complex_mod_constant_gate" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run the current context-plus-complex candidate:

```bash
ABLATIONS="context_plus_complex_modulation_high_gate_lr" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Resume the full default grid without recomputing existing logs:

```bash
SKIP_EXISTING=True ABLATIONS="core" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

## 2026-05-29 Historical Residual-Only Formula Checklist

This is the deprecated minimal conservative Scheme 6 path still implemented in
`Spectral-master/tta/cosa_scheme6.py` for provenance. The key change was replacing the old
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
| $G_r = \sigma(a_r \odot \lvert F_y \rvert + b_r)$ | Residual-content gate generated from the current prediction spectrum. |
| $Z_t = G_r \odot F_r$ | Gated residual spectrum; this is the corrected residual-only core. |
| $Z_t' = W_o(Z_t)$ | Optional lightweight spectral projection; `core` uses diagonal `W_o`, while `no_out_proj` disables it. |
| $\Delta Y_t = \mathrm{irFFT}(Z_t', n=L)$ | Time-domain correction recovered from the gated spectrum. |
| $s_t = [\mathrm{mean}(Y_t^{(0)}), \lVert F_r \rVert_2, \rho_{low}(F_r)]$ | Global gate features: prediction summary, reference-spectrum energy, and low-frequency ratio. |
| $g_t = \sigma(\mathrm{MLP}_g(s_t))$ | Global correction gate controlling how much correction is applied; initialized near zero through a negative final bias. |
| $\hat{Y}_t = Y_t^{(0)} + g_t \odot \Delta Y_t$ | Final corrected output with residual addition. |
| $L_{adapt} = \mathrm{MSE}(\hat{Y}_t, Y_t)$ | Adaptation loss for updating only the Scheme 6 adapter parameters. |

Disabled by default:

| Formula | Meaning |
|---|---|
| $G_y \odot F_y$ | Prediction-spectrum correction path. It is disabled by default because it can only amplify or reshape the base prediction spectrum and was empirically harmful. |
| $G_y \odot F_y + G_r \odot F_r$ | Two-path correction. It is kept as `two_path` after residual-only smoke confirms the residual path is useful. |

Minimal correction-generator change:

| Version | Correction Generator |
|---|---|
| Before | $H_t = W [Y_t^{(0)} \mid C_t] + b$ |
| After | $\Delta Y_t = \mathrm{irFFT}(G_r \odot \mathrm{rFFT}(R_{buf}), n=L)$ |

## Collector

- Entrypoint: `collect_scheme6_results.py`
- Reads `Spectral-master/results/summary/`
- Understands `COSA_SCHEME6`, `COSA_SELECT`, and `SIMPLE` result families.
- Writes per-seed, averaged, and 5.26-comparison CSVs:
  `scheme6_collected_results.csv`,
  `scheme6_collected_results_averaged.csv`,
  `scheme6_improvement_vs_526.csv`.
- Positive `*_improvement_pct` means Scheme 6 is better than the selected 5.26
  reference row for the same `(model, dataset, horizon)`.
- Excel fill-in: `fill_scheme6_excel.py` reads the official 5.26 workbook
  `result/5.26_paper8_experiment/final_tables/paper8_mse_improvement_analysis.xlsx`
  and writes `Spectral-master/results/summary/paper8_mse_improvement_analysis_scheme6.xlsx`
  with V5 MSE, V5-vs-Spectral/F/P/Base comparisons, and a `Scheme6_All`
  sheet for every collected ablation variant.

```bash
python script/05.29_residual_spectral_gating/fill_scheme6_excel.py --require-complete
```

## Migration Notes

The older flat scripts remain for provenance but are no longer the active Scheme
6 entrypoint. Existing datasets, checkpoints, logs, and results are not renamed
or deleted.
