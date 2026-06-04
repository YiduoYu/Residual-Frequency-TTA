# 05.29 Residual Spectral Gating

- Created: `2026-05-29`.
- Previous version: legacy flat `script/` files from `3-versions-final-next-all-in-spectral`.
- Status: `deprecated / halted after 2026-06-04 smoke gate`.

## 2026-06-04 Shutdown Decision

This branch is no longer the active Scheme 6 direction. Do not launch the full
`6 models x 8 datasets x 4 horizons x 3 seeds` grid from this version, and do
not use its partial `adapter_residual_only` outputs to fill the paper Excel.

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
worse. This indicates the current residual spectral correction direction is
harmful or at best too unstable, so the branch is abandoned in favor of a new
scheme.

## Change Overview

This version attempted Scheme 6 as the conservative residual spectral gating path.
It is wired through the main-compatible `COSA_SCHEME6` adapter path instead of
the older `cosa_select.py` path, so future merges into `main` do not depend on a
file that `main` has already replaced.
The default method now uses the historical residual spectrum as the correction
content. It computes one-sided rFFT for the current base prediction and the
residual profile, gates `F_r` with a gate generated from `|F_y|`, then applies a
near-zero-initialized sample-level global gate before residual addition.

Prediction-spectrum correction and the two-path sum remain available only as
explicit ablations through environment variables. They are not the default.

## Runner

- Entrypoint: `run_residual_spectral_gating.sh`
- Status: kept for provenance and reproduction only; do not use for full-grid
  production results.
- Result family: `COSA_SCHEME6`
- Adapter: `Spectral-master/tta/cosa_scheme6.py`
- Default scope: the 5.26-style evaluation grid, `6 models x 8 datasets x 4 horizons x 3 seeds`.
  The runner is checkpoint-only and expects pretrained weights at
  `Spectral-master/checkpoints/seed<seed>/<model>/<dataset>_<pred_len>/checkpoint_best.pth`.
- Default ablations: `core`, `no_out_proj`, `no_global_gate`.
  `core` is residual-only and writes the variant `adapter_residual_only`.
  Optional diagnostics: `prediction_only`, `two_path`, `force_global_gate_open`.
  `force_global_gate_open` keeps the residual-only path but initializes the
  sample-level global gate open and writes
  `adapter_residual_force_global_gate_open`, so it cannot contaminate the main
  `adapter_residual_only` Excel rows.
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

Gate the active residual-only method before launching the full grid:

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

Resume the full default grid without recomputing existing logs:

```bash
SKIP_EXISTING=True ABLATIONS="core" \
  bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
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
