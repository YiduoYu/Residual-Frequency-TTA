# Improvements

## 2026-05-29 12:00 Asia/Shanghai

- Promoted `script/05.29_residual_spectral_gating/` as the active Scheme 6 script version.
- Default Scheme 6 runner uses the conservative direct residual spectral gating path: unidirectional `Gy * Fy`, Hann residual window, diagonal spectral output projection, global gate enabled, and reference-content injection disabled.
- New result family is `COSA_SCHEME6`; existing datasets, checkpoints, and prior results remain unchanged.
- Ported Scheme 6 to the main-compatible `TTA.COSA.SPECTRAL.*` config namespace and standalone `Spectral-master/tta/cosa_scheme6.py` adapter to avoid future `cosa_select.py` modify/delete merge conflicts.

## 2026-05-29 Adapter Ablation Simplification

- Reduced `run_residual_spectral_gating.sh` to a minimal adapter-only ablation runner.
- Default test is intentionally small: `DLinear`, `ETTh1`, `pred96`, `seed0`.
- Default ablations are `core`, `no_out_proj`, and `no_global_gate`.
- Normalization, PAAS, dataloader tuning, and broad runtime knobs are left at config defaults so the first pass isolates the core adapter behavior.

## 2026-06-02 00:00 Asia/Shanghai

- Per user directive, kept the active implementation in `script/05.29_residual_spectral_gating/` instead of creating a new `06.02` script version.
- Expanded the default runner grid to match the 5.26 evaluation scope: `DLinear FreTS iTransformer MICN OLS PatchTST`, all 8 datasets, horizons `96 192 336 720`, and seeds `0 1 2`.
- Switched Scheme 6 evaluation to seed-aware checkpoint lookup at `Spectral-master/checkpoints/seed<seed>/<model>/<dataset>_<pred_len>/checkpoint_best.pth`; the runner evaluates existing checkpoints and does not retrain them.
- Added failure logging, resume support through `SKIP_EXISTING=True`, and stderr capture in per-run logs.
- Extended `collect_scheme6_results.py` to produce per-seed, averaged, and 5.26-reference improvement CSVs for listing Scheme 6 uplift after the logs are generated.
- Added `fill_scheme6_excel.py` to populate the official 5.26 `paper8_mse_improvement_analysis.xlsx` template with V5/Scheme6 MSE, comparison formulas, per-model/per-dataset summaries, and all collected Scheme6 ablation variants.

## 2026-06-04 00:00 Asia/Shanghai

- Replaced the default Scheme 6 correction source from prediction-spectrum correction (`Gy * Fy`) to residual-spectrum correction (`Gr * Fr`), so the correction is built from recent residual content rather than from a scaled copy of the base prediction.
- Added `TTA.COSA.SPECTRAL.CORRECTION_SOURCE` with `residual`, `prediction`, and `both` modes; `core` now maps to `adapter_residual_only`, while `prediction_only` and `two_path` are explicit diagnostics.
- Made the global correction gate identity-safe by exposing `GLOBAL_GATE_INIT_BIAS` and defaulting it to `-6.0`, so initial correction strength is close to zero.
- Updated collector metadata and Excel fill defaults so residual-only results are labeled and selected as the main V5/Scheme6 variant.
- Fixed a protocol mismatch in the 05.29 runner by explicitly restoring the 5.26 baseline overrides: `DATA.TRAIN_RATIO=0.7`, `DATA.TEST_RATIO=0.2`, `DATA.SCALE=standard`, `NORMALIZE=NST`, `NORM_MODULE.ENABLE=False`, and `MODEL.channel_independence=True`. Without these overrides, `update_cfg_from_dataset()` resets ETT datasets to `TRAIN_RATIO=0.6`, making Scheme6 outputs incomparable with the 5.26 Excel baseline.
- Added `smoke_scheme6_vs_baseline.sh` and `check_scheme6_smoke_vs_baseline.py` so the active method first runs a small DLinear gate and exits with `DO_NOT_RUN_FULL_GRID` if it cannot beat the official 5.26 no-TTA baseline.
- Added the explicit `force_global_gate_open` diagnostic ablation, writing `adapter_residual_force_global_gate_open`, to test whether the residual spectral correction has useful signal when the sample-level global gate is forced open instead of near-zero initialized.
- Added the mechanism-B probe `complex_modulation`, writing `adapter_complex_modulation`, where a zero-initialized complex multiplier `M(Fr)` is generated from the residual spectrum and applied as `irFFT(M(Fr) * Fy)`. The forced-open diagnostic writes `adapter_complex_modulation_force_global_gate_open`.
- Added gate diagnostics for mechanism B: desaturated learnable gate init, high global-gate LR multiplier, and constant moderate global gate. These diagnose whether the learned gate can open without relying on the forced-open ablation.
- Renamed the current direction to Scheme6-B in the README and documented the distinction from COSA and COSA-Spectral: Spectral uses residual FFT as context features for a time-domain adapter, while Scheme6-B directly applies a residual-conditioned complex operator to the prediction spectrum.
- Added the current main candidate `adapter_context_plus_complex_modulation_high_gate_lr`, which combines the Spectral-COSA context adapter with the Scheme6-B high-gate-LR direct complex modulation path. The 2026-06-04 seed0 long-horizon check completed 42/42 runnable `pred336/720` points and improved over Spectral-COSA on 27/30 comparable points with average MSE improvement `+2.56%`.
- Added complex-modulation frequency cutoff diagnostics and small constant-gate ablations for isolating whether low-frequency complex modulation is enough to keep long-horizon gains stable.

## 2026-06-02 17:52 Asia/Shanghai

- Fixed `collect_scheme6_results.py` path inference for 5.26-style `SIMPLE/<variant>/<model>/<dataset>/pred<horizon>_seed<seed>.log` logs while preserving the older `SIMPLE/<model>/<dataset>/<horizon>.txt` layout.
- Added Scheme 6 summary metadata fallbacks so collected rows retain `spectral_usage`, `spectral_k`, `spectral_repr`, `context_dim`, and `direct_spectral_enabled` for new and existing Scheme 6 logs.
- Declared the `openpyxl` dependency required by `fill_scheme6_excel.py` and added a clear missing-dependency error message.
