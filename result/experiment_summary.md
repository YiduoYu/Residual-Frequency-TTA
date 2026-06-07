# Result Folder Experiment Summary

Updated: 2026-06-07

This document summarizes the purpose, contents, result files, and main findings
under `result/`. Historical file and variant names still use `scheme6`, but the
current presentation/report name for the final method is **ReSpect-TTA**.

## 0. Folder Map

`result/` is the GitHub-facing archive for experiment results. The original
runtime output location is still `Spectral-master/results/`; this folder copies
and organizes the most important files for reporting.

Main folders:

- `result/5.26_paper8_experiment/`
  - Purpose: archive the 5.26 Paper8 baseline/COSA/Spectral-COSA experiments.
  - Main role: establishes no-TTA, COSA fixed, COSA PAAS, and COSA-Spectral
    reference results.
- `result/06.05_scheme6_experiment_records/`
  - Purpose: archive the later residual spectral gating / ReSpect-TTA
    experiments.
  - Main role: records direct complex modulation variants, gate diagnostics,
    long-horizon diagnostics, and the final main ReSpect-TTA candidate.

Important note:

- `spectral_master_results_quarantine/` contains wrong-split residual-only
  outputs. These are kept for auditability but excluded from final conclusions.

## 1. 5.26 Paper8 Experiment

### Purpose

The 5.26 experiment provides the main reference grid before ReSpect-TTA. It
answers:

- How good is the frozen no-TTA baseline?
- How much do original COSA variants improve over no-TTA?
- How much does COSA-Spectral improve when residual FFT features are used only
  as context for a time-domain adapter?

This is the baseline foundation for the later claim that ReSpect-TTA improves
beyond COSA-Spectral.

### Experiment Design

Script:

- `script/5.26/run_paper8_spectral.sh`

Pipeline:

```text
train -> baseline -> cosa -> spectral -> collect
```

Grid:

- Models: `DLinear`, `FreTS`, `iTransformer`, `MICN`, `OLS`, `PatchTST`
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`,
  `electricity`, `traffic`
- Horizons: `96`, `192`, `336`, `720`
- Seeds: `0`, `1`, `2`
- Main compared families:
  - `BASELINE`: `baseline_no_tta`
  - `SIMPLE`: `cosa_paper8_fixed`
  - `SIMPLE`: `cosa_paper8_paas`
  - `COSA_SPECTRAL`:
    `spectral_paper8_residual_profile_real_imag_hist8_projTrue_dim8`

### Key Files

`result/5.26_paper8_experiment/final_tables/`

- `paper8_full_comparison.csv`
  - Per-seed table.
  - 1908 result rows.
- `paper8_full_comparison_averaged.csv`
  - Main averaged table used for reporting.
  - 684 averaged rows.
- `paper8_full_comparison_averaged.xlsx`
  - Excel version of the averaged table.
- `paper8_mse_improvement_analysis.xlsx`
  - MSE improvement workbook.
- `paper8_mse_improvement_analysis_context_plus_complex.xlsx`
  - Later Excel report including context-plus-complex analysis.

`result/5.26_paper8_experiment/failure_logs/`

- `paper8_8datasets_20260527_020659_failures.tsv`
  - Main run failure records.
- `paper8_resume_after_reboot_20260528_120157_failures.tsv`
  - Resume-after-reboot failure records.

`result/5.26_paper8_experiment/run_logs/`

- `paper8_8datasets_20260527_020659.log`
- `paper8_resume_after_reboot_20260528_120157.log`

Other tracking files:

- `MANIFEST.tsv`
- `CHECKSUMS.sha256`

### Coverage

The averaged reporting table contains:

- 171 `BASELINE` averaged conditions
- 171 `COSA_SPECTRAL` averaged conditions
- 171 `cosa_paper8_fixed` averaged conditions
- 171 `cosa_paper8_paas` averaged conditions

Total: 684 averaged rows.

Seed coverage:

- 540 rows have 3 seeds.
- 144 rows have 2 seeds.

### Main Results Against No-TTA Baseline

Using `paper8_full_comparison_averaged.csv`:

| Method | Comparable conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement | Mean MAE improvement |
|---|---:|---:|---:|---:|---:|---:|
| COSA-Spectral | 171 | 156/171 | +11.67% | +7.85% | -8.42% | +5.75% |
| COSA fixed | 171 | 124/171 | +6.28% | +3.41% | -15.75% | +3.73% |
| COSA PAAS | 171 | 113/171 | +4.58% | +2.60% | -19.77% | +2.48% |

### Interpretation

COSA-Spectral is the strongest pre-ReSpect reference. It already shows that
residual frequency information is useful. However, it uses spectrum only as
context for a time-domain adapter. This motivates the later ReSpect-TTA question:
can residual spectrum become a direct correction operator rather than only a
context feature?

### Failure Records

Main run failure log:

- 294 failure rows.
- Stage counts:
  - `train`: 63
  - `baseline`: 63
  - `cosa`: 126
  - `spectral`: 42
- Model counts:
  - `FreTS`: 56
  - `iTransformer`: 56
  - `OLS`: 70
  - `PatchTST`: 112

Resume-after-reboot failure log:

- 63 failure rows.
- All are `spectral` stage failures.
- These mainly reflect missing prior checkpoints or skipped upstream outputs.

## 2. 06.05 ReSpect-TTA / Residual Spectral Gating Experiments

### Purpose

The 06.05 archive explores whether residual spectral information can be used as
a direct correction operator. The current final story is:

> COSA-Spectral uses residual spectrum as context; ReSpect-TTA uses residual
> spectrum as an operator.

This folder includes early failed ideas, gate diagnostics, long-horizon sweeps,
and the final main candidate:

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

### Main Contents

`result/06.05_scheme6_experiment_records/spectral_master_results/`

- `BASELINE/`
  - Copied baseline runtime outputs.
- `COSA_SCHEME6/`
  - Runtime outputs for residual spectral gating / ReSpect-TTA variants.
- `summary/`
  - Main collected CSVs, averaged CSVs, improvement CSVs, logs, and failures.

`result/06.05_scheme6_experiment_records/spectral_master_results_quarantine/`

- 673 files from a wrong-split residual-only run.
- Kept for audit only.
- Excluded from final paper/report conclusions.

### Variant Families Recorded

Major internal variants include:

- `adapter_residual_only`
- `adapter_residual_force_global_gate_open`
- `adapter_complex_modulation`
- `adapter_complex_modulation_force_global_gate_open`
- `adapter_complex_modulation_constant_gate`
- `adapter_complex_modulation_desat_gate`
- `adapter_complex_modulation_high_gate_lr`
- `adapter_complex_modulation_constant_gate_005/010/020`
- `adapter_complex_modulation_high_gate_lr_lowpass_005/010/020`
- `adapter_context_plus_complex_modulation_high_gate_lr`
- `adapter_context_plus_complex_modulation_constant_gate_010/020`
- `adapter_context_plus_complex_modulation_high_gate_lr_floor_005/010`
- `adapter_context_plus_complex_modulation_high_gate_lr_lowpass_010/020`

Not all variants are final candidates. Many are diagnostics.

## 3. Early Residual-Only Smoke Tests

### Purpose

Test whether directly injecting residual spectral information can help before
building the full complex modulation path.

### Key Files

- `scheme6_smoke_results.csv`
- `scheme6_smoke_averaged.csv`
- `scheme6_smoke_improvement_vs_baseline.csv`
- `scheme6_force_open_smoke_results.csv`
- `scheme6_force_open_smoke_averaged.csv`
- `scheme6_force_open_smoke_improvement_vs_baseline.csv`

### Coverage

- Model: `DLinear`
- Datasets: `ETTh2`, `ETTm2`, `electricity`, `exchange_rate`
- Horizon: `96`
- Conditions: 4 averaged conditions
- Seeds: 3

### Results

| Variant | Baseline wins | Mean MSE improvement vs baseline | Mean MAE improvement vs baseline |
|---|---:|---:|---:|
| `adapter_residual_only` | 1/4 | -0.006% | +0.000% |
| `adapter_residual_force_global_gate_open` | 1/4 | -0.194% | -0.166% |

### Conclusion

Residual-only correction is not useful by itself. Forcing the gate open also
does not solve the problem. This supports moving from direct residual injection
to residual-conditioned modulation of the prediction spectrum.

## 4. Direct Complex Modulation Tier-1 Tests

### Purpose

Test whether a complex modulation path alone can improve predictions before
adding the Spectral-COSA context path.

### Key Files

- `scheme6_complex_mod_tier1_results.csv`
- `scheme6_complex_mod_tier1_averaged.csv`
- `scheme6_complex_mod_tier1_improvement.csv`

### Coverage

- 16 averaged rows
- Models: `DLinear`, `PatchTST`, `iTransformer`
- Datasets: `ETTh2`, `ETTm2`, `electricity`, `exchange_rate`, `traffic`,
  `weather`
- Horizon: `96`

### Results

Overall:

| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 16 | 14/16 | +1.80% |
| COSA-Spectral | 16 | 5/16 | -0.99% |
| COSA fixed/PAAS combined | 32 | 12/32 | -0.80% |

By variant:

| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| `adapter_complex_modulation` | baseline | 4 | 4/4 | +0.07% |
| `adapter_complex_modulation` | COSA-Spectral | 4 | 0/4 | -1.80% |
| `adapter_complex_modulation_force_global_gate_open` | baseline | 12 | 10/12 | +2.37% |
| `adapter_complex_modulation_force_global_gate_open` | COSA-Spectral | 12 | 5/12 | -0.73% |

### Conclusion

Direct complex modulation has some signal against no-TTA, but it is not strong
enough against COSA-Spectral. The direct path needs better gating and/or the
Spectral-COSA context path.

## 5. Complex Gate Diagnostics

### Purpose

Test whether the weak direct modulation result is mainly caused by a gate policy
problem.

### Key Files

- `scheme6_complex_gate_diag_results.csv`
- `scheme6_complex_gate_diag_averaged.csv`
- `scheme6_complex_gate_diag_improvement.csv`

### Coverage

- 20 averaged rows
- Models: `DLinear`, `PatchTST`, `iTransformer`
- Horizon: `96`
- Variants: constant gate, desaturated gate, and high gate learning rate

### Results

Overall:

| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 20 | 19/20 | +5.23% |
| COSA-Spectral | 20 | 18/20 | +2.74% |
| COSA fixed/PAAS combined | 40 | 36/40 | +3.08% |

By variant against COSA-Spectral:

| Variant | Conditions | Wins | Mean MSE improvement |
|---|---:|---:|---:|
| `adapter_complex_modulation_constant_gate` | 4 | 4/4 | +4.07% |
| `adapter_complex_modulation_desat_gate` | 4 | 4/4 | +1.84% |
| `adapter_complex_modulation_high_gate_lr` | 12 | 10/12 | +2.59% |

### Conclusion

Gate policy matters. A better gate can turn direct complex modulation from weak
to promising. This justifies later long-horizon gate sweeps and constant/floor
gate diagnostics.

## 6. Long-Horizon Direct-Path Sweeps

### Purpose

Test whether direct complex modulation can handle long-horizon settings without
the Spectral-COSA context path. These sweeps focus on gate size and low-pass
frequency cutoff.

### Key Files

- `scheme6_long_gate_sweep_seed0_results.csv`
- `scheme6_long_gate_sweep_seed0_improvement.csv`
- `scheme6_long_lowpass_sweep_seed0_results.csv`
- `scheme6_long_lowpass_sweep_seed0_improvement.csv`
- `scheme6_tier2_seed0_results.csv`
- `scheme6_tier2_seed0_improvement.csv`

### Gate Sweep Results

Against no-TTA baseline, direct-path variants are mostly positive:

| Variant | Conditions | Wins | Mean MSE improvement vs baseline |
|---|---:|---:|---:|
| `constant_gate_005` | 42 | 42/42 | +4.55% |
| `constant_gate_010` | 42 | 42/42 | +5.71% |
| `constant_gate_020` | 42 | 39/42 | +6.29% |
| `high_gate_lr` | 42 | 40/42 | +6.01% |

Against COSA-Spectral, the same direct-path variants are poor:

| Variant | Conditions | Wins | Mean MSE improvement vs COSA-Spectral |
|---|---:|---:|---:|
| `constant_gate_005` | 30 | 1/30 | -24.22% |
| `constant_gate_010` | 30 | 3/30 | -22.39% |
| `constant_gate_020` | 30 | 4/30 | -21.03% |
| `high_gate_lr` | 30 | 4/30 | -20.82% |

### Low-Pass Sweep Results

Low-pass variants do not rescue the direct-only path:

| Variant | Conditions | Wins vs COSA-Spectral | Mean MSE improvement vs COSA-Spectral |
|---|---:|---:|---:|
| `high_gate_lr_lowpass_005` | 30 | 3/30 | -21.16% |
| `high_gate_lr_lowpass_010` | 30 | 4/30 | -20.90% |
| `high_gate_lr_lowpass_020` | 30 | 4/30 | -20.85% |

### Conclusion

Direct spectral correction alone is not enough for long-horizon settings. It can
beat no-TTA but fails badly against COSA-Spectral. This motivates combining the
direct complex path with the Spectral-COSA context path.

## 7. Context-Plus-Complex Seed0 Breakthrough

### Purpose

Test whether combining Spectral-COSA context features with direct complex
modulation solves the weakness of the direct-only path.

### Key Files

- `scheme6_long_context_plus_complex_seed0_results.csv`
- `scheme6_long_context_plus_complex_seed0_improvement.csv`

### Results

Against seed0 long-horizon references:

| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| direct-only `high_gate_lr` | baseline | 42 | 40/42 | +6.01% |
| direct-only `high_gate_lr` | COSA-Spectral | 30 | 4/30 | -20.82% |
| context-plus-complex `high_gate_lr` | baseline | 42 | 42/42 | +21.63% |
| context-plus-complex `high_gate_lr` | COSA fixed | 30 | 30/30 | +10.99% |
| context-plus-complex `high_gate_lr` | COSA-Spectral | 30 | 27/30 | +2.56% |

### Conclusion

This is the first strong evidence for the final design direction. The direct
complex modulation path becomes useful when it is combined with the
Spectral-COSA context path. This leads to the full 3-seed main run.

## 8. Targeted Pred720 Diagnostics

### Purpose

Focus on hard long-horizon cases, especially ETTm1/ETTm2/ETTh2 at horizon 720
for DLinear, FreTS, OLS, and PatchTST. The goal is to diagnose whether constant
gate or low-pass restrictions improve stability.

### Key Files

- `scheme6_context_plus_targeted_seed0_results.csv`
- `scheme6_context_plus_targeted_seed0_averaged.csv`
- `scheme6_context_plus_targeted_seed0_improvement.csv`
- `scheme6_context_plus_constant010_targeted_results.csv`
- `scheme6_context_plus_constant010_targeted_averaged.csv`
- `scheme6_context_plus_constant010_targeted_improvement.csv`

### Seed0 Targeted Results

| Variant | Reference | Conditions | Wins | Mean MSE improvement |
|---|---|---:|---:|---:|
| `context_plus_constant_gate_010` | baseline | 12 | 12/12 | +21.04% |
| `context_plus_constant_gate_010` | COSA-Spectral | 12 | 8/12 | +1.28% |
| `context_plus_high_gate_lr_lowpass_010` | COSA-Spectral | 12 | 4/12 | -0.57% |
| `context_plus_high_gate_lr_lowpass_020` | COSA-Spectral | 12 | 4/12 | -0.57% |

### 3-Seed Constant Gate 0.10 Targeted Check

| Reference | Conditions | Wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 12 | 12/12 | +20.95% |
| COSA-Spectral | 12 | 10/12 | +1.18% |
| COSA fixed/PAAS combined | 24 | 24/24 | +14.51% |

### Conclusion

The constant gate 0.10 diagnostic is more stable than the low-pass variants on
targeted hard long-horizon cases. Low-pass restrictions are not the main fix.

## 9. Main ReSpect-TTA Full Context-Plus-Complex Run

### Purpose

Evaluate the main candidate across the broad available grid using 3 seeds:

```text
adapter_context_plus_complex_modulation_high_gate_lr
```

This is the main result used in the report and presentation.

### Key Files

- `scheme6_full_context_plus_complex_results.csv`
- `scheme6_full_context_plus_complex_averaged.csv`
- `scheme6_full_context_plus_complex_improvement.csv`
- `scheme6_full_context_plus_complex_20260604_194207.log`
- `scheme6_full_context_plus_complex_20260604_194207_failures.tsv`

### Coverage

- 171 averaged model-dataset-horizon conditions
- 3 seeds for every collected condition
- Models: `DLinear`, `FreTS`, `MICN`, `OLS`, `PatchTST`, `iTransformer`
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`,
  `electricity`, `traffic`
- Horizons: `96`, `192`, `336`, `720`

The theoretical grid is 192 conditions. The collected result covers 171.

Missing 21 conditions:

- `FreTS-traffic`: 96, 192, 336, 720
- `iTransformer-traffic`: 96, 192, 336, 720
- `OLS-traffic`: 96, 192, 336, 720
- `OLS-electricity`: 720
- `PatchTST-electricity`: 96, 192, 336, 720
- `PatchTST-traffic`: 96, 192, 336, 720

### Aggregate Results

| Reference | Comparable conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement | Mean MAE improvement |
|---|---:|---:|---:|---:|---:|---:|
| No-TTA baseline | 171 | 161/171 | +14.37% | +11.13% | -3.48% | +6.81% |
| COSA-Spectral | 171 | 162/171 | +3.07% | +2.72% | -2.54% | +1.14% |
| COSA fixed/PAAS combined | 342 | 332/342 | +9.43% | +6.35% | -1.67% | +3.89% |

### Results by Backbone Against COSA-Spectral

| Backbone | Conditions | MSE wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| DLinear | 32 | 30/32 | +2.70% | -1.79% |
| FreTS | 28 | 25/28 | +2.65% | -1.84% |
| MICN | 32 | 32/32 | +4.86% | +1.15% |
| OLS | 27 | 25/27 | +2.70% | -2.54% |
| PatchTST | 24 | 22/24 | +2.41% | -1.05% |
| iTransformer | 28 | 28/28 | +2.78% | +0.89% |

### Results by Dataset Against COSA-Spectral

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

### Results by Horizon Against COSA-Spectral

| Horizon | Conditions | MSE wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| 96 | 43 | 42/43 | +3.05% | -0.21% |
| 192 | 43 | 43/43 | +3.30% | +0.64% |
| 336 | 43 | 42/43 | +3.25% | -0.36% |
| 720 | 42 | 35/42 | +2.68% | -2.54% |

### Worst Cases Against COSA-Spectral

| Backbone | Dataset | Horizon | MSE improvement |
|---|---|---:|---:|
| OLS | ETTm2 | 720 | -2.54% |
| OLS | ETTm1 | 720 | -2.06% |
| FreTS | ETTm1 | 720 | -1.84% |
| DLinear | ETTm2 | 720 | -1.79% |
| DLinear | ETTm1 | 720 | -1.58% |
| PatchTST | ETTm1 | 720 | -1.05% |

### Best Cases Against COSA-Spectral

| Backbone | Dataset | Horizon | MSE improvement |
|---|---|---:|---:|
| MICN | ETTh1 | 336 | +8.59% |
| OLS | exchange_rate | 720 | +8.54% |
| MICN | ETTh1 | 720 | +8.27% |
| MICN | ETTh2 | 336 | +8.10% |
| DLinear | exchange_rate | 720 | +7.89% |

### Conclusion

The main candidate is the best broad result in the archive. It shows that direct
frequency-domain correction adds value beyond COSA-Spectral's spectral-context
features. Remaining negative transfer is concentrated in ETTm1/ETTm2 at horizon
720, suggesting the next target is gate stabilization for long-horizon drift.

## 10. Constant Gate 0.10 Pred720 Diagnostic

### Purpose

Test whether a fixed conservative gate improves long-horizon stability at
horizon 720.

### Key Files

- `scheme6_context_plus_constant010_pred720_results.csv`
- `scheme6_context_plus_constant010_pred720_averaged.csv`
- `scheme6_context_plus_constant010_pred720_improvement.csv`
- `scheme6_context_plus_constant010_pred720_failures.tsv`

### Coverage

- 42 averaged conditions
- Horizon: `720`
- 3 seeds

### Results

| Reference | Conditions | MSE wins | Mean MSE improvement | Median MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|---:|
| No-TTA baseline | 42 | 42/42 | +28.78% | +23.76% | +2.03% |
| COSA-Spectral | 42 | 40/42 | +3.15% | +3.23% | -0.69% |
| COSA fixed/PAAS combined | 84 | 81/84 | +16.35% | +13.45% | -3.02% |

### Conclusion

A fixed gate of 0.10 is a strong long-horizon diagnostic. It slightly improves
the pred720 slice over the learned high-gate variant in some comparisons, which
shows that long-horizon stability is strongly gate-dependent.

## 11. Learned Gate Floor 0.005 Pred720 Diagnostic

### Purpose

Test whether a learnable gate with a minimum floor can inherit the stability of
a fixed constant gate while preserving adaptivity.

### Key Files

- `scheme6_gate_floor005_pred720_results.csv`
- `scheme6_gate_floor005_pred720_averaged.csv`
- `scheme6_gate_floor005_pred720_improvement.csv`
- `scheme6_gate_floor005_pred720_comparison.csv`

### Results

| Reference | Conditions | MSE wins | Mean MSE improvement |
|---|---:|---:|---:|
| No-TTA baseline | 42 | 42/42 | +28.46% |
| COSA-Spectral | 42 | 35/42 | +2.74% |
| COSA fixed/PAAS combined | 84 | 83/84 | +16.15% |

Direct comparison:

| Comparison | Conditions | Wins | Mean improvement |
|---|---:|---:|---:|
| floor005 vs learned high-gate | 42 | 25/42 | +0.07% |
| floor005 vs constant gate 0.10 | 42 | 17/42 | -0.41% |

### Conclusion

The 0.005 gate floor is useful as a diagnostic but not strong enough to replace
the main learned high-gate variant or the constant 0.10 long-horizon diagnostic.

## 12. MICN Pred720 Gate Diagnostics

### Purpose

MICN behaved especially well with ReSpect-TTA. These experiments test whether
MICN simply needs a larger fixed gate, or whether model-specific adaptivity still
matters.

### Key Files

- `scheme6_context_plus_constant020_micn_pred720_results.csv`
- `scheme6_context_plus_constant020_micn_pred720_averaged.csv`
- `scheme6_context_plus_constant020_micn_pred720_improvement.csv`
- `scheme6_context_plus_constant020_micn_pred720_comparison.csv`
- `scheme6_gate_floor005_micn_pred720_results.csv`
- `scheme6_gate_floor005_micn_pred720_averaged.csv`
- `scheme6_gate_floor005_micn_pred720_improvement.csv`
- `scheme6_gate_floor005_micn_pred720_comparison.csv`
- `scheme6_gate_floor010_micn_pred720_results.csv`
- `scheme6_gate_floor010_micn_pred720_averaged.csv`
- `scheme6_gate_floor010_micn_pred720_improvement.csv`
- `scheme6_gate_floor010_micn_pred720_comparison.csv`

### Coverage

- Model: `MICN`
- Horizon: `720`
- Datasets: all 8 benchmark datasets
- Seeds: 3

### Results Against COSA-Spectral

| Variant | Conditions | Wins | Mean MSE improvement | Worst MSE improvement |
|---|---:|---:|---:|---:|
| constant gate 0.20 | 8 | 8/8 | +5.26% | +3.41% |
| learned gate floor 0.005 | 8 | 8/8 | +5.39% | +1.75% |
| learned gate floor 0.010 | 8 | 8/8 | +5.20% | +0.10% |

Direct comparison to learned high-gate:

| Variant | Conditions | Wins vs high-gate | Mean improvement vs high-gate |
|---|---:|---:|---:|
| constant gate 0.20 | 8 | 3/8 | +0.01% |
| floor 0.005 | 8 | 7/8 | +0.16% |
| floor 0.010 | 8 | 6/8 | -0.03% |

### Conclusion

MICN benefits consistently from ReSpect-TTA-style spectral correction, but the
result is not simply explained by using a larger fixed gate. Learned adaptivity
still matters.

## 13. Failure Logs for ReSpect-TTA Experiments

Important failure files:

- `paper8_failures.tsv`
  - 0 rows in this copied summary.
- `scheme6_0604_core_20260604_154740_failures.tsv`
  - 12 rows, all `adapter_residual_only`.
- `scheme6_failures.tsv`
  - 55 rows across early direct-modulation variants.
- `scheme6_full_context_plus_complex_20260604_194207_failures.tsv`
  - 63 rows for the full main candidate collection.
- `scheme6_context_plus_constant010_pred720_failures.tsv`
  - 18 rows.
- `scheme6_gate_floor005_pred720_failures.tsv`
  - 18 rows.
- MICN-specific gate diagnostics and targeted checks generally have 0 failure
  rows in their listed failure files.

Interpretation:

- Many failures are missing-checkpoint or missing-run effects inherited from the
  incomplete original grid.
- Final reporting should claim the 171 collected comparable conditions, not the
  full theoretical 192 grid.

## 14. What Each Experiment Contributed to the Final Story

| Stage | Purpose | Main result | Decision |
|---|---|---|---|
| 5.26 Paper8 | Build baseline/COSA/COSA-Spectral references | COSA-Spectral beats no-TTA by +11.67% MSE on average | Use COSA-Spectral as strongest reference |
| Residual-only smoke | Test direct residual injection | No improvement vs baseline | Reject residual-only path |
| Direct complex modulation tier1 | Test direct spectral modulation alone | Slightly beats baseline but loses to COSA-Spectral | Not enough alone |
| Gate diagnostics | Test whether gate policy matters | Constant/high-gate variants can beat COSA-Spectral in small settings | Gate is important |
| Long direct-path sweeps | Test direct path at long horizons | Beats no-TTA but loses badly to COSA-Spectral | Need context path |
| Context-plus-complex seed0 | Combine context and direct correction | +2.56% MSE vs COSA-Spectral on seed0 long-horizon subset | Promote to main candidate |
| Full context-plus-complex | Broad 3-seed evaluation | +3.07% MSE vs COSA-Spectral on 171 conditions | Main ReSpect-TTA result |
| Constant gate 0.10 pred720 | Diagnose long-horizon stability | +3.15% MSE vs COSA-Spectral on pred720 | Strong diagnostic |
| Gate floor diagnostics | Test learnable minimum gate | Helpful but not clearly superior | Keep as diagnostic |
| MICN gate diagnostics | Test model-specific gate behavior | All MICN pred720 variants beat COSA-Spectral | MICN is robust; fixed larger gate is not the full explanation |

## 15. Final Takeaway

The result folder records a clear experimental progression:

1. Paper8 establishes that COSA-Spectral is a strong context-only spectral
   baseline.
2. Residual-only and direct-only spectral correction are not sufficient.
3. Gate policy strongly affects whether spectral correction helps or hurts.
4. Combining Spectral-COSA context with residual-conditioned complex modulation
   produces the best broad result.
5. The final ReSpect-TTA candidate improves over no-TTA, COSA, PAAS COSA, and
   COSA-Spectral on most collected conditions.
6. Remaining failures are concentrated in ETTm1/ETTm2 at horizon 720, so future
   work should focus on horizon-aware or state-aware gate stabilization.

For reports and presentations, the most important files are:

- `result/5.26_paper8_experiment/final_tables/paper8_full_comparison_averaged.csv`
- `result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_full_context_plus_complex_averaged.csv`
- `result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_full_context_plus_complex_improvement.csv`
- `result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_context_plus_constant010_pred720_improvement.csv`
- `result/06.05_scheme6_experiment_records/spectral_master_results/summary/scheme6_gate_floor005_pred720_comparison.csv`
