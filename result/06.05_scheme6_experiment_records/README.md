# Scheme6 Experiment Records

Date: 2026-06-05

This directory archives the Scheme6 / residual spectral gating experiment records
for the `2026-05-29-residual-spectral-gating` branch.

## Contents

- `spectral_master_results/`
  - Copy of runtime outputs originally written by `Spectral-master/results/`.
  - Includes raw per-run result folders, per-run configs, `.npy` metrics, logs,
    batch CSVs, and collected summary CSVs.
- `spectral_master_results_quarantine/`
  - Copy of quarantined wrong-split residual-only outputs.
  - Kept for auditability and excluded from the final paper-style Excel tables.

## Key Summary Files

- `spectral_master_results/summary/scheme6_full_context_plus_complex_results.csv`
- `spectral_master_results/summary/scheme6_full_context_plus_complex_averaged.csv`
- `spectral_master_results/summary/scheme6_full_context_plus_complex_improvement.csv`
- `spectral_master_results/summary/scheme6_context_plus_targeted_seed0_results.csv`
- `spectral_master_results/summary/scheme6_context_plus_constant010_targeted_averaged.csv`
- `spectral_master_results/summary/scheme6_context_plus_constant010_targeted_improvement.csv`

## Notes

- Runtime scripts remain under `script/05.29_residual_spectral_gating/`.
- The final Excel report is stored with the paper-style tables under
  `result/5.26_paper8_experiment/final_tables/`.
- This top-level `result/` directory is the GitHub-facing archive location.
  `Spectral-master/results/` remains a local runtime output location.
