# 2026-05-29 Balanced Spectral-PEFT Results

This directory archives the latest GPU Balanced Spectral-PEFT result set for
branch `c/2026-05-29-balanced-spectral-peft`.

The previous smaller balanced package in this directory has been replaced by
the GPU schema6 matrix run collected on 2026-06-08.

## Source

- Remote package root:
  `/home/normal/3830_FinalProject_COSA_latest_plan2_1/balanced_branch_c20260529_schema6_20260608`
- Remote run root:
  `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/COSA_BALANCED`
- Variant: `balanced_branch_c20260529_schema6_20260608`
- Family: `COSA_BALANCED`
- Seeds: `0 1 2`
- Models: `DLinear`, `FreTS`, `iTransformer`, `MICN`, `OLS`, and `PatchTST`
- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`,
  `electricity`, and `traffic`
- Prediction lengths: `96`, `192`, `336`, and `720`

The driver log records 512 completed jobs and 1 skipped job. The skipped job was
already present in the collected outputs, so the final full table contains all
513 seed-level results and the averaged table contains 171 model/dataset/horizon
combinations.

## Files

- `final_tables/balanced_branch_schema6_averaged.csv`
  - Averaged GPU balanced results.
  - 172 lines including the header.
  - Contains 171 combinations, each averaged over 3 seeds.

- `final_tables/balanced_branch_schema6_full_results.csv`
  - Seed-level GPU balanced results.
  - 514 lines including the header.
  - Contains all 513 seed-level rows.

- `input/balanced_schema6_matrix.tsv`
  - The run matrix used for the GPU schema6 balanced run.
  - 172 lines including the header.

- `run_logs/balanced_schema6_matrix_20260608.log`
  - Driver log for the GPU run.
  - Contains 512 `DONE` entries and 1 `SKIP` entry.

- `failure_logs/balanced_schema6_matrix_failures.tsv`
  - Header-only failure log for this archived package.
  - No failed or missing jobs were found in the driver log.

## Headline Metrics

Mean over the 171 averaged combinations:

| Metric | Value |
| --- | ---: |
| Test MSE | 0.316349 |
| Test MAE | 0.360346 |

By model:

| Model | Mean Test MSE | Combinations |
| --- | ---: | ---: |
| DLinear | 0.338402 | 32 |
| FreTS | 0.290866 | 28 |
| iTransformer | 0.296773 | 28 |
| MICN | 0.357781 | 32 |
| OLS | 0.294067 | 27 |
| PatchTST | 0.309335 | 24 |

By dataset:

| Dataset | Mean Test MSE | Combinations |
| --- | ---: | ---: |
| ETTh1 | 0.541307 | 24 |
| ETTh2 | 0.237791 | 24 |
| ETTm1 | 0.455367 | 24 |
| ETTm2 | 0.170715 | 24 |
| electricity | 0.192744 | 19 |
| exchange_rate | 0.267947 | 24 |
| traffic | 0.570054 | 8 |
| weather | 0.238250 | 24 |

By horizon:

| Horizon | Mean Test MSE | Combinations |
| --- | ---: | ---: |
| 96 | 0.237598 | 43 |
| 192 | 0.284648 | 43 |
| 336 | 0.326880 | 43 |
| 720 | 0.418647 | 42 |
