# 2026-05-29 Balanced Spectral-PEFT Results

This directory archives the latest Balanced Spectral-PEFT result set for the
branch `c/2026-05-29-balanced-spectral-peft`. It is intentionally separate from
the older `result/5.26_paper8_experiment/` archive so the previous Spectral
baseline remains directly comparable and untouched.

## Source

- Runner: `script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh`
- Remote result root:
  `/home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master/results/summary/`
- Run ID: `balanced_3models_4datasets_seed012_electricity`
- Stage: `balanced`
- Variant:
  `balanced_spectral_peft_multi_window_real_imag_win3_softmax_gate_backendsfilm_ia3_adapter_lora`
- Seeds: `0 1 2`
- Models: `DLinear`, `FreTS`, `iTransformer`
- Datasets: `ETTh1`, `ETTm1`, `weather`, `electricity`
- Prediction lengths: `96`, `192`, `336`, `720`

The failure log contains only the header, so no failed jobs were recorded for
this run. The summary table has 48 rows, covering all
`3 models x 4 datasets x 4 horizons` combinations, each with all 3 seeds.

## Files

- `final_tables/balanced_3models_4datasets_seed012_electricity_balanced_summary.csv`
  - Raw balanced result summary from the remote GPU run.
  - 49 lines including the header.
  - Contains per-combination mean/std MSE and MAE plus seed-level metrics.

- `final_tables/balanced_spectral_comparison_vs_5_26.csv`
  - Balanced results joined against the 5.26 archived average table.
  - 49 lines including the header.
  - Includes baseline, COSA fixed, COSA PaaS, old Spectral MSE/MAE, percent
    improvements, and win flags.

- `failure_logs/balanced_3models_4datasets_seed012_electricity_failures.tsv`
  - Failure log for this run.
  - Header only; no failures recorded.

- `run_logs/balanced_3models_4datasets_seed012_electricity.driver.log`
  - Driver log copied from the remote GPU summary directory.

## Headline Metrics

Mean MSE over the 48 matched combinations:

| Method | Mean MSE | Balanced result |
| --- | ---: | --- |
| Balanced Spectral-PEFT | 0.3526 | current method |
| Baseline | 0.3759 | Balanced improves by 6.11%; 47/48 wins |
| COSA fixed | 0.3628 | Balanced improves by 3.13%; 43/48 wins |
| COSA PaaS | 0.3636 | Balanced improves by 3.39%; 36/48 wins |
| 5.26 old Spectral | 0.3340 | Balanced is lower by 5.55%; 16/48 wins |

## Breakdown

By model:

| Model | Balanced MSE | Baseline MSE | Old Spectral MSE | vs Baseline | vs Old Spectral |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLinear | 0.3599 | 0.3801 | 0.3382 | +5.22% | -6.85% |
| FreTS | 0.3503 | 0.3692 | 0.3342 | +5.19% | -4.75% |
| iTransformer | 0.3476 | 0.3786 | 0.3296 | +7.92% | -5.05% |

By dataset:

| Dataset | Balanced MSE | Baseline MSE | Old Spectral MSE | vs Baseline | vs Old Spectral |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETTh1 | 0.5246 | 0.5568 | 0.5089 | +5.33% | -2.77% |
| ETTm1 | 0.4495 | 0.4720 | 0.4192 | +4.52% | -6.95% |
| electricity | 0.1974 | 0.2098 | 0.1940 | +5.67% | -1.55% |
| weather | 0.2390 | 0.2651 | 0.2138 | +8.94% | -10.93% |

By horizon:

| Horizon | Balanced MSE | Baseline MSE | Old Spectral MSE | vs Baseline | vs Old Spectral |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96 | 0.2834 | 0.2974 | 0.2791 | +4.79% | -0.76% |
| 192 | 0.3298 | 0.3401 | 0.3253 | +3.39% | -1.09% |
| 336 | 0.3698 | 0.3889 | 0.3586 | +5.15% | -4.26% |
| 720 | 0.4275 | 0.4773 | 0.3731 | +11.12% | -16.09% |

## Analysis

Balanced Spectral-PEFT is stable in this run and clearly improves over the
baseline and simple COSA variants. The best interpretation is that the balanced
adapter is effective as a test-time adaptation method.

It does not beat the older 5.26 Spectral variant overall. The gap is not uniform:
short horizons are close to old Spectral, with only -0.76% at horizon 96 and
-1.09% at horizon 192. The main weakness appears at longer horizons, especially
horizon 720, where Balanced trails old Spectral by 16.09% on average.

The largest regressions against old Spectral are concentrated in weather and
long-horizon ETTm1:

| Model | Dataset | Horizon | Balanced MSE | Old Spectral MSE | vs Old Spectral |
| --- | --- | ---: | ---: | ---: | ---: |
| DLinear | weather | 720 | 0.3120 | 0.2232 | -39.79% |
| FreTS | weather | 720 | 0.2935 | 0.2247 | -30.61% |
| iTransformer | weather | 720 | 0.2878 | 0.2244 | -28.28% |
| DLinear | weather | 336 | 0.2846 | 0.2334 | -21.92% |
| DLinear | ETTm1 | 720 | 0.5340 | 0.4654 | -14.74% |

The strongest improvements against old Spectral are smaller and mostly at short
or mid horizons:

| Model | Dataset | Horizon | Balanced MSE | Old Spectral MSE | vs Old Spectral |
| --- | --- | ---: | ---: | ---: | ---: |
| FreTS | weather | 96 | 0.1713 | 0.1766 | +3.00% |
| DLinear | ETTh1 | 336 | 0.5330 | 0.5466 | +2.49% |
| DLinear | ETTh1 | 192 | 0.4892 | 0.5014 | +2.45% |
| FreTS | ETTh1 | 336 | 0.5299 | 0.5427 | +2.36% |
| FreTS | ETTh1 | 192 | 0.4807 | 0.4913 | +2.16% |

For the next experiment, the most useful target is long-horizon stability:
tune or ablate the spectral fusion/backend gate for horizon 336/720, with
weather and ETTm1 as the primary diagnostic datasets.

