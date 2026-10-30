# Spectral-COSA: Frequency-Domain Context for Test-Time Adaptation in Time Series Forecasting

This branch is one design studied in [Residual-Frequency Test-Time Adaptation for Time-Series Forecasting](https://github.com/YiduoYu/Residual-Frequency-TTA). See the `main` branch for the final method (DSA) and the full comparison. This branch also holds the 8-dataset benchmark pipeline used to train the backbones.

This project extends [COSA](https://openreview.net/forum?id=L7Z5wBMPrW) (ICLR 2026), a lightweight output-space adapter for test-time adaptation (TTA) of time series forecasters. COSA corrects a frozen model's predictions with a gated linear residual, using recent ground-truth statistics as context. **Spectral-COSA** adds the Fourier spectrum of the recent residual profile to that context, so the adapter can see *which frequencies* the base model is currently getting wrong.

Across 8 datasets, 6 backbones, and 4 horizons, Spectral-COSA lowers test MSE over COSA in **143 of 171** configurations, and improves on the non-adaptive baseline by **11.7%** on average (vs. 6.3% for COSA under the same setup).

![Architecture of Spectral-COSA](architecture_of_spectral.drawio.png)

## Method

| Step | Formula | Meaning |
|---|---|---|
| Base prediction | $Y_t^{(0)} = f_{\theta}(X_t)$ | Frozen pretrained backbone |
| Residual | $R_t = Y_t - Y_t^{(0)}$ | Recent base-model error profile |
| Spectral context | $C_t^{spec} = [\mathrm{Re}(\hat{R}), \mathrm{Im}(\hat{R}), p, \mu_R, \sigma_R]$ | FFT of the residual profile plus statistics |
| Context | $C_t = \mathrm{Proj}([C_t^{mean} \mid C_t^{spec}])$ | COSA mean-history context + spectral context |
| Correction | $\hat{Y}_t = Y_t^{(0)} + \tanh(g) \odot (W [Y_t^{(0)} \mid C_t] + b)$ | Gated linear residual |
| Adaptation | $\mathcal{L} = \mathrm{MSE}(\hat{Y}_t, Y_t)$ | Only adapter parameters are updated |

The backbone stays frozen. Spectral features are used as context only (no direct spectral correction), with history size 8 and context dimension 8. The implementation is in [`Spectral-master/tta/cosa_spectral.py`](Spectral-master/tta/cosa_spectral.py).

## Results

Test MSE averaged over 6 backbones (DLinear, FreTS, iTransformer, MICN, OLS, PatchTST) and 4 horizons (96, 192, 336, 720), 3 seeds each. Look-back 96, split 0.7/0.1/0.2.

| Dataset | No TTA | COSA | Spectral-COSA | vs. COSA |
|---|---:|---:|---:|---:|
| ETTh1 | 0.588 | 0.549 | **0.525** | −4.4% |
| ETTh2 | 0.253 | 0.241 | **0.241** | −0.0% |
| ETTm1 | 0.488 | 0.462 | **0.427** | −7.5% |
| ETTm2 | 0.179 | 0.179 | **0.161** | −10.5% |
| Exchange | 0.392 | 0.216 | **0.214** | −0.7% |
| Weather | 0.265 | 0.264 | **0.215** | −18.5% |
| Electricity* | 0.205 | 0.197 | **0.191** | −2.7% |
| Traffic* | 0.590 | 0.572 | **0.570** | −0.3% |

\*Some backbone/horizon pairs on Electricity and Traffic failed during backbone training (mostly GPU out-of-memory on a 32 GB card), leaving 19 and 8 configurations respectively; the failure logs are included. Full per-configuration results, including MAE and standard deviations, are in [`result/5.26_paper8_experiment/final_tables/`](result/5.26_paper8_experiment/final_tables/).

## Running

```bash
cd Spectral-master && pip install -r requirements.txt && cd ..
bash script/5.26/run_paper8_spectral.sh
```

The script runs `train → baseline → cosa → spectral → collect` over the full grid. Useful overrides:

```bash
SEEDS="0 1 2 3 4 5 6 7 8 9" bash script/5.26/run_paper8_spectral.sh   # ten seeds
MODELS="DLinear" DATASETS="ETTh1" PRED_LENS="96" bash script/5.26/run_paper8_spectral.sh
CONTINUE_ON_ERROR=False bash script/5.26/run_paper8_spectral.sh       # stop on first failure
```

Failed jobs are logged to `Spectral-master/results/summary/paper8_failures.tsv` and the grid continues. Datasets go in `Spectral-master/datasets/` (see the [original COSA README](Spectral-master/README.md)).

## Repository structure
```
Spectral-master/          COSA codebase with the Spectral-COSA adapter added
  tta/cosa_spectral.py    Spectral-COSA implementation
script/5.26/              experiment orchestration and result collection
result/                   final tables, run logs, failure logs, checksums
```

## Acknowledgements and license

Built on the official COSA implementation:

> COSA: Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting. ICLR 2026. [OpenReview](https://openreview.net/forum?id=L7Z5wBMPrW)

The COSA code is released under [CC BY-NC-SA 4.0](Spectral-master/LICENSE), and this derivative work is shared under the same license.
