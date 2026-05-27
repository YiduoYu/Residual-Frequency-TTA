# COSA to Spectral Experiment Project

## Method Overview: Spectral-COSA

The current Spectral-COSA method follows the same test-time adaptation structure as COSA, but adds residual-profile spectral information to the adapter context. The overall structure is shown below.

![Architecture of Spectral COSA](architecture_of_spectral.drawio.png)

The base forecasting model first produces a frozen base prediction:

$$
Y_t^{(0)} = f_{\theta}(X_t)
$$

where $f_{\theta}$ is the pretrained forecasting backbone. During test-time adaptation, the backbone is not updated. Only the lightweight adapter-related parameters are optimized.

After the base prediction is generated, Spectral-COSA computes the residual profile using the true target and the frozen base prediction:

$$
R_t = Y_t - Y_t^{(0)}
$$

where $Y_t$ is the true $L$-step target and $Y_t^{(0)}$ is the base prediction. This residual profile describes the prediction error pattern of the frozen base model on the current test batch.

The residual profile is then used to build spectral features. In the current implementation, the spectral features are represented as:

$$
C_t^{spectral} = [\mathrm{Re}(\hat{R}), \mathrm{Im}(\hat{R}), p, \mu_R, \sigma_R]
$$

where $\mathrm{Re}(\hat{R})$ and $\mathrm{Im}(\hat{R})$ are the real and imaginary parts of the selected FFT coefficients from the residual profile, $p$ is the residual power-ratio feature, $\mu_R$ is the residual mean, and $\sigma_R$ is the residual standard deviation.

The context buffer stores recent context information and combines the original mean-history context with the spectral residual-profile context. The final adapter context is:

$$
C_t = \mathrm{Proj}([C_t^{mean} \mid C_t^{spectral}])
$$

where $\mathrm{Proj}(\cdot)$ is the context projection module. This projection keeps the effective context dimension small before the context is passed into the adapter.

The adapter input is constructed by concatenating the base prediction with the projected context:

$$
X_t^{(a)} = [Y_t^{(0)} \mid C_t]
$$

The adapter then applies a lightweight linear layer:

$$
H_t = W X_t^{(a)} + b
$$

where $H_t$ is the adapter correction output.

The correction is controlled by a gating function:

$$
g_t = \tanh(g)
$$

The final corrected output is produced by adding the gated adapter correction to the base prediction:

$$
\hat{Y}_t = Y_t^{(0)} + g_t \odot H_t
$$

The corrected output $\hat{Y}_t$ is compared with the true $L$-step target $Y_t$ to compute the adaptation loss:

$$
\mathcal{L}_{adapt} = \mathrm{MSE}(\hat{Y}_t, Y_t)
$$

For evaluation, both MSE and MAE are reported.

The loss updates only the adapter-related parameters during test-time adaptation. The base forecasting model remains frozen throughout this process.

Overall, Spectral-COSA can be understood as a residual-style correction method: the base model provides the main prediction, while the adapter learns a small test-time correction using recent context and residual-profile spectral features.

## 2026-05-27 Experiment Design

This repository keeps the original COSA/Spectral implementation under
`Spectral-master/` and the course experiment orchestration scripts under
`script/5.26/`.

The current main experiment is driven by:

```bash
bash script/5.26/run_paper8_spectral.sh
```

This script runs the paper-style grid plus two additional datasets:

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `exchange_rate`, `weather`,
  `electricity`, and `traffic`
- Models: `DLinear`, `FreTS`, `iTransformer`, `MICN`, `OLS`, and `PatchTST`
- Prediction horizons: `96`, `192`, `336`, and `720`
- Window setup: look-back length `96`, label length `48`
- Split: train/validation/test = `0.7/0.1/0.2`
- Default seeds for this managed run: `0 1 2`

The script executes these stages by default:

```text
train -> baseline -> cosa -> spectral -> collect
```

The comparison includes:

- No-TTA baseline
- COSA with fixed batching
- COSA with PAAS batching
- Spectral-COSA using residual-profile context

For the Spectral-COSA setting, the current design uses spectral information as
context only. It builds residual-profile features, represents them with
`real_imag`, uses history size `8`, projects the context to dimension `8`, and
keeps direct spectral correction disabled.

## Training Reliability Logic

The 2026-05-27 update changes the training script to make long grid runs easier
to manage:

- Default seeds are reduced from ten seeds to `0 1 2` for a tractable managed
  run.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported by default to
  reduce CUDA memory fragmentation.
- `CONTINUE_ON_ERROR=True` lets the grid continue after a failed train,
  baseline, COSA, Spectral-COSA, or collection job.
- Failures are recorded in `Spectral-master/results/summary/paper8_failures.tsv`
  with run id, timestamp, stage, model, dataset, horizon, seed, variant, and
  exit code.
- Result collection allows partial outputs by default. Set
  `COLLECT_REQUIRE_COMPLETE=True` when a strict complete-grid collection is
  required.

Useful overrides:

```bash
# Run the original ten-seed grid.
SEEDS="0 1 2 3 4 5 6 7 8 9" bash script/5.26/run_paper8_spectral.sh

# Stop immediately on the first failed job.
CONTINUE_ON_ERROR=False bash script/5.26/run_paper8_spectral.sh

# Collect only after all expected outputs are present.
COLLECT_REQUIRE_COMPLETE=True bash script/5.26/run_paper8_spectral.sh
```

The original implementation README remains available at
`Spectral-master/README.md`.
