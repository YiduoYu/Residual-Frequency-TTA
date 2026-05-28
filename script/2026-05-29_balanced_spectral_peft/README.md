# 2026-05-29 Balanced Spectral-PEFT Code

This folder contains the new Balanced Spectral-PEFT runner and collector.

- `run_paper8_balanced_spectral.sh` uses the new `COSA_BALANCED` result family and the `SpectralBalancedCOSAAdapter`.
- `collect_results.py` understands the balanced spectral summary fields such as window fusion weights, backend weights, spectral energy bands, and adaptation delta norm.
- The original 5.26 paper8 runner remains in `script/5.26/` for provenance, but this folder is the entrypoint for the new balanced spectral version.

Example dry run:

```bash
DRY_RUN=True STAGES=balanced MODELS=DLinear DATASETS=ETTh1 PRED_LENS=96 SEEDS=0 \
  bash script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh
```

## Recommended checkpoint smoke test

The first validation run should reuse the existing trained checkpoints instead
of retraining the base forecasting models. On the course GPU mirror, the
expected checkpoint layout is:

```text
Spectral-master/checkpoints/seed0/<MODEL>/<DATASET>_<PRED_LEN>/checkpoint_best.pth
Spectral-master/checkpoints/seed1/<MODEL>/<DATASET>_<PRED_LEN>/checkpoint_best.pth
```

Before running on the GPU, sync only code and scripts. Do not overwrite or
delete local experiment artifacts:

```bash
rsync -az --delete \
  --exclude='/data' \
  --exclude='/data/**' \
  --exclude='/checkpoints' \
  --exclude='/checkpoints/**' \
  --exclude='/results' \
  --exclude='/results/**' \
  --exclude='/Spectral-master/data' \
  --exclude='/Spectral-master/data/**' \
  --exclude='/Spectral-master/checkpoints' \
  --exclude='/Spectral-master/checkpoints/**' \
  --exclude='/Spectral-master/results' \
  --exclude='/Spectral-master/results/**' \
  ./ normal@10.33.104.12:/home/normal/3830_FinalProject_COSA_latest_plan2_1/
```

For the first real code-result check, run the smallest checkpoint-backed
balanced-only job:

```bash
cd /home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master

SEEDS="0 1" \
STAGES=balanced \
MODELS=DLinear \
DATASETS=ETTh1 \
PRED_LENS=96 \
PYTHON=/home/normal/miniconda3/envs/cosa/bin/python \
bash ../script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh
```

For this smoke test, keep the Balanced Spectral-PEFT method parameters at their
defaults. In particular, do not change `SPECTRAL_NUM_WINDOWS`,
`SPECTRAL_FUSION`, or `BALANCED_BACKENDS` until this two-seed DLinear/ETTh1/96
run succeeds. After that, expand gradually, for example to `MODELS="DLinear
iTransformer"` or to additional prediction horizons.
