# 2026-05-29 Balanced Spectral-PEFT Code

This folder contains the new Balanced Spectral-PEFT runner and collector.

- `run_paper8_balanced_spectral.sh` uses the new `COSA_BALANCED` result family and the `SpectralBalancedCOSAAdapter`.
- `collect_results.py` understands the balanced spectral summary fields such as window fusion weights, backend weights, spectral energy bands, and adaptation delta norm.

When launching the experiment from this branch, use only the balanced runner in
this folder:

```bash
bash ../script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh
```

Write outputs under the `COSA_BALANCED` result family and the dated balanced
result archive. This branch intentionally does not keep the old 5.26
Spectral/paper8 runner or result package.

Example dry run:

```bash
DRY_RUN=True STAGES=balanced MODELS=DLinear DATASETS=ETTh1 PRED_LENS=96 SEEDS=0 \
  bash script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh
```

## Recommended balanced-only run

This preset reuses existing trained checkpoints and runs only the Balanced
Spectral-PEFT stage. It does not train base forecasting models, run baseline
evaluation, run simple COSA, or collect the comparison CSV.

The base checkpoints should come from the already completed training runs. Do
not rerun the baseline stage for this preset. The Balanced Spectral-PEFT
optimization results produced here reuse the same trained base-model
checkpoints and selected model/dataset/horizon grid.

The run covers:

- seeds: `0 1 2`
- stage: `balanced`
- models: `DLinear iTransformer FreTS`
- datasets: `ETTh1 ETTm1 weather electricity`
- prediction lengths: `96 192 336 720`
- total jobs: `3 seeds x 3 models x 4 datasets x 4 horizons = 144`

On the course GPU mirror, the expected checkpoint layout is:

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

Run from `Spectral-master`:

```bash
cd /home/normal/3830_FinalProject_COSA_latest_plan2_1/Spectral-master

SEEDS="0 1 2" \
STAGES=balanced \
MODELS="DLinear iTransformer FreTS" \
DATASETS="ETTh1 ETTm1 weather electricity" \
PRED_LENS="96 192 336 720" \
CONTINUE_ON_ERROR=True \
RUN_ID=balanced_3models_4datasets_seed012_electricity \
FAILURE_LOG="./results/summary/balanced_3models_4datasets_seed012_electricity_failures.tsv" \
PYTHON=/home/normal/miniconda3/envs/cosa/bin/python \
bash ../script/2026-05-29_balanced_spectral_peft/run_paper8_balanced_spectral.sh
```

Keep the Balanced Spectral-PEFT method parameters at their defaults for this
run. In particular, leave `SPECTRAL_NUM_WINDOWS`, `SPECTRAL_FUSION`, and
`BALANCED_BACKENDS` unchanged unless the experiment is intentionally testing
method variants.

Outputs are written under:

```text
Spectral-master/results/summary/COSA_BALANCED/<variant>/<MODEL>/<DATASET>/pred<PRED_LEN>_seed<SEED>.log
Spectral-master/results/summary/COSA_BALANCED/<variant>/<MODEL>/<DATASET>/pred<PRED_LEN>_seed<SEED>_batches.csv
Spectral-master/results/COSA_BALANCED/<variant>/
```

Failures are appended to:

```text
Spectral-master/results/summary/balanced_3models_4datasets_seed012_electricity_failures.tsv
```
