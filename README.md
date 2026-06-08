# Direct Spectral Adapter Branch

Branch: `c/2026-06-06-direct-spectral-adapter`

This branch contains the Direct Spectral Adapter (DSA) experiment family for
time-series test-time adaptation. DSA keeps the pretrained forecasting backbone
frozen and learns a compact output-space correction from recent residual
spectral context.

DSA is intentionally separate from the Balanced Spectral-PEFT branch. It does
not use the balanced multi-backend PEFT mixture, and it does not add FiLM, IA3,
LoRA, backend softmax routing, identity branches, post-bias calibration, or SRDC
modules.

## What This Branch Implements

DSA uses residual profiles from recent adaptation batches as a direct spectral
context signal for the output adapter.

At a high level:

1. The frozen forecasting backbone produces a base prediction
   `Y0 = f_theta(X)`.
2. The adapter records recent residual profiles `R = Y - Y0`.
3. `DirectSpectralContextBuilder` averages the recent residual profiles,
   centers them, runs `rFFT`, and keeps the first `K` non-DC frequency bins as
   real and imaginary features.
4. Optional scalar features add residual power ratio, residual mean, and
   residual scale.
5. `DirectSpectralCorrection` optionally projects the raw context and feeds it
   into a small `DirectOutputAdapter`.
6. The output adapter predicts a gated correction directly on the forecast
   output.

Only the direct adapter parameters are updated during test-time adaptation. The
base forecasting model is not retrained by the DSA runner.

Current default setting:

- spectral bins `K = 8`
- residual history size `4`
- context projection dimension `8`
- residual stats enabled
- power ratio enabled
- mean-context diagnostic disabled

## Key Files

- `Spectral-master/tta/direct_spectral.py`
  - DSA implementation.
  - Defines `DirectSpectralContextBuilder`, `DirectSpectralCorrection`, and
    `DirectSpectralAdapter`.
- `Spectral-master/tta/direct_adapter_base.py`
  - Shared direct-adapter base class and `DirectOutputAdapter`.
- `Spectral-master/config.py`
  - DSA config surface under `TTA.DIRECT_SPECTRAL.*` and
    `TTA.DIRECT_SPECTRAL.ADAPTER.*`.
- `Spectral-master/main.py`
  - Routes to DSA when `TTA.DIRECT_SPECTRAL.ENABLED` is true or the result
    directory uses `DIRECT_SPECTRAL`.
- `Spectral-master/tests/test_direct_spectral_adapter.py`
  - Focused DSA tests and guardrails against balanced-only concepts.
- `script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh`
  - DSA runner for targeted, hard-subset, all-dataset, and paper8-style scopes.
- `docs/2026-06-06_direct_spectral_adapter_hard_subset_diagnosis.md`
  - Early hard-subset diagnosis.
- `result/2026-06-06_direct_spectral_adapter_ablation_diagnosis/`
  - Curated DSA result package.

The base model, trainer, predictor, dataset loader, TAFAS, PETSA, and DynaTTA
paths remain in the repository because they are part of the runnable forecasting
infrastructure.

## Result Packages

The main curated DSA package is:

```text
result/2026-06-06_direct_spectral_adapter_ablation_diagnosis/
```

It contains:

- `dsa_paper8_full_results.csv`
  - Seed-level DSA results.
- `dsa_paper8_full_averaged.csv`
  - Averaged DSA results by variant/model/dataset/horizon.
- `README.md`
  - Diagnosis notes for the targeted ablation slices.

This branch also keeps:

```text
result/5.26_paper8_experiment/
```

That directory is not a DSA method output. It is the archived old-Spectral
reference used for DSA comparison, missing-combination checks, and regression
analysis.

## How To Run

The runner defaults to a small hard subset and reuses existing trained
checkpoints. It does not train base forecasting models.

Dry run:

```bash
DRY_RUN=True \
STAGES=direct_spectral \
MODELS=DLinear \
DATASETS=weather \
PRED_LENS=336 \
SEEDS=0 \
bash script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh
```

Default hard subset:

- Models: `DLinear`, `FreTS`, `iTransformer`
- Datasets: `weather`, `ETTm1`
- Prediction lengths: `336`, `720`
- Seeds: `0 1 2`

Useful scope presets:

- `TARGET_SCOPE=targeted`
- `TARGET_SCOPE=hard_subset`
- `TARGET_SCOPE=all_datasets_3models`
- `TARGET_SCOPE=paper8_full`

Managed GPU runs should use the project conda environment:

```bash
PYTHON=/home/normal/miniconda3/envs/cosa/bin/python \
TARGET_SCOPE=paper8_full \
STAGES=direct_spectral \
CONTINUE_ON_ERROR=True \
bash ../script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh
```

The remote GPU workspace used for these experiments was:

```text
/home/normal/3830_FinalProject_COSA_latest_plan2_1/
```

## Validation

Run the focused DSA test before publishing code changes:

```bash
/opt/anaconda3/condabin/conda run -n torch_env python Spectral-master/tests/test_direct_spectral_adapter.py
```

For shell runner changes, use `DRY_RUN=True` and verify that the generated
commands enable `TTA.DIRECT_SPECTRAL.ENABLED True` and write under:

```text
results/summary/DIRECT_SPECTRAL/
```

## Artifact Policy

Datasets, checkpoints, raw GPU result trees, and generated local scratch outputs
stay local-only. Do not commit:

- `data/`
- `checkpoints/`
- `results/`
- `Spectral-master/data/`
- `Spectral-master/checkpoints/`
- `Spectral-master/results/`
- any `checkpoint_best.pth`

Only curated result packages under `result/` should be committed.

## Relationship To Other Branches

- Direct Spectral Adapter: this branch.
- Balanced Spectral-PEFT: `c/2026-05-29-balanced-spectral-peft`.
- The balanced branch uses multi-window spectral conditioning plus PEFT-style
  backend mixture. DSA removes that mixture and uses a direct residual-spectrum
  context adapter instead.
