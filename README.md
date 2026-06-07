# Direct Spectral Adapter Experiment Project

## Direct Spectral Adapter Branch

This branch keeps the original forecasting project infrastructure intact while
adding a standalone Direct Spectral Adapter (DSA) path for test-time adaptation.

DSA uses residual-profile spectral context directly in a lightweight
output-space correction module. It is intentionally separate from the Balanced
Spectral-PEFT experiment family and does not use multi-backend PEFT routing.

## Main Files

- `Spectral-master/tta/direct_spectral.py`: DSA implementation.
- `Spectral-master/tta/direct_adapter_base.py`: DSA adapter base and output
  correction layer.
- `Spectral-master/config.py`: `TTA.DIRECT_SPECTRAL` config surface.
- `Spectral-master/main.py`: DSA routing through `TTA.DIRECT_SPECTRAL.ENABLED`
  or a `DIRECT_SPECTRAL` result directory.
- `Spectral-master/tests/test_direct_spectral_adapter.py`: focused DSA tests.
- `script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh`:
  hard-subset DSA runner.
- `docs/2026-06-06_direct_spectral_adapter_hard_subset_diagnosis.md`:
  hist4 hard-subset diagnosis.

The base model, trainer, predictor, dataset-loader, TAFAS, PETSA, and DynaTTA
paths remain part of the runnable project infrastructure.

## Runner

The DSA hard-subset runner reuses existing trained checkpoints and does not
train base forecasting models.

```bash
DRY_RUN=True \
STAGES=direct_spectral \
MODELS=DLinear \
DATASETS=weather \
PRED_LENS=336 \
SEEDS=0 \
bash script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh
```

The default hard subset is:

- Models: `DLinear`, `FreTS`, `iTransformer`
- Datasets: `weather`, `ETTm1`
- Prediction lengths: `336`, `720`
- Seeds: `0 1 2`

## Validation

Run the focused DSA test before publishing code changes:

```bash
/opt/anaconda3/condabin/conda run -n torch_env python Spectral-master/tests/test_direct_spectral_adapter.py
```

For shell runner changes, use `DRY_RUN=True` and verify that the generated
commands call `TTA.DIRECT_SPECTRAL.ENABLED True` and write under
`results/summary/DIRECT_SPECTRAL/`.

## Artifact Policy

Do not commit datasets, checkpoints, large result logs, generated CSV outputs,
or synced GPU result trees. In particular, keep these local-only:

- `data/`
- `checkpoints/`
- `results/`
- `Spectral-master/data/`
- `Spectral-master/checkpoints/`
- `Spectral-master/results/`
- any `checkpoint_best.pth`

Historical result artifacts are preserved under `result/`.
