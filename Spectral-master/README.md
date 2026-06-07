# Direct Spectral Adapter

This directory contains the runnable forecasting code used by the Direct
Spectral Adapter branch.

DSA is implemented as a standalone test-time adaptation path. It builds
spectral context from recent residual profiles, projects that context when
configured, and applies an output-space correction to the frozen base model
prediction.

## Key Paths

- `tta/direct_spectral.py`: DSA context builder, correction module, adapter,
  logging, and summary output.
- `tta/direct_adapter_base.py`: shared DSA adapter base, output correction
  layer, batching helpers, and CSV export helpers.
- `config.py`: `TTA.DIRECT_SPECTRAL` and `TTA.DIRECT_SPECTRAL.ADAPTER`
  defaults.
- `main.py`: runtime routing for `TTA.DIRECT_SPECTRAL.ENABLED True`.
- `tests/test_direct_spectral_adapter.py`: focused DSA unit tests.

## Example Dry Run

From the repository root:

```bash
DRY_RUN=True \
STAGES=direct_spectral \
MODELS=DLinear \
DATASETS=weather \
PRED_LENS=720 \
SEEDS=0 \
bash script/2026-06-06_direct_spectral_adapter/run_direct_spectral_hard_subset.sh
```

The runner emits commands that read existing checkpoints and write summaries
under `results/summary/DIRECT_SPECTRAL/`.

## Focused Test

From the repository root:

```bash
/opt/anaconda3/condabin/conda run -n torch_env python Spectral-master/tests/test_direct_spectral_adapter.py
```

## Local Artifacts

Datasets, checkpoints, generated runtime summaries, and synced GPU artifacts
should remain local-only unless explicitly requested.
