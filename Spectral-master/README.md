# ReSpect-TTA Runtime

This directory contains the forecasting backbones and the ReSpect-TTA adapter used in the final project.

## Main Files

- `main.py`: training, testing, and online adaptation entrypoint.
- `config.py`: default experiment configuration, including `TTA.RESPECT`.
- `tta/respect_tta.py`: ReSpect-TTA adapter with residual spectral correction.
- `tta/online_adapter_base.py`: shared online adapter utilities.
- `tta/residual_spectral_context.py`: residual-memory spectral context builder.
- `scripts/train.sh` and `scripts/test.sh`: base training and testing helpers.

## Running ReSpect-TTA

Use the project-level script:

```bash
bash ../script/respect_tta/run_respect_tta.sh
```

The frozen forecasting backbone produces a base prediction, ReSpect-TTA builds a recent residual spectrum from online errors, learns a complex spectral modulator, applies the correction in frequency space, and gates the final residual correction before adding it back to the base forecast.
