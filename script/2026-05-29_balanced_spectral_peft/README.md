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
