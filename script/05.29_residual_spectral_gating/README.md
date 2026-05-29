# 05.29 Residual Spectral Gating

- Created: `2026-05-29`.
- Previous version: legacy flat `script/` files from `3-versions-final-next-all-in-spectral`.
- Status: `active`.

## Change Overview

This version deploys Scheme 6 as the conservative residual spectral gating path.
It is wired through the main-compatible `COSA_SCHEME6` adapter path instead of
the older `cosa_select.py` path, so future merges into `main` do not depend on a
file that `main` has already replaced.
The default method uses one-sided rFFT on the current base prediction and the
historical residual profile, applies a unidirectional residual-spectrum gate
`Gy * Fy`, then applies a sample-level global gate before residual addition.

The bidirectional reference-content path remains available only as an explicit
ablation through environment variables. It is not the default.

## Runner

- Entrypoint: `run_residual_spectral_gating.sh`
- Result family: `COSA_SCHEME6`
- Adapter: `Spectral-master/tta/cosa_scheme6.py`
- Default scope: one small adapter ablation on `DLinear / ETTh1 / pred96 / seed0`.
- Default ablations: `core`, `no_out_proj`, `no_global_gate`.
- The runner intentionally leaves normalization, PAAS, dataloader tuning, and
  broader training/runtime knobs at config defaults. First-pass testing should
  isolate the core adapter behavior.

Useful conservative smoke test:

```bash
DRY_RUN=True bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

Run only the conservative adapter:

```bash
ABLATIONS="core" bash script/05.29_residual_spectral_gating/run_residual_spectral_gating.sh
```

## Collector

- Entrypoint: `collect_scheme6_results.py`
- Reads `Spectral-master/results/summary/`
- Understands `COSA_SCHEME6`, `COSA_SELECT`, and `SIMPLE` result families.

## Migration Notes

The older flat scripts remain for provenance but are no longer the active Scheme
6 entrypoint. Existing datasets, checkpoints, logs, and results are not renamed
or deleted.
