# Improvements

## 2026-05-29 12:00 Asia/Shanghai

- Promoted `script/05.29_residual_spectral_gating/` as the active Scheme 6 script version.
- Default Scheme 6 runner uses the conservative direct residual spectral gating path: unidirectional `Gy * Fy`, Hann residual window, diagonal spectral output projection, global gate enabled, and reference-content injection disabled.
- New result family is `COSA_SCHEME6`; existing datasets, checkpoints, and prior results remain unchanged.
- Ported Scheme 6 to the main-compatible `TTA.COSA.SPECTRAL.*` config namespace and standalone `Spectral-master/tta/cosa_scheme6.py` adapter to avoid future `cosa_select.py` modify/delete merge conflicts.

## 2026-05-29 Adapter Ablation Simplification

- Reduced `run_residual_spectral_gating.sh` to a minimal adapter-only ablation runner.
- Default test is intentionally small: `DLinear`, `ETTh1`, `pred96`, `seed0`.
- Default ablations are `core`, `no_out_proj`, and `no_global_gate`.
- Normalization, PAAS, dataloader tuning, and broad runtime knobs are left at config defaults so the first pass isolates the core adapter behavior.
