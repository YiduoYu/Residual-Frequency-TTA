# Improvements

## 2026-05-29 12:00 Asia/Shanghai

- Promoted `script/05.29_residual_spectral_gating/` as the active Scheme 6 script version.
- Default Scheme 6 runner uses the conservative direct residual spectral gating path: unidirectional `Gy * Fy`, Hann residual window, diagonal spectral output projection, global gate enabled, and reference-content injection disabled.
- New result family is `COSA_SCHEME6`; existing datasets, checkpoints, and prior results remain unchanged.
- Ported Scheme 6 to the main-compatible `TTA.COSA.SPECTRAL.*` config namespace and standalone `Spectral-master/tta/cosa_scheme6.py` adapter to avoid future `cosa_select.py` modify/delete merge conflicts.
