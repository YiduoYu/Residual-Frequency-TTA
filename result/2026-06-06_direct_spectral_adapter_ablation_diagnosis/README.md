# Direct Spectral Adapter Ablation Diagnosis

## Scope

This diagnosis only covers the two targeted hard-subset slices used to explain the first DSA gap:

- `ETTm1`, prediction length `336`
- `weather`, prediction length `720`

Each subset ran 5 DSA variants across 3 models and 3 seeds:

- Models: `DLinear`, `FreTS`, `iTransformer`
- Seeds: `0`, `1`, `2`
- Runs per subset: 5 variants x 3 models x 3 seeds = 45 runs

## Run Status

- `ETTm1 336`: 45/45 runs succeeded.
- `weather 720`: 45/45 runs succeeded.
- Failure logs contained no failed runs; both failure logs only had the header row.

## Variants Diagnosed

- `dsa_proj8_default`: current default DSA.
- `dsa_raw19_no_projection`: raw 19-dimensional spectral context without projection.
- `dsa_no_power_ratio`: default projection with power-ratio feature disabled.
- `dsa_no_residual_stats`: default projection with residual mean/scale stats disabled.
- `dsa_concat_mean_context_diagnostic`: diagnostic-only mean-context concat.

## Findings

The default DSA remains the main method.

`dsa_raw19_no_projection` clearly degraded performance. This suggests the projection is not the main source of the old Spectral gap; removing it makes the adapter less stable or less useful.

`dsa_no_power_ratio` and `dsa_no_residual_stats` did not improve the targeted gap. Both features should remain enabled in the default DSA path.

`dsa_concat_mean_context_diagnostic` gave a small improvement on `ETTm1 336`, but hurt `weather 720`. Because it does not improve the gap cleanly and damages a DSA strength case, it should not be adopted as the main method.

## Decision

Keep `dsa_proj8_default` as the DSA baseline and main line.

Do not add new modules, backend mixtures, FiLM, IA3, LoRA, frequency dropout, or energy scaling based on this diagnosis.

## Next Step

Run only DSA hyperparameter diagnosis with single-factor sweeps:

- `projection_dim`: 4, 8, 12, 16
- `K`: 4, 8, 12
- `history_size`: 4, 8, 16

The next sweep should stay on the same targeted slices first: `ETTm1 336` and `weather 720`. Only consider a broader hard-subset run if one setting improves `ETTm1 336` without hurting `weather 720`.
