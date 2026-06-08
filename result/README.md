# ReSpect-TTA Result Files

This folder keeps only the cleaned ReSpect-TTA result tables used for the report
and presentation.

## Main Files

```text
respect_tta/respect_tta_main_results.csv
respect_tta/respect_tta_main_averaged.csv
respect_tta/respect_tta_main_improvement.csv
```

Meaning:

```text
results.csv      = collected run-level result table
averaged.csv     = seed-averaged model-dataset-horizon table
improvement.csv  = relative improvement table against reference methods
```

## Diagnostic Files

```text
respect_tta/respect_tta_pred720_constant_gate010_improvement.csv
respect_tta/respect_tta_pred720_gate_floor005_comparison.csv
respect_tta/respect_tta_micn_constant_gate020_comparison.csv
respect_tta/respect_tta_micn_gate_floor010_comparison.csv
```

These files support the presentation claim that the remaining long-horizon
weakness is mostly related to correction gate stability.

## Main Aggregate Result

```text
161/171 wins vs no-TTA baseline
+14.37% mean MSE improvement vs no-TTA baseline
162/171 wins vs context-only spectral reference
+3.07% mean MSE improvement vs context-only spectral reference
```
