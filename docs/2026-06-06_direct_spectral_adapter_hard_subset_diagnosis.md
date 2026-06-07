# Direct Spectral Adapter Hard Subset Diagnosis

## Scope

This note summarizes the Direct Spectral Adapter (DSA) hard-subset diagnosis on:

- Datasets: `ETTm1`, `weather`
- Prediction lengths: `336`, `720`
- Models: `DLinear`, `FreTS`, `iTransformer`
- Seeds: `0`, `1`, `2`

The hard subset has 36 runs per variant: 2 datasets x 2 horizons x 3 models x 3 seeds.

## Reference Results

| method | hard-subset mean MSE | hard-subset mean MAE | delta MSE vs hist4 |
|---|---:|---:|---:|
| DSA hist4 (`dsa_hist4_proj8`) | 0.345393 | 0.368756 | 0.000000 |
| old Spectral reference | 0.348721 | n/a | +0.003328 |
| DSA hist8 (`dsa_hist8_proj8_default`) | 0.365271 | 0.379276 | +0.019878 |
| best balanced no_lora reference | 0.382168 | n/a | +0.036775 |
| default balanced reference | 0.395532 | n/a | +0.050139 |

The first DSA hard-subset run used the hist8 default and reached mean MSE 0.365271 / MAE 0.379276. That already improved over balanced no_lora and default balanced, but still trailed old Spectral.

The hist4 validation reached mean MSE 0.345393 / MAE 0.368756. It improves hist8 by 0.019878 MSE and is also better than the old Spectral hard-subset reference by 0.003328 MSE.

## By Pair

| pair | hist4 MSE | hist4 MAE | hist8 MSE | hist8 MAE | delta MSE |
|---|---:|---:|---:|---:|---:|
| `ETTm1 336` | 0.468791 | 0.465778 | 0.495726 | 0.475476 | -0.026935 |
| `ETTm1 720` | 0.489460 | 0.481357 | 0.502901 | 0.487071 | -0.013442 |
| `weather 336` | 0.212505 | 0.261961 | 0.237152 | 0.278719 | -0.024647 |
| `weather 720` | 0.210817 | 0.265928 | 0.225305 | 0.275836 | -0.014488 |

Hist4 improves all four hard-subset pairs.

## By Model

| model | hist4 MSE | hist4 MAE | hist8 MSE | hist8 MAE | delta MSE |
|---|---:|---:|---:|---:|---:|
| `DLinear` | 0.348369 | 0.367358 | 0.371459 | 0.379762 | -0.023090 |
| `FreTS` | 0.341556 | 0.365954 | 0.362142 | 0.376598 | -0.020586 |
| `iTransformer` | 0.346254 | 0.372955 | 0.362212 | 0.381466 | -0.015958 |

Hist4 improves all three models.

## Interpretation

Short residual spectral memory is beneficial for this DSA path. A shorter `HISTORY_SIZE=4` appears to reduce stale residual context in online test-time adaptation while preserving the useful recent spectral signal.

This diagnosis supports changing the DSA default history size from 8 to 4 and keeping hist8 as a legacy/default-ablation setting.

## Limitations

This conclusion is only validated on the hard subset: `ETTm1/weather x 336/720` with `DLinear`, `FreTS`, and `iTransformer` over seeds 0, 1, and 2.

It does not prove that hist4 is optimal on the full paper8 grid or on every model/dataset/horizon combination. The next broader validation should keep the same DSA structure and compare hist4 against the legacy hist8 setting without adding new modules.
