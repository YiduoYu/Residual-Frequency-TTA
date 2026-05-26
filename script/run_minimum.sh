#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

PYTHON=${PYTHON:-python}
export PYTHON

VARIANT_TAG=cosa_baseline_mean_context \
ENABLE_SELECTIVE=False \
SPECTRAL_MODE=mean \
bash "${SCRIPT_DIR}/cosa_select.sh"

VARIANT_TAG=selective_only_mean_context \
ENABLE_SELECTIVE=True \
SPECTRAL_MODE=mean \
bash "${SCRIPT_DIR}/cosa_select.sh"

VARIANT_TAG=new_spectral_only_residual_profile_realimag \
ENABLE_SELECTIVE=False \
SPECTRAL_MODE=concat \
SPECTRAL_USAGE=context_only \
SPECTRAL_SOURCE=residual_profile \
SPECTRAL_REPR=real_imag \
USE_CONTEXT_PROJECTION=True \
CONTEXT_PROJECTION_DIM=8 \
bash "${SCRIPT_DIR}/cosa_select.sh"

VARIANT_TAG=selective_spectral_residual_profile_realimag \
ENABLE_SELECTIVE=True \
SPECTRAL_MODE=concat \
SPECTRAL_USAGE=context_only \
SPECTRAL_SOURCE=residual_profile \
SPECTRAL_REPR=real_imag \
USE_CONTEXT_PROJECTION=True \
CONTEXT_PROJECTION_DIM=8 \
bash "${SCRIPT_DIR}/cosa_select.sh"

${PYTHON} "${SCRIPT_DIR}/collect_results.py" \
  --root results/summary \
  --output results/summary/minimum_grid_collected_results.csv \
  --families COSA_SELECT \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather \
  --models DLinear iTransformer \
  --variants \
    cosa_baseline_mean_context \
    selective_only_mean_context \
    new_spectral_only_residual_profile_realimag \
    selective_spectral_residual_profile_realimag
