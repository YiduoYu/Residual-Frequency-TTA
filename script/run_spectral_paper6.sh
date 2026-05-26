#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

MODELS=${MODELS:-"DLinear iTransformer"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather"}
PRED_LEN=${PRED_LEN:-96}
SEED=${SEED:-0}
SPECTRAL_HISTORY_SIZE=${SPECTRAL_HISTORY_SIZE:-8}

MODELS="${MODELS}" \
DATASETS="${DATASETS}" \
PRED_LEN="${PRED_LEN}" \
SEED="${SEED}" \
ENABLE_SELECTIVE=False \
SPECTRAL_MODE=concat \
SPECTRAL_USAGE=context_only \
SPECTRAL_SOURCE=residual_profile \
SPECTRAL_REPR=real_imag \
USE_CONTEXT_PROJECTION=True \
CONTEXT_PROJECTION_DIM=8 \
VARIANT_TAG="new_spectral_context_only_realimag_hist${SPECTRAL_HISTORY_SIZE}_seed${SEED}" \
bash "${SCRIPT_DIR}/cosa_select.sh"

# Selective-spectral example:
# ENABLE_SELECTIVE=True SPECTRAL_MODE=concat VARIANT_TAG="selective_spectral_context_only_realimag" bash script/cosa_select.sh
