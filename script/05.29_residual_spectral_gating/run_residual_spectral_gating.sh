#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-False}

MODELS=${MODELS:-"DLinear"}
DATASETS=${DATASETS:-"ETTh1"}
PRED_LENS=${PRED_LENS:-"96"}
SEEDS=${SEEDS:-"0"}
ABLATIONS=${ABLATIONS:-"core no_out_proj no_global_gate"}

SEQ_LEN=${SEQ_LEN:-96}
LABEL_LEN=${LABEL_LEN:-48}
TTA_BATCH_SIZE=${TTA_BATCH_SIZE:-48}
TTA_STEPS=${TTA_STEPS:-3}
BUFFER_CONTEXT_SIZE=${BUFFER_CONTEXT_SIZE:-10}

SPECTRAL_K=${SPECTRAL_K:-8}
SPECTRAL_HISTORY_SIZE=${SPECTRAL_HISTORY_SIZE:-8}
FFT_PRECISION=${FFT_PRECISION:-fp32}
FFT_LENGTH_MODE=${FFT_LENGTH_MODE:-exact}

read -r -a MODEL_LIST <<< "${MODELS}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS}"
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a ABLATION_LIST <<< "${ABLATIONS}"
read -r -a PYTHON_CMD <<< "${PYTHON}"

is_true() {
  case "${1}" in
    True|true|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

run_to_log() {
  if is_true "${DRY_RUN}"; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@" > "${OUTPUT}"
  fi
}

configure_ablation() {
  SPECTRAL_USAGE=direct
  SPECTRAL_MODE=concat
  GATE_MODE=unidirectional
  USE_GLOBAL_GATE=True
  OUT_PROJ=diagonal
  WINDOW_TYPE=hann
  ENABLE_REFERENCE_CONTENT_INJECTION=False

  case "${1}" in
    core)
      VARIANT_TAG="adapter_core"
      ;;
    no_out_proj)
      OUT_PROJ=none
      VARIANT_TAG="adapter_no_out_proj"
      ;;
    no_global_gate)
      USE_GLOBAL_GATE=False
      VARIANT_TAG="adapter_no_global_gate"
      ;;
    context_plus_core)
      SPECTRAL_USAGE=context_plus_direct
      VARIANT_TAG="adapter_context_plus_core"
      ;;
    *)
      echo "Unknown ablation: ${1}" >&2
      exit 2
      ;;
  esac
}

for ABLATION in "${ABLATION_LIST[@]}"; do
  configure_ablation "${ABLATION}"
  for MODEL in "${MODEL_LIST[@]}"; do
    for DATASET in "${DATASET_LIST[@]}"; do
      for RUN_PRED_LEN in "${PRED_LEN_LIST[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
          CHECKPOINT_DIR="./checkpoints/${MODEL}/${DATASET}_${RUN_PRED_LEN}/"
          RESULT_DIR="./results/COSA_SCHEME6/${VARIANT_TAG}/"
          OUTPUT="./results/summary/COSA_SCHEME6/${VARIANT_TAG}/${MODEL}/${DATASET}/pred${RUN_PRED_LEN}_seed${SEED}.log"
          LOG_CSV="./results/summary/COSA_SCHEME6/${VARIANT_TAG}/${MODEL}/${DATASET}/pred${RUN_PRED_LEN}_seed${SEED}_batches.csv"
          mkdir -p "$(dirname "${OUTPUT}")"

          run_to_log "${PYTHON_CMD[@]}" main.py \
            SEED "${SEED}" \
            DATA.BASE_DIR ./data/ \
            DATA.NAME "${DATASET}" \
            DATA.SEQ_LEN "${SEQ_LEN}" \
            DATA.LABEL_LEN "${LABEL_LEN}" \
            DATA.PRED_LEN "${RUN_PRED_LEN}" \
            MODEL.NAME "${MODEL}" \
            MODEL.seq_len "${SEQ_LEN}" \
            MODEL.label_len "${LABEL_LEN}" \
            MODEL.pred_len "${RUN_PRED_LEN}" \
            TRAIN.ENABLE False \
            TRAIN.CHECKPOINT_DIR "${CHECKPOINT_DIR}" \
            TEST.ENABLE False \
            TTA.ENABLE True \
            TTA.COSA.BATCH_SIZE "${TTA_BATCH_SIZE}" \
            TTA.COSA.STEPS "${TTA_STEPS}" \
            TTA.COSA.BUFFER_CONTEXT_SIZE "${BUFFER_CONTEXT_SIZE}" \
            TTA.COSA.SPECTRAL.MODE "${SPECTRAL_MODE}" \
            TTA.COSA.SPECTRAL.USAGE "${SPECTRAL_USAGE}" \
            TTA.COSA.SPECTRAL.K "${SPECTRAL_K}" \
            TTA.COSA.SPECTRAL.HISTORY_SIZE "${SPECTRAL_HISTORY_SIZE}" \
            TTA.COSA.SPECTRAL.ENABLE_DIRECT_CORRECTION True \
            TTA.COSA.SPECTRAL.GATE_MODE "${GATE_MODE}" \
            TTA.COSA.SPECTRAL.ENABLE_REFERENCE_CONTENT_INJECTION "${ENABLE_REFERENCE_CONTENT_INJECTION}" \
            TTA.COSA.SPECTRAL.USE_GLOBAL_GATE "${USE_GLOBAL_GATE}" \
            TTA.COSA.SPECTRAL.WINDOW_TYPE "${WINDOW_TYPE}" \
            TTA.COSA.SPECTRAL.OUT_PROJ "${OUT_PROJ}" \
            TTA.COSA.SPECTRAL.FFT_PRECISION "${FFT_PRECISION}" \
            TTA.COSA.SPECTRAL.FFT_LENGTH_MODE "${FFT_LENGTH_MODE}" \
            TTA.COSA.SPECTRAL.LOG_CSV_PATH "${LOG_CSV}" \
            RESULT_DIR "${RESULT_DIR}"
        done
      done
    done
  done
done
