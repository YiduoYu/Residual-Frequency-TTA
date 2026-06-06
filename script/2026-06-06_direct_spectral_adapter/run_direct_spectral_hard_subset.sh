#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-True}
STAGES=${STAGES:-direct_spectral}
MODELS=${MODELS:-"DLinear FreTS iTransformer"}
DATASETS=${DATASETS:-"weather ETTm1"}
PRED_LENS=${PRED_LENS:-"336 720"}
SEEDS=${SEEDS:-"0 1 2"}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${VISIBLE_DEVICES}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOOK_BACK=${LOOK_BACK:-96}
LABEL_LEN=${LABEL_LEN:-48}
TRAIN_RATIO=${TRAIN_RATIO:-0.7}
TEST_RATIO=${TEST_RATIO:-0.2}
NUM_WORKERS=${NUM_WORKERS:-2}

STEPS=${STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-48}
PERIOD_N=${PERIOD_N:-1}
BASE_LR=${BASE_LR:-0.005}
MAX_LR=${MAX_LR:-0.005}
MIN_LR=${MIN_LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
CONVERGENCE_THRESHOLD=${CONVERGENCE_THRESHOLD:-1e-6}
DIRECT_SPECTRAL_PAAS=${DIRECT_SPECTRAL_PAAS:-False}
DIRECT_SPECTRAL_K=${DIRECT_SPECTRAL_K:-8}
DIRECT_SPECTRAL_HISTORY_SIZE=${DIRECT_SPECTRAL_HISTORY_SIZE:-8}
DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM=${DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM:-8}
DIRECT_SPECTRAL_USE_RESIDUAL_STATS=${DIRECT_SPECTRAL_USE_RESIDUAL_STATS:-True}
DIRECT_SPECTRAL_USE_POWER_RATIO=${DIRECT_SPECTRAL_USE_POWER_RATIO:-True}

read -r -a MODEL_LIST <<< "${MODELS}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS}"
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a PYTHON_CMD <<< "${PYTHON}"

is_true() {
  case "${1}" in
    True|true|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

has_stage() {
  local wanted="${1}"
  if [[ " ${STAGES} " == *" all "* ]]; then
    [[ "${wanted}" == "direct_spectral" ]]
    return
  fi
  [[ " ${STAGES} " == *" ${wanted} "* ]]
}

dataset_csv() {
  local dataset="${1}"
  case "${dataset}" in
    ETTh1|ETTh2|ETTm1|ETTm2|exchange_rate|weather|electricity|traffic)
      echo "./data/${dataset}/${dataset}.csv"
      ;;
    *)
      echo "Unsupported dataset: ${dataset}" >&2
      return 1
      ;;
  esac
}

require_data() {
  local dataset="${1}"
  local csv_path
  csv_path="$(dataset_csv "${dataset}")" || return
  if ! is_true "${DRY_RUN}" && [ ! -f "${csv_path}" ]; then
    echo "Missing dataset file: ${csv_path}" >&2
    return 1
  fi
}

checkpoint_dir_for() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  echo "./checkpoints/seed${seed}/${model}/${dataset}_${pred_len}/"
}

require_checkpoint() {
  local checkpoint_dir="${1}"
  if ! is_true "${DRY_RUN}" && [ ! -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
    echo "Missing checkpoint: ${checkpoint_dir}/checkpoint_best.pth" >&2
    echo "This DSA hard-subset runner does not train checkpoints." >&2
    return 1
  fi
}

run_to_log() {
  local output="${1}"
  shift
  if is_true "${DRY_RUN}"; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf ' > %q\n' "${output}"
  else
    mkdir -p "$(dirname "${output}")"
    "$@" > "${output}"
  fi
}

direct_spectral_variant() {
  echo "dsa_k${DIRECT_SPECTRAL_K}_hist${DIRECT_SPECTRAL_HISTORY_SIZE}_proj${DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM}_stats${DIRECT_SPECTRAL_USE_RESIDUAL_STATS}_power${DIRECT_SPECTRAL_USE_POWER_RATIO}" | tr ' /,' '___'
}

run_direct_spectral_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local variant_tag
  variant_tag="$(direct_spectral_variant)"

  require_data "${dataset}" || return
  require_checkpoint "${checkpoint_dir}" || return

  local output="./results/summary/DIRECT_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}.log"
  local log_csv="./results/summary/DIRECT_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}_batches.csv"

  run_to_log "${output}" "${PYTHON_CMD[@]}" main.py \
    SEED "${seed}" \
    VISIBLE_DEVICES "${VISIBLE_DEVICES}" \
    DATA_LOADER.NUM_WORKERS "${NUM_WORKERS}" \
    DATA.BASE_DIR ./data/ \
    DATA.NAME "${dataset}" \
    DATA.SEQ_LEN "${LOOK_BACK}" \
    DATA.LABEL_LEN "${LABEL_LEN}" \
    DATA.PRED_LEN "${pred_len}" \
    DATA.TRAIN_RATIO "${TRAIN_RATIO}" \
    DATA.TEST_RATIO "${TEST_RATIO}" \
    DATA.SCALE standard \
    NORMALIZE NST \
    NORM_MODULE.ENABLE False \
    MODEL.NAME "${model}" \
    MODEL.seq_len "${LOOK_BACK}" \
    MODEL.label_len "${LABEL_LEN}" \
    MODEL.pred_len "${pred_len}" \
    MODEL.channel_independence True \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${checkpoint_dir}" \
    TEST.ENABLE False \
    TTA.ENABLE True \
    TTA.SOLVER.BASE_LR "${BASE_LR}" \
    TTA.SOLVER.WEIGHT_DECAY "${WEIGHT_DECAY}" \
    TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
    TTA.COSA.STEPS "${STEPS}" \
    TTA.COSA.PAAS "${DIRECT_SPECTRAL_PAAS}" \
    TTA.COSA.PERIOD_N "${PERIOD_N}" \
    TTA.COSA.FAST_ADAPTATION True \
    TTA.COSA.ADAPTIVE_LR True \
    TTA.COSA.PER_BATCH_LR_RESET True \
    TTA.COSA.MAX_LR "${MAX_LR}" \
    TTA.COSA.MIN_LR "${MIN_LR}" \
    TTA.COSA.CONVERGENCE_THRESHOLD "${CONVERGENCE_THRESHOLD}" \
    TTA.COSA.ADAPTER_LAYERS 1 \
    TTA.COSA.VAR_WISE_GATING True \
    TTA.DIRECT_SPECTRAL.ENABLED True \
    TTA.DIRECT_SPECTRAL.K "${DIRECT_SPECTRAL_K}" \
    TTA.DIRECT_SPECTRAL.HISTORY_SIZE "${DIRECT_SPECTRAL_HISTORY_SIZE}" \
    TTA.DIRECT_SPECTRAL.CONTEXT_PROJECTION_DIM "${DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM}" \
    TTA.DIRECT_SPECTRAL.USE_RESIDUAL_STATS "${DIRECT_SPECTRAL_USE_RESIDUAL_STATS}" \
    TTA.DIRECT_SPECTRAL.USE_POWER_RATIO "${DIRECT_SPECTRAL_USE_POWER_RATIO}" \
    TTA.DIRECT_SPECTRAL.LOG_CSV_PATH "${log_csv}" \
    RESULT_DIR "./results/DIRECT_SPECTRAL/${variant_tag}/"
}

if has_stage direct_spectral; then
  for seed in "${SEED_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      for dataset in "${DATASET_LIST[@]}"; do
        for pred_len in "${PRED_LEN_LIST[@]}"; do
          run_direct_spectral_one "${model}" "${dataset}" "${pred_len}" "${seed}"
        done
      done
    done
  done
fi
