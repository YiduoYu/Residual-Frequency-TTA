#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-True}
STAGES=${STAGES:-direct_spectral}
TARGET_SCOPE=${TARGET_SCOPE:-targeted}
VARIANTS=${VARIANTS:-${VARIANT:-dsa_proj8_default}}
SKIP_EXISTING=${SKIP_EXISTING:-False}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-True}
FAILURE_LOG=${FAILURE_LOG:-"./results/summary/direct_spectral_failures.tsv"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
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
DIRECT_SPECTRAL_HISTORY_SIZE=${DIRECT_SPECTRAL_HISTORY_SIZE:-4}
DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM=${DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM:-8}
DIRECT_SPECTRAL_USE_RESIDUAL_STATS=${DIRECT_SPECTRAL_USE_RESIDUAL_STATS:-True}
DIRECT_SPECTRAL_USE_POWER_RATIO=${DIRECT_SPECTRAL_USE_POWER_RATIO:-True}
DIRECT_SPECTRAL_INCLUDE_MEAN_CONTEXT=${DIRECT_SPECTRAL_INCLUDE_MEAN_CONTEXT:-False}
DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE=${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE:-5}

read -r -a VARIANT_LIST <<< "${VARIANTS}"
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

init_failure_log() {
  mkdir -p "$(dirname "${FAILURE_LOG}")"
  if [ ! -f "${FAILURE_LOG}" ]; then
    printf 'run_id\ttimestamp\tstage\tmodel\tdataset\tpred_len\tseed\tvariant\texit_code\n' > "${FAILURE_LOG}"
  fi
}

record_failure() {
  local stage="${1}"
  local model="${2}"
  local dataset="${3}"
  local pred_len="${4}"
  local seed="${5}"
  local variant="${6}"
  local exit_code="${7}"
  init_failure_log
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${RUN_ID}" "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${stage}" "${model}" "${dataset}" "${pred_len}" "${seed}" "${variant}" "${exit_code}" \
    >> "${FAILURE_LOG}"
  echo "failed ${stage}: model=${model} dataset=${dataset} pred_len=${pred_len} seed=${seed} variant=${variant} exit_code=${exit_code}" >&2
}

run_with_recovery() {
  local stage="${1}"
  local model="${2}"
  local dataset="${3}"
  local pred_len="${4}"
  local seed="${5}"
  local variant="${6}"
  shift 6

  local exit_code
  if "$@"; then
    return 0
  fi

  exit_code=$?
  record_failure "${stage}" "${model}" "${dataset}" "${pred_len}" "${seed}" "${variant}" "${exit_code}"
  if is_true "${CONTINUE_ON_ERROR}"; then
    return 0
  fi
  return "${exit_code}"
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

direct_spectral_variant_settings() {
  local variant="${1}"
  case "${variant}" in
    dsa_proj4)
      echo "8 4 4 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_proj8_default)
      echo "8 4 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_proj12)
      echo "8 4 12 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_proj16)
      echo "8 4 16 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_k4_proj8)
      echo "4 4 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_k8_proj8_default)
      echo "8 4 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_k12_proj8)
      echo "12 4 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_hist4_proj8)
      echo "8 4 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_hist8_proj8|dsa_hist8_proj8_default)
      echo "8 8 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_hist16_proj8)
      echo "8 16 8 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_raw19_no_projection)
      echo "8 4 0 True True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_no_power_ratio)
      echo "8 4 8 True False False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_no_residual_stats)
      echo "8 4 8 False True False ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    dsa_concat_mean_context_diagnostic)
      echo "8 4 8 True True True ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    custom)
      echo "${DIRECT_SPECTRAL_K} ${DIRECT_SPECTRAL_HISTORY_SIZE} ${DIRECT_SPECTRAL_CONTEXT_PROJECTION_DIM} ${DIRECT_SPECTRAL_USE_RESIDUAL_STATS} ${DIRECT_SPECTRAL_USE_POWER_RATIO} ${DIRECT_SPECTRAL_INCLUDE_MEAN_CONTEXT} ${DIRECT_SPECTRAL_MEAN_CONTEXT_SIZE}"
      ;;
    *)
      echo "Unsupported DIRECT_SPECTRAL variant: ${variant}" >&2
      return 1
      ;;
  esac
}

is_target_scope_pair() {
  local dataset="${1}"
  local pred_len="${2}"
  case "${TARGET_SCOPE}" in
    targeted)
      case "${dataset}_${pred_len}" in
        ETTm1_336|weather_720) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    hard_subset)
      case "${dataset}_${pred_len}" in
        ETTm1_336|ETTm1_720|weather_336|weather_720) return 0 ;;
        *) return 1 ;;
      esac
      ;;
    *)
      echo "Unsupported TARGET_SCOPE: ${TARGET_SCOPE}" >&2
      return 2
      ;;
  esac
}

validate_target_scope() {
  case "${TARGET_SCOPE}" in
    targeted|hard_subset) return 0 ;;
    *)
      echo "Unsupported TARGET_SCOPE: ${TARGET_SCOPE}" >&2
      return 1
      ;;
  esac
}

run_direct_spectral_one() {
  local variant_tag="${1}"
  local model="${2}"
  local dataset="${3}"
  local pred_len="${4}"
  local seed="${5}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local spectral_k
  local history_size
  local context_projection_dim
  local use_residual_stats
  local use_power_ratio
  local include_mean_context
  local mean_context_size
  read -r spectral_k history_size context_projection_dim use_residual_stats use_power_ratio include_mean_context mean_context_size < <(direct_spectral_variant_settings "${variant_tag}")

  require_data "${dataset}" || return
  require_checkpoint "${checkpoint_dir}" || return

  local output="./results/summary/DIRECT_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}.log"
  local log_csv="./results/summary/DIRECT_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}_batches.csv"
  if is_true "${SKIP_EXISTING}" && [ -f "${output}" ]; then
    echo "skip direct_spectral: ${variant_tag}/${model}/${dataset}/${pred_len}/seed${seed}"
    return
  fi

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
    TTA.DIRECT_SPECTRAL.ADAPTER.BATCH_SIZE "${BATCH_SIZE}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.STEPS "${STEPS}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.PAAS "${DIRECT_SPECTRAL_PAAS}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.PERIOD_N "${PERIOD_N}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.FAST_ADAPTATION True \
    TTA.DIRECT_SPECTRAL.ADAPTER.ADAPTIVE_LR True \
    TTA.DIRECT_SPECTRAL.ADAPTER.PER_BATCH_LR_RESET True \
    TTA.DIRECT_SPECTRAL.ADAPTER.MAX_LR "${MAX_LR}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.MIN_LR "${MIN_LR}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.CONVERGENCE_THRESHOLD "${CONVERGENCE_THRESHOLD}" \
    TTA.DIRECT_SPECTRAL.ADAPTER.ADAPTER_LAYERS 1 \
    TTA.DIRECT_SPECTRAL.ADAPTER.VAR_WISE_GATING True \
    TTA.DIRECT_SPECTRAL.ENABLED True \
    TTA.DIRECT_SPECTRAL.K "${spectral_k}" \
    TTA.DIRECT_SPECTRAL.HISTORY_SIZE "${history_size}" \
    TTA.DIRECT_SPECTRAL.CONTEXT_PROJECTION_DIM "${context_projection_dim}" \
    TTA.DIRECT_SPECTRAL.USE_RESIDUAL_STATS "${use_residual_stats}" \
    TTA.DIRECT_SPECTRAL.USE_POWER_RATIO "${use_power_ratio}" \
    TTA.DIRECT_SPECTRAL.INCLUDE_MEAN_CONTEXT "${include_mean_context}" \
    TTA.DIRECT_SPECTRAL.MEAN_CONTEXT_SIZE "${mean_context_size}" \
    TTA.DIRECT_SPECTRAL.LOG_CSV_PATH "${log_csv}" \
    RESULT_DIR "./results/DIRECT_SPECTRAL/${variant_tag}/"
}

validate_target_scope
init_failure_log

if has_stage direct_spectral; then
  for variant in "${VARIANT_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      for dataset in "${DATASET_LIST[@]}"; do
        for pred_len in "${PRED_LEN_LIST[@]}"; do
          if ! is_target_scope_pair "${dataset}" "${pred_len}"; then
            continue
          fi
          run_with_recovery direct_spectral "${model}" "${dataset}" "${pred_len}" "${seed}" "${variant}" \
            run_direct_spectral_one "${variant}" "${model}" "${dataset}" "${pred_len}" "${seed}"
        done
      done
    done
  done
  done
fi
