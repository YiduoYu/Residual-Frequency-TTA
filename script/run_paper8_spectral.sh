#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

# Paper grid plus two extra datasets:
# W=96, L in {96,192,336,720}, train:val:test=7:1:2, K=10, S=3.
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-${SEED:-"0 1 2 3 4 5 6 7 8 9"}}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${VISIBLE_DEVICES}}

PAPER_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather"
EXTRA_DATASETS="electricity traffic"
MODELS=${MODELS:-"DLinear FreTS iTransformer MICN OLS PatchTST"}
DATASETS=${DATASETS:-"${PAPER_DATASETS} ${EXTRA_DATASETS}"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}

STAGES=${STAGES:-"train baseline cosa spectral collect"}
COSA_BATCHING=${COSA_BATCHING:-"fixed paas"}
DRY_RUN=${DRY_RUN:-False}
SKIP_EXISTING=${SKIP_EXISTING:-False}

LOOK_BACK=${LOOK_BACK:-96}
LABEL_LEN=${LABEL_LEN:-48}
TRAIN_RATIO=${TRAIN_RATIO:-0.7}
TEST_RATIO=${TEST_RATIO:-0.2}
NUM_WORKERS=${NUM_WORKERS:-2}

CONTEXT_K=${CONTEXT_K:-10}
STEPS=${STEPS:-3}
BATCH_SIZE=${BATCH_SIZE:-48}
PERIOD_N=${PERIOD_N:-1}
BASE_LR=${BASE_LR:-0.005}
MAX_LR=${MAX_LR:-0.005}
MIN_LR=${MIN_LR:-0.0001}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
CONVERGENCE_THRESHOLD=${CONVERGENCE_THRESHOLD:-1e-6}

SPECTRAL_MODE=${SPECTRAL_MODE:-concat}
SPECTRAL_PAAS=${SPECTRAL_PAAS:-False}
SPECTRAL_USAGE=${SPECTRAL_USAGE:-context_only}
SPECTRAL_SOURCE=${SPECTRAL_SOURCE:-residual_profile}
SPECTRAL_REPR=${SPECTRAL_REPR:-real_imag}
SPECTRAL_K=${SPECTRAL_K:-8}
SPECTRAL_HISTORY_SIZE=${SPECTRAL_HISTORY_SIZE:-8}
SPECTRAL_INCLUDE_POWER_RATIO=${SPECTRAL_INCLUDE_POWER_RATIO:-True}
SPECTRAL_INCLUDE_RESIDUAL_STATS=${SPECTRAL_INCLUDE_RESIDUAL_STATS:-True}
SPECTRAL_MIN_POWER_RATIO_FOR_DIRECT=${SPECTRAL_MIN_POWER_RATIO_FOR_DIRECT:-0.20}
USE_CONTEXT_PROJECTION=${USE_CONTEXT_PROJECTION:-True}
CONTEXT_PROJECTION_DIM=${CONTEXT_PROJECTION_DIM:-8}
MAX_CONTEXT_DIM_FOR_FULL_ADAPTER=${MAX_CONTEXT_DIM_FOR_FULL_ADAPTER:-16}
PARAM_BUDGET_MULTIPLIER=${PARAM_BUDGET_MULTIPLIER:-1.25}
ENABLE_DIRECT_CORRECTION=${ENABLE_DIRECT_CORRECTION:-False}

read -r -a MODEL_LIST <<< "${MODELS}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS}"
read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a COSA_BATCH_MODE_LIST <<< "${COSA_BATCHING}"
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
    [[ "${wanted}" == "train" || "${wanted}" == "baseline" || "${wanted}" == "cosa" || "${wanted}" == "spectral" || "${wanted}" == "collect" ]]
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
  csv_path="$(dataset_csv "${dataset}")"
  if ! is_true "${DRY_RUN}" && [ ! -f "${csv_path}" ]; then
    echo "Missing dataset file: ${csv_path}" >&2
    return 1
  fi
}

require_checkpoint() {
  local checkpoint_dir="${1}"
  if ! is_true "${DRY_RUN}" && [ ! -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
    echo "Missing checkpoint: ${checkpoint_dir}/checkpoint_best.pth" >&2
    echo "Run with STAGES=train first, or include train in STAGES." >&2
    return 1
  fi
}

run_to_log() {
  local output="${1}"
  shift
  mkdir -p "$(dirname "${output}")"
  if is_true "${DRY_RUN}"; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf ' > %q\n' "${output}"
  else
    "$@" > "${output}"
  fi
}

run_cmd() {
  if is_true "${DRY_RUN}"; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

spectral_variant() {
  echo "spectral_paper8_${SPECTRAL_SOURCE}_${SPECTRAL_REPR}_hist${SPECTRAL_HISTORY_SIZE}_proj${USE_CONTEXT_PROJECTION}_dim${CONTEXT_PROJECTION_DIM}" | tr ' /' '__'
}

cosa_variant_for_mode() {
  local batch_mode="${1}"
  echo "cosa_paper8_${batch_mode}" | tr ' /' '__'
}

checkpoint_dir_for() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  echo "./checkpoints/seed${seed}/${model}/${dataset}_${pred_len}/"
}

run_train_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local output="./results/summary/PAPER8_TRAIN/${model}/${dataset}/pred${pred_len}_seed${seed}.log"

  require_data "${dataset}"
  if is_true "${SKIP_EXISTING}" && [ -f "${checkpoint_dir}/checkpoint_best.pth" ]; then
    echo "skip train: ${model}/${dataset}/${pred_len}"
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
    TRAIN.ENABLE True \
    TRAIN.CHECKPOINT_DIR "${checkpoint_dir}" \
    TEST.ENABLE False \
    TTA.ENABLE False \
    RESULT_DIR ./results/PAPER8_TRAIN/
}

run_baseline_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local output="./results/summary/BASELINE/${model}/${dataset}/pred${pred_len}_seed${seed}.log"

  require_data "${dataset}"
  require_checkpoint "${checkpoint_dir}"
  if is_true "${SKIP_EXISTING}" && [ -f "./results/BASELINE/seed${seed}/${model}/${dataset}_${pred_len}/test.txt" ]; then
    echo "skip baseline: ${model}/${dataset}/${pred_len}"
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
    TEST.ENABLE True \
    TTA.ENABLE False \
    RESULT_DIR ./results/BASELINE/
}

run_cosa_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local batch_mode="${4}"
  local seed="${5}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local variant_tag
  local paas
  variant_tag="$(cosa_variant_for_mode "${batch_mode}")"

  case "${batch_mode}" in
    fixed) paas=False ;;
    paas) paas=True ;;
    *)
      echo "Unsupported COSA_BATCHING mode: ${batch_mode}" >&2
      return 1
      ;;
  esac

  require_data "${dataset}"
  require_checkpoint "${checkpoint_dir}"

  local output="./results/summary/SIMPLE/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}.log"
  if is_true "${SKIP_EXISTING}" && [ -f "${output}" ]; then
    echo "skip cosa: ${variant_tag}/${model}/${dataset}/${pred_len}"
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
    TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
    TTA.COSA.STEPS "${STEPS}" \
    TTA.COSA.BUFFER_CONTEXT_SIZE "${CONTEXT_K}" \
    TTA.COSA.PAAS "${paas}" \
    TTA.COSA.PERIOD_N "${PERIOD_N}" \
    TTA.COSA.FAST_ADAPTATION True \
    TTA.COSA.ADAPTIVE_LR True \
    TTA.COSA.PER_BATCH_LR_RESET True \
    TTA.COSA.MAX_LR "${MAX_LR}" \
    TTA.COSA.MIN_LR "${MIN_LR}" \
    TTA.COSA.CONVERGENCE_THRESHOLD "${CONVERGENCE_THRESHOLD}" \
    TTA.COSA.ADAPTER_LAYERS 1 \
    TTA.COSA.VAR_WISE_GATING True \
    RESULT_DIR "./results/SIMPLE/${variant_tag}/"
}

run_spectral_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"
  local variant_tag
  variant_tag="$(spectral_variant)"

  require_data "${dataset}"
  require_checkpoint "${checkpoint_dir}"

  local output="./results/summary/COSA_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}.log"
  local log_csv="./results/summary/COSA_SPECTRAL/${variant_tag}/${model}/${dataset}/pred${pred_len}_seed${seed}_batches.csv"
  if is_true "${SKIP_EXISTING}" && [ -f "${output}" ]; then
    echo "skip spectral: ${variant_tag}/${model}/${dataset}/${pred_len}"
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
    TTA.COSA.BATCH_SIZE "${BATCH_SIZE}" \
    TTA.COSA.STEPS "${STEPS}" \
    TTA.COSA.BUFFER_CONTEXT_SIZE "${CONTEXT_K}" \
    TTA.COSA.PAAS "${SPECTRAL_PAAS}" \
    TTA.COSA.PERIOD_N "${PERIOD_N}" \
    TTA.COSA.FAST_ADAPTATION True \
    TTA.COSA.ADAPTIVE_LR True \
    TTA.COSA.PER_BATCH_LR_RESET True \
    TTA.COSA.MAX_LR "${MAX_LR}" \
    TTA.COSA.MIN_LR "${MIN_LR}" \
    TTA.COSA.CONVERGENCE_THRESHOLD "${CONVERGENCE_THRESHOLD}" \
    TTA.COSA.ADAPTER_LAYERS 1 \
    TTA.COSA.VAR_WISE_GATING True \
    TTA.COSA.SPECTRAL.MODE "${SPECTRAL_MODE}" \
    TTA.COSA.SPECTRAL.USAGE "${SPECTRAL_USAGE}" \
    TTA.COSA.SPECTRAL.SOURCE "${SPECTRAL_SOURCE}" \
    TTA.COSA.SPECTRAL.REPR "${SPECTRAL_REPR}" \
    TTA.COSA.SPECTRAL.K "${SPECTRAL_K}" \
    TTA.COSA.SPECTRAL.HISTORY_SIZE "${SPECTRAL_HISTORY_SIZE}" \
    TTA.COSA.SPECTRAL.INCLUDE_POWER_RATIO "${SPECTRAL_INCLUDE_POWER_RATIO}" \
    TTA.COSA.SPECTRAL.INCLUDE_RESIDUAL_STATS "${SPECTRAL_INCLUDE_RESIDUAL_STATS}" \
    TTA.COSA.SPECTRAL.MIN_POWER_RATIO_FOR_DIRECT "${SPECTRAL_MIN_POWER_RATIO_FOR_DIRECT}" \
    TTA.COSA.SPECTRAL.USE_CONTEXT_PROJECTION "${USE_CONTEXT_PROJECTION}" \
    TTA.COSA.SPECTRAL.CONTEXT_PROJECTION_DIM "${CONTEXT_PROJECTION_DIM}" \
    TTA.COSA.SPECTRAL.MAX_CONTEXT_DIM_FOR_FULL_ADAPTER "${MAX_CONTEXT_DIM_FOR_FULL_ADAPTER}" \
    TTA.COSA.SPECTRAL.PARAM_BUDGET_MULTIPLIER "${PARAM_BUDGET_MULTIPLIER}" \
    TTA.COSA.SPECTRAL.ENABLE_DIRECT_CORRECTION "${ENABLE_DIRECT_CORRECTION}" \
    TTA.COSA.SPECTRAL.LOG_CSV_PATH "${log_csv}" \
    RESULT_DIR "./results/COSA_SPECTRAL/${variant_tag}/"
}

run_collect() {
  local variants=("baseline_no_tta")
  local cosa_batch_mode
  for cosa_batch_mode in "${COSA_BATCH_MODE_LIST[@]}"; do
    variants+=("$(cosa_variant_for_mode "${cosa_batch_mode}")")
  done
  variants+=("$(spectral_variant)")

  run_cmd "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/collect_results.py" \
    --root results/summary \
    --baseline-root results/BASELINE \
    --output results/summary/paper8_full_comparison.csv \
    --average-output results/summary/paper8_full_comparison_averaged.csv \
    --families BASELINE SIMPLE COSA_SPECTRAL \
    --datasets "${DATASET_LIST[@]}" \
    --models "${MODEL_LIST[@]}" \
    --horizons "${PRED_LEN_LIST[@]}" \
    --variants "${variants[@]}" \
    --expected-seeds "${SEED_LIST[@]}" \
    --require-complete
}

if has_stage train; then
  for seed in "${SEED_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      for pred_len in "${PRED_LEN_LIST[@]}"; do
        run_train_one "${model}" "${dataset}" "${pred_len}" "${seed}"
      done
    done
  done
  done
fi

if has_stage baseline; then
  for seed in "${SEED_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      for pred_len in "${PRED_LEN_LIST[@]}"; do
        run_baseline_one "${model}" "${dataset}" "${pred_len}" "${seed}"
      done
    done
  done
  done
fi

if has_stage cosa; then
  for seed in "${SEED_LIST[@]}"; do
  for batch_mode in "${COSA_BATCH_MODE_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      for dataset in "${DATASET_LIST[@]}"; do
        for pred_len in "${PRED_LEN_LIST[@]}"; do
          run_cosa_one "${model}" "${dataset}" "${pred_len}" "${batch_mode}" "${seed}"
        done
      done
    done
  done
  done
fi

if has_stage spectral; then
  for seed in "${SEED_LIST[@]}"; do
    for model in "${MODEL_LIST[@]}"; do
      for dataset in "${DATASET_LIST[@]}"; do
        for pred_len in "${PRED_LEN_LIST[@]}"; do
          run_spectral_one "${model}" "${dataset}" "${pred_len}" "${seed}"
        done
      done
    done
  done
fi

if has_stage collect; then
  run_collect
fi
