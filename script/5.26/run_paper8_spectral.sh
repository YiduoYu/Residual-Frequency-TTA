#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

# Paper grid plus two extra datasets:
# W=96, L in {96,192,336,720}, train:val:test=7:1:2, K=10, S=3.
PYTHON=${PYTHON:-python}
SEEDS=${SEEDS:-${SEED:-"0 1 2"}}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${VISIBLE_DEVICES}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

PAPER_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather"
EXTRA_DATASETS="electricity traffic"
MODELS=${MODELS:-"DLinear FreTS iTransformer MICN OLS PatchTST"}
DATASETS=${DATASETS:-"${PAPER_DATASETS} ${EXTRA_DATASETS}"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}

STAGES=${STAGES:-"train baseline cosa collect"}
COSA_BATCHING=${COSA_BATCHING:-"fixed paas"}
DRY_RUN=${DRY_RUN:-False}
SKIP_EXISTING=${SKIP_EXISTING:-False}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-True}
COLLECT_REQUIRE_COMPLETE=${COLLECT_REQUIRE_COMPLETE:-False}
FAILURE_LOG=${FAILURE_LOG:-"./results/summary/paper8_failures.tsv"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

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
    [[ "${wanted}" == "train" || "${wanted}" == "baseline" || "${wanted}" == "cosa" || "${wanted}" == "collect" ]]
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
  else
    exit_code=$?
    record_failure "${stage}" "${model}" "${dataset}" "${pred_len}" "${seed}" "${variant}" "${exit_code}"
    if is_true "${CONTINUE_ON_ERROR}"; then
      return 0
    fi
    return "${exit_code}"
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

  require_data "${dataset}" || return
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

  require_data "${dataset}" || return
  require_checkpoint "${checkpoint_dir}" || return
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

  require_data "${dataset}" || return
  require_checkpoint "${checkpoint_dir}" || return

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

run_collect() {
  local variants=("baseline_no_tta")
  local cosa_batch_mode
  for cosa_batch_mode in "${COSA_BATCH_MODE_LIST[@]}"; do
    variants+=("$(cosa_variant_for_mode "${cosa_batch_mode}")")
  done
  local collect_args=(
    "${PYTHON_CMD[@]}" "${SCRIPT_DIR}/collect_results.py"
    --root results/summary \
    --baseline-root results/BASELINE \
    --output results/summary/paper8_full_comparison.csv \
    --average-output results/summary/paper8_full_comparison_averaged.csv \
    --families BASELINE SIMPLE \
    --datasets "${DATASET_LIST[@]}" \
    --models "${MODEL_LIST[@]}" \
    --horizons "${PRED_LEN_LIST[@]}" \
    --variants "${variants[@]}" \
    --expected-seeds "${SEED_LIST[@]}"
  )
  if is_true "${COLLECT_REQUIRE_COMPLETE}"; then
    collect_args+=(--require-complete)
  fi
  run_cmd "${collect_args[@]}"
}

init_failure_log

if has_stage train; then
  for seed in "${SEED_LIST[@]}"; do
  for model in "${MODEL_LIST[@]}"; do
    for dataset in "${DATASET_LIST[@]}"; do
      for pred_len in "${PRED_LEN_LIST[@]}"; do
        run_with_recovery train "${model}" "${dataset}" "${pred_len}" "${seed}" checkpoint \
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
        run_with_recovery baseline "${model}" "${dataset}" "${pred_len}" "${seed}" baseline_no_tta \
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
          run_with_recovery cosa "${model}" "${dataset}" "${pred_len}" "${seed}" "$(cosa_variant_for_mode "${batch_mode}")" \
            run_cosa_one "${model}" "${dataset}" "${pred_len}" "${batch_mode}" "${seed}"
        done
      done
    done
  done
  done
fi

if has_stage collect; then
  run_with_recovery collect - - - - summary run_collect
fi
