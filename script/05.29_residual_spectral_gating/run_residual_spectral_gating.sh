#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SPECTRAL_ROOT="${PROJECT_ROOT}/Spectral-master"
cd "${SPECTRAL_ROOT}"

PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-False}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-True}
SKIP_EXISTING=${SKIP_EXISTING:-False}
FAILURE_LOG=${FAILURE_LOG:-"./results/summary/scheme6_failures.tsv"}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}

MODELS=${MODELS:-"DLinear FreTS iTransformer MICN OLS PatchTST"}
DATASETS=${DATASETS:-"ETTh1 ETTh2 ETTm1 ETTm2 exchange_rate weather electricity traffic"}
PRED_LENS=${PRED_LENS:-"96 192 336 720"}
SEEDS=${SEEDS:-"0 1 2"}
ABLATIONS=${ABLATIONS:-"core no_out_proj no_global_gate"}

SEQ_LEN=${SEQ_LEN:-96}
LABEL_LEN=${LABEL_LEN:-48}
TRAIN_RATIO=${TRAIN_RATIO:-0.7}
TEST_RATIO=${TEST_RATIO:-0.2}
TTA_BATCH_SIZE=${TTA_BATCH_SIZE:-48}
TTA_STEPS=${TTA_STEPS:-3}
BUFFER_CONTEXT_SIZE=${BUFFER_CONTEXT_SIZE:-10}

SPECTRAL_K=${SPECTRAL_K:-8}
SPECTRAL_HISTORY_SIZE=${SPECTRAL_HISTORY_SIZE:-8}
GLOBAL_GATE_INIT_BIAS=${GLOBAL_GATE_INIT_BIAS:--6.0}
FORCE_OPEN_GLOBAL_GATE_INIT_BIAS=${FORCE_OPEN_GLOBAL_GATE_INIT_BIAS:-6.0}
DESAT_GLOBAL_GATE_INIT_BIAS=${DESAT_GLOBAL_GATE_INIT_BIAS:--2.0}
HIGH_GATE_LR_MULTIPLIER=${HIGH_GATE_LR_MULTIPLIER:-10.0}
CONSTANT_GLOBAL_GATE=${CONSTANT_GLOBAL_GATE:-0.5}
COMPLEX_MOD_FREQ_CUTOFF_RATIO=${COMPLEX_MOD_FREQ_CUTOFF_RATIO:-1.0}
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
    "$@" > "${OUTPUT}" 2>&1
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
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local variant="${5}"
  shift 5

  local exit_code
  set +e
  "$@"
  exit_code=$?
  set -e
  if [ "${exit_code}" -eq 0 ]; then
    return 0
  fi
  record_failure scheme6 "${model}" "${dataset}" "${pred_len}" "${seed}" "${variant}" "${exit_code}"
  if is_true "${CONTINUE_ON_ERROR}"; then
    return 0
  fi
  return "${exit_code}"
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
    return 1
  fi
}

configure_ablation() {
  SPECTRAL_USAGE=direct
  SPECTRAL_MODE=concat
  CORRECTION_SOURCE=residual
  GATE_MODE=unidirectional
  USE_GLOBAL_GATE=True
  OUT_PROJ=diagonal
  WINDOW_TYPE=hann
  ENABLE_REFERENCE_CONTENT_INJECTION=False
  RUN_GLOBAL_GATE_INIT_BIAS="${GLOBAL_GATE_INIT_BIAS}"
  RUN_GLOBAL_GATE_LR_MULTIPLIER=1.0
  RUN_CONSTANT_GLOBAL_GATE=-1.0
  RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO="${COMPLEX_MOD_FREQ_CUTOFF_RATIO}"

  case "${1}" in
    core)
      VARIANT_TAG="adapter_residual_only"
      ;;
    no_out_proj)
      OUT_PROJ=none
      VARIANT_TAG="adapter_residual_no_out_proj"
      ;;
    no_global_gate)
      USE_GLOBAL_GATE=False
      VARIANT_TAG="adapter_residual_no_global_gate"
      ;;
    prediction_only)
      CORRECTION_SOURCE=prediction
      VARIANT_TAG="adapter_prediction_only"
      ;;
    two_path)
      CORRECTION_SOURCE=both
      VARIANT_TAG="adapter_two_path"
      ;;
    context_plus_core)
      SPECTRAL_USAGE=context_plus_direct
      VARIANT_TAG="adapter_context_plus_residual"
      ;;
    context_plus_complex_modulation_high_gate_lr|context_plus_complex_mod_high_gate_lr)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      VARIANT_TAG="adapter_context_plus_complex_modulation_high_gate_lr"
      ;;
    context_plus_complex_modulation_high_gate_lr_lowpass_005|context_plus_complex_mod_high_gate_lr_lowpass_005)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.05
      VARIANT_TAG="adapter_context_plus_complex_modulation_high_gate_lr_lowpass_005"
      ;;
    context_plus_complex_modulation_high_gate_lr_lowpass_010|context_plus_complex_mod_high_gate_lr_lowpass_010)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.10
      VARIANT_TAG="adapter_context_plus_complex_modulation_high_gate_lr_lowpass_010"
      ;;
    context_plus_complex_modulation_high_gate_lr_lowpass_020|context_plus_complex_mod_high_gate_lr_lowpass_020)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.20
      VARIANT_TAG="adapter_context_plus_complex_modulation_high_gate_lr_lowpass_020"
      ;;
    context_plus_complex_modulation_constant_gate_005|context_plus_complex_mod_const_gate_005)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.05
      VARIANT_TAG="adapter_context_plus_complex_modulation_constant_gate_005"
      ;;
    context_plus_complex_modulation_constant_gate_010|context_plus_complex_mod_const_gate_010)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.10
      VARIANT_TAG="adapter_context_plus_complex_modulation_constant_gate_010"
      ;;
    context_plus_complex_modulation_constant_gate_020|context_plus_complex_mod_const_gate_020)
      SPECTRAL_USAGE=context_plus_direct
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.20
      VARIANT_TAG="adapter_context_plus_complex_modulation_constant_gate_020"
      ;;
    complex_modulation|complex_mod)
      CORRECTION_SOURCE=complex_modulation
      VARIANT_TAG="adapter_complex_modulation"
      ;;
    complex_modulation_force_open|complex_mod_force_open)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${FORCE_OPEN_GLOBAL_GATE_INIT_BIAS}"
      VARIANT_TAG="adapter_complex_modulation_force_global_gate_open"
      ;;
    complex_modulation_desat_gate|complex_mod_desat_gate|complex_mod_desat)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      VARIANT_TAG="adapter_complex_modulation_desat_gate"
      ;;
    complex_modulation_high_gate_lr|complex_mod_high_gate_lr)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      VARIANT_TAG="adapter_complex_modulation_high_gate_lr"
      ;;
    complex_modulation_high_gate_lr_lowpass_005|complex_mod_high_gate_lr_lowpass_005)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.05
      VARIANT_TAG="adapter_complex_modulation_high_gate_lr_lowpass_005"
      ;;
    complex_modulation_high_gate_lr_lowpass_010|complex_mod_high_gate_lr_lowpass_010)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.10
      VARIANT_TAG="adapter_complex_modulation_high_gate_lr_lowpass_010"
      ;;
    complex_modulation_high_gate_lr_lowpass_020|complex_mod_high_gate_lr_lowpass_020)
      CORRECTION_SOURCE=complex_modulation
      RUN_GLOBAL_GATE_INIT_BIAS="${DESAT_GLOBAL_GATE_INIT_BIAS}"
      RUN_GLOBAL_GATE_LR_MULTIPLIER="${HIGH_GATE_LR_MULTIPLIER}"
      RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO=0.20
      VARIANT_TAG="adapter_complex_modulation_high_gate_lr_lowpass_020"
      ;;
    complex_modulation_constant_gate|complex_mod_constant_gate|complex_mod_const_gate)
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE="${CONSTANT_GLOBAL_GATE}"
      VARIANT_TAG="adapter_complex_modulation_constant_gate"
      ;;
    complex_modulation_constant_gate_005|complex_mod_constant_gate_005|complex_mod_const_gate_005)
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.05
      VARIANT_TAG="adapter_complex_modulation_constant_gate_005"
      ;;
    complex_modulation_constant_gate_010|complex_mod_constant_gate_010|complex_mod_const_gate_010)
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.10
      VARIANT_TAG="adapter_complex_modulation_constant_gate_010"
      ;;
    complex_modulation_constant_gate_020|complex_mod_constant_gate_020|complex_mod_const_gate_020)
      CORRECTION_SOURCE=complex_modulation
      RUN_CONSTANT_GLOBAL_GATE=0.20
      VARIANT_TAG="adapter_complex_modulation_constant_gate_020"
      ;;
    force_global_gate_open|force_open)
      RUN_GLOBAL_GATE_INIT_BIAS="${FORCE_OPEN_GLOBAL_GATE_INIT_BIAS}"
      VARIANT_TAG="adapter_residual_force_global_gate_open"
      ;;
    *)
      echo "Unknown ablation: ${1}" >&2
      exit 2
      ;;
  esac
}

run_scheme6_one() {
  local model="${1}"
  local dataset="${2}"
  local pred_len="${3}"
  local seed="${4}"
  local checkpoint_dir
  checkpoint_dir="$(checkpoint_dir_for "${model}" "${dataset}" "${pred_len}" "${seed}")"

  require_checkpoint "${checkpoint_dir}" || return

  RESULT_DIR="./results/COSA_SCHEME6/${VARIANT_TAG}/"
  OUTPUT="./results/summary/COSA_SCHEME6/${VARIANT_TAG}/${model}/${dataset}/pred${pred_len}_seed${seed}.log"
  LOG_CSV="./results/summary/COSA_SCHEME6/${VARIANT_TAG}/${model}/${dataset}/pred${pred_len}_seed${seed}_batches.csv"
  mkdir -p "$(dirname "${OUTPUT}")"

  if is_true "${SKIP_EXISTING}" && [ -f "${OUTPUT}" ]; then
    echo "skip scheme6: ${VARIANT_TAG}/${model}/${dataset}/pred${pred_len}/seed${seed}"
    return
  fi

  run_to_log "${PYTHON_CMD[@]}" main.py \
    SEED "${seed}" \
    DATA.BASE_DIR ./data/ \
    DATA.NAME "${dataset}" \
    DATA.SEQ_LEN "${SEQ_LEN}" \
    DATA.LABEL_LEN "${LABEL_LEN}" \
    DATA.PRED_LEN "${pred_len}" \
    DATA.TRAIN_RATIO "${TRAIN_RATIO}" \
    DATA.TEST_RATIO "${TEST_RATIO}" \
    DATA.SCALE standard \
    NORMALIZE NST \
    NORM_MODULE.ENABLE False \
    MODEL.NAME "${model}" \
    MODEL.seq_len "${SEQ_LEN}" \
    MODEL.label_len "${LABEL_LEN}" \
    MODEL.pred_len "${pred_len}" \
    MODEL.channel_independence True \
    TRAIN.ENABLE False \
    TRAIN.CHECKPOINT_DIR "${checkpoint_dir}" \
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
    TTA.COSA.SPECTRAL.CORRECTION_SOURCE "${CORRECTION_SOURCE}" \
    TTA.COSA.SPECTRAL.GATE_MODE "${GATE_MODE}" \
    TTA.COSA.SPECTRAL.ENABLE_REFERENCE_CONTENT_INJECTION "${ENABLE_REFERENCE_CONTENT_INJECTION}" \
    TTA.COSA.SPECTRAL.USE_GLOBAL_GATE "${USE_GLOBAL_GATE}" \
    TTA.COSA.SPECTRAL.GLOBAL_GATE_INIT_BIAS "${RUN_GLOBAL_GATE_INIT_BIAS}" \
    TTA.COSA.SPECTRAL.GLOBAL_GATE_LR_MULTIPLIER "${RUN_GLOBAL_GATE_LR_MULTIPLIER}" \
    TTA.COSA.SPECTRAL.CONSTANT_GLOBAL_GATE "${RUN_CONSTANT_GLOBAL_GATE}" \
    TTA.COSA.SPECTRAL.COMPLEX_MOD_FREQ_CUTOFF_RATIO "${RUN_COMPLEX_MOD_FREQ_CUTOFF_RATIO}" \
    TTA.COSA.SPECTRAL.WINDOW_TYPE "${WINDOW_TYPE}" \
    TTA.COSA.SPECTRAL.OUT_PROJ "${OUT_PROJ}" \
    TTA.COSA.SPECTRAL.FFT_PRECISION "${FFT_PRECISION}" \
    TTA.COSA.SPECTRAL.FFT_LENGTH_MODE "${FFT_LENGTH_MODE}" \
    TTA.COSA.SPECTRAL.LOG_CSV_PATH "${LOG_CSV}" \
    RESULT_DIR "${RESULT_DIR}"
}

init_failure_log

for ABLATION in "${ABLATION_LIST[@]}"; do
  configure_ablation "${ABLATION}"
  for MODEL in "${MODEL_LIST[@]}"; do
    for DATASET in "${DATASET_LIST[@]}"; do
      for RUN_PRED_LEN in "${PRED_LEN_LIST[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
          run_with_recovery "${MODEL}" "${DATASET}" "${RUN_PRED_LEN}" "${SEED}" "${VARIANT_TAG}" \
            run_scheme6_one "${MODEL}" "${DATASET}" "${RUN_PRED_LEN}" "${SEED}"
        done
      done
    done
  done
done
