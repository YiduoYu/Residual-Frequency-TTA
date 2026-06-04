#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON=${PYTHON:-python}
DRY_RUN=${DRY_RUN:-False}

MODELS=${MODELS:-"DLinear"}
DATASETS=${DATASETS:-"ETTh2 ETTm2 exchange_rate electricity"}
PRED_LENS=${PRED_LENS:-"96"}
SEEDS=${SEEDS:-"0 1 2"}
ABLATIONS=${ABLATIONS:-"core"}

CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-False}
SKIP_EXISTING=${SKIP_EXISTING:-False}
RUN_ID=${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}

SMOKE_VARIANT=${SMOKE_VARIANT:-adapter_residual_only}
REFERENCE_VARIANT=${REFERENCE_VARIANT:-baseline_no_tta}
MIN_BASELINE_WINS=${MIN_BASELINE_WINS:-1}
MIN_AVG_IMPROVEMENT_PCT=${MIN_AVG_IMPROVEMENT_PCT:-0}

SMOKE_PER_SEED=${SMOKE_PER_SEED:-Spectral-master/results/summary/scheme6_smoke_results.csv}
SMOKE_AVERAGED=${SMOKE_AVERAGED:-Spectral-master/results/summary/scheme6_smoke_averaged.csv}
SMOKE_IMPROVEMENT=${SMOKE_IMPROVEMENT:-Spectral-master/results/summary/scheme6_smoke_improvement_vs_baseline.csv}
REFERENCE_AVERAGED=${REFERENCE_AVERAGED:-result/5.26_paper8_experiment/final_tables/paper8_full_comparison_averaged.csv}

read -r -a PYTHON_CMD <<< "${PYTHON}"
read -r -a MODEL_LIST <<< "${MODELS}"
read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a PRED_LEN_LIST <<< "${PRED_LENS}"
read -r -a SEED_LIST <<< "${SEEDS}"

is_true() {
  case "${1}" in
    True|true|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

if [ "${ABLATIONS}" != "core" ]; then
  echo "Warning: smoke gate defaults to ${SMOKE_VARIANT}; set SMOKE_VARIANT if ABLATIONS is not core." >&2
fi

echo "Running Scheme6 smoke grid"
echo "  models: ${MODELS}"
echo "  datasets: ${DATASETS}"
echo "  horizons: ${PRED_LENS}"
echo "  seeds: ${SEEDS}"
echo "  ablations: ${ABLATIONS}"

MODELS="${MODELS}" \
DATASETS="${DATASETS}" \
PRED_LENS="${PRED_LENS}" \
SEEDS="${SEEDS}" \
ABLATIONS="${ABLATIONS}" \
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
SKIP_EXISTING="${SKIP_EXISTING}" \
RUN_ID="${RUN_ID}" \
DRY_RUN="${DRY_RUN}" \
PYTHON="${PYTHON}" \
bash "${SCRIPT_DIR}/run_residual_spectral_gating.sh"

if is_true "${DRY_RUN}"; then
  echo "DRY_RUN complete; skipped collect/check to avoid reading stale smoke logs."
  exit 0
fi

"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/collect_scheme6_results.py" \
  --root Spectral-master/results/summary \
  --output "${SMOKE_PER_SEED}" \
  --average-output "${SMOKE_AVERAGED}" \
  --improvement-output "${SMOKE_IMPROVEMENT}" \
  --reference-averaged "${REFERENCE_AVERAGED}" \
  --reference-variants "${REFERENCE_VARIANT}" \
  --families COSA_SCHEME6 \
  --variants "${SMOKE_VARIANT}" \
  --models "${MODEL_LIST[@]}" \
  --datasets "${DATASET_LIST[@]}" \
  --horizons "${PRED_LEN_LIST[@]}" \
  --expected-seeds "${SEED_LIST[@]}" \
  --require-complete

"${PYTHON_CMD[@]}" "${SCRIPT_DIR}/check_scheme6_smoke_vs_baseline.py" \
  --scheme6-averaged "${SMOKE_AVERAGED}" \
  --reference-averaged "${REFERENCE_AVERAGED}" \
  --variant "${SMOKE_VARIANT}" \
  --reference-variant "${REFERENCE_VARIANT}" \
  --models "${MODEL_LIST[@]}" \
  --datasets "${DATASET_LIST[@]}" \
  --horizons "${PRED_LEN_LIST[@]}" \
  --expected-seeds "${SEED_LIST[@]}" \
  --min-baseline-wins "${MIN_BASELINE_WINS}" \
  --min-avg-improvement-pct "${MIN_AVG_IMPROVEMENT_PCT}"
