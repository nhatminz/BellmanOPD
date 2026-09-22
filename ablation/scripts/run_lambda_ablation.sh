#!/usr/bin/env bash
set -euo pipefail

# Sequential successor-lambda sweep. Every child is canonical/full CMT (g_d)
# and inherits the production CMT defaults from train.sh. Only lambda changes
# within a seed.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES="${LAMBDA_VALUES:-0.25 0.5 1.0 2.0}"
SEED_VALUES="${SEEDS:-${SEED:-42}}"
SWEEP_TAG="${LAMBDA_SWEEP_TAG:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${LAMBDA_SWEEP_DIR:-${SCRIPT_DIR}/../outputs/lambda_successor_${SWEEP_TAG}}"

# These two settings are methodological invariants of this sweep. Reject a
# conflicting inherited environment instead of silently comparing methods.
if [[ -n "${CMT_CORRECTION_MODE:-}" && "${CMT_CORRECTION_MODE}" != "tanh_q99" ]]; then
  echo "Lambda ablation requires CMT_CORRECTION_MODE=tanh_q99" >&2
  exit 2
fi
if [[ -n "${CMT_ALLOCATION_MODE:-}" && "${CMT_ALLOCATION_MODE}" != "direct_bounded_gibbs" ]]; then
  echo "Lambda ablation requires CMT_ALLOCATION_MODE=direct_bounded_gibbs" >&2
  exit 2
fi
export CMT_CORRECTION_MODE=tanh_q99
export CMT_CORRECTION_QUANTILE=0.99
export CMT_ALLOCATION_MODE=direct_bounded_gibbs
export ABLATION_KIND=lambda_successor

mkdir -p "${SWEEP_DIR}"
echo "Lambda sweep output root: ${SWEEP_DIR}"
echo "Lambda values: ${VALUES}"
echo "Seeds: ${SEED_VALUES}"
echo "CMT gain support: student_topk"
echo "Robust correction: ${CMT_CORRECTION_MODE} (q=${CMT_CORRECTION_QUANTILE})"
echo "Allocation mode: ${CMT_ALLOCATION_MODE}"

slug() {
  local value="$1"
  value="${value//-/m}"
  value="${value//./p}"
  printf '%s' "${value}"
}

for seed in ${SEED_VALUES}; do
  for value in ${VALUES}; do
    value_slug="$(slug "${value}")"
    run_name="cmt_lambda${value_slug}_seed${seed}_${SWEEP_TAG}"
    output_dir="${SWEEP_DIR}/${run_name}"
    echo
    echo "============================================================"
    echo "Run: ${run_name}"
    echo "successor_lambda=${value}"
    echo "correction=${CMT_CORRECTION_MODE}"
    echo "allocation=${CMT_ALLOCATION_MODE}"
    echo "output=${output_dir}"
    echo "============================================================"
    env \
      SEED="${seed}" \
      ROLLOUT_SEED="${seed}" \
      CMT_SUCCESSOR_LAMBDA="${value}" \
      RUN_NAME="${run_name}" \
      OUTPUT_DIR="${output_dir}" \
      bash "${SCRIPT_DIR}/train.sh" g_d "$@"
  done
done

echo
echo "Completed lambda sweep: ${SWEEP_DIR}"
echo "Aggregate: bash ${SCRIPT_DIR}/aggregate_lambda_ablation.sh '${SWEEP_DIR}'"
echo "Plot:     bash ${SCRIPT_DIR}/plot_lambda_ablation.sh '${SWEEP_DIR}'"
