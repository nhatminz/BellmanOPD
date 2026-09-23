#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# One one-epoch full-CMT run for one final bounded-Gibbs KL budget.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CMT_FINAL_ALLOCATION_KL="${CMT_FINAL_ALLOCATION_KL:-0.02}"
export NUM_EPOCHS=1

# Methodological invariants of this ablation.
if [[ -n "${CMT_CORRECTION_MODE:-}" && "${CMT_CORRECTION_MODE}" != "tanh_q99" ]]; then
  echo "Final-allocation-KL ablation requires CMT_CORRECTION_MODE=tanh_q99" >&2
  exit 2
fi
if [[ -n "${CMT_ALLOCATION_MODE:-}" && "${CMT_ALLOCATION_MODE}" != "direct_bounded_gibbs" ]]; then
  echo "Final-allocation-KL ablation requires CMT_ALLOCATION_MODE=direct_bounded_gibbs" >&2
  exit 2
fi
export CMT_CORRECTION_MODE=tanh_q99
export CMT_CORRECTION_QUANTILE=0.99
export CMT_ALLOCATION_MODE=direct_bounded_gibbs
export ABLATION_KIND=final_allocation_kl

if [[ -z "${RUN_NAME:-}" ]]; then
  _final_kl_slug="${CMT_FINAL_ALLOCATION_KL//-/m}"
  _final_kl_slug="${_final_kl_slug//./p}"
  export RUN_NAME="cmt_finalkl${_final_kl_slug}_seed${SEED:-42}_$(date +%Y%m%d_%H%M%S)"
fi

echo "Final allocation KL: ${CMT_FINAL_ALLOCATION_KL}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Robust correction: ${CMT_CORRECTION_MODE} (q=${CMT_CORRECTION_QUANTILE})"
echo "Allocation mode: ${CMT_ALLOCATION_MODE}"

exec bash "${SCRIPT_DIR}/train.sh" g_d "$@"
