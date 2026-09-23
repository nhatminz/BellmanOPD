#!/usr/bin/env bash
set -euo pipefail

# Sweep the final KL budget while holding every other CMT setting fixed.
# RUN_MODE=sequential uses CUDA_VISIBLE_DEVICES for every run.
# RUN_MODE=parallel maps one value to each GPU in GPU_LIST.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALUES="${FINAL_ALLOCATION_KL_VALUES:-0.0 0.005 0.02 0.05}"
RUN_MODE="${FINAL_KL_RUN_MODE:-sequential}"

export NUM_EPOCHS=1
export CMT_CORRECTION_MODE=tanh_q99
export CMT_CORRECTION_QUANTILE=0.99
export CMT_ALLOCATION_MODE=direct_bounded_gibbs
export ABLATION_KIND=final_allocation_kl

echo "Final-allocation-KL values: ${VALUES}"
echo "Epochs: ${NUM_EPOCHS}"
echo "CMT gain support: student_topk"
echo "Robust correction: ${CMT_CORRECTION_MODE} (q=${CMT_CORRECTION_QUANTILE})"
echo "Allocation mode: ${CMT_ALLOCATION_MODE}"
echo "Run mode: ${RUN_MODE}"

case "${RUN_MODE}" in
  sequential)
    export FINAL_ALLOCATION_KL_VALUES="${VALUES}"
    exec bash "${SCRIPT_DIR}/sweep.sh" final_allocation_kl "$@"
    ;;
  parallel)
    export FINAL_ALLOCATION_KL_VALUES="${VALUES}"
    exec bash "${SCRIPT_DIR}/sweep_parallel.sh" final_allocation_kl "$@"
    ;;
  *)
    echo "FINAL_KL_RUN_MODE must be sequential or parallel" >&2
    exit 2
    ;;
esac
