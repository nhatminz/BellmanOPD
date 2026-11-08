#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
RUN_OUTPUT_ROOT="${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs"
EVAL_TAG="${EVAL_TAG:-extra43_44}"
FAILED_RUNS=()
COMPLETED_RUNS=0

run_eval() {
  local run_output="$1"
  local checkpoint="$2"
  local status
  printf '\nEvaluating: %s (checkpoint %s)\n' "${run_output}" "${checkpoint}"

  # An explicit if handles nonzero exit codes even with set -e enabled.
  # Keep the evaluator's original baseline/history validation intact.
  if bash "${REPO_DIR}/scripts/eval_multiseed_b200.sh" \
    --run-output "${run_output}" \
    --checkpoints "${checkpoint}" \
    --seeds 43 44 \
    --tag "${EVAL_TAG}"; then
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    printf 'Completed: %s (checkpoint %s)\n' "${run_output}" "${checkpoint}"
  else
    status=$?
    FAILED_RUNS+=("${run_output} | checkpoint=${checkpoint} | exit=${status}")
    printf 'FAILED (exit %s), continuing: %s (checkpoint %s)\n' \
      "${status}" "${run_output}" "${checkpoint}" >&2
  fi
  return 0
}

run_eval "${RUN_OUTPUT_ROOT}/ta_20260923_045233_485044857/ta_opd" 1500
run_eval "${RUN_OUTPUT_ROOT}/cmt_20260923_045745_710825266/cmt_opd" 1500
run_eval "${RUN_OUTPUT_ROOT}/iw_20260924_135151_691784646/iw" 750
run_eval "${RUN_OUTPUT_ROOT}/opd_20260921_074239_255652030/opd" 1350
run_eval "${RUN_OUTPUT_ROOT}/ta_20260923_054101_842665158/ta_opd" 1050
run_eval "${RUN_OUTPUT_ROOT}/cmt_20260923_045952_329544246/cmt_opd" 1050
run_eval "${RUN_OUTPUT_ROOT}/opd_20260923_054359_147056835/opd" 1050
run_eval "${RUN_OUTPUT_ROOT}/ta_20260923_050135_103387716/ta_opd" 1050

printf '\nFinished all runs: %s succeeded, %s failed.\n' \
  "${COMPLETED_RUNS}" "${#FAILED_RUNS[@]}"
if (( ${#FAILED_RUNS[@]} > 0 )); then
  printf 'Failed runs:\n' >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  # Report failure only AFTER every requested run has been attempted.
  exit 1
fi
