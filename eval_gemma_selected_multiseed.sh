#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES=0,1
RUN_OUTPUT_ROOT="${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs"
EVAL_TAG="${EVAL_TAG:-gemma_s42_43_44_steps200_1050_1500}"
FAILED_RUNS=()

for method_dir in \
  cmt_20261108_103126_485124171/cmt_opd \
  opd_gemma_micro4_seed42/opd \
  iw_gemma_micro4_seed42/iw \
  ta_gemma_micro4_seed42/ta_opd; do
  if bash "${REPO_DIR}/scripts/eval_multiseed_b200.sh" \
    --run-output "${RUN_OUTPUT_ROOT}/${method_dir}" \
    --checkpoints 200 1050 1500 \
    --seeds 42 43 44 --fresh-seeds \
    --world-size 2 --tag "${EVAL_TAG}" "$@"; then
    printf 'Completed: %s\n' "${method_dir}"
  else
    status=$?
    FAILED_RUNS+=("${method_dir} (exit=${status})")
    printf 'FAILED, continuing: %s (exit=%s)\n' "${method_dir}" "${status}" >&2
  fi
done
if (( ${#FAILED_RUNS[@]} > 0 )); then
  printf '\nFailed runs:\n' >&2
  printf '  %s\n' "${FAILED_RUNS[@]}" >&2
  exit 1
fi
