#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PLOT_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
INPUT_ROOT="${1:-${LAMBDA_SWEEP_DIR:-}}"
if [[ -z "${INPUT_ROOT}" ]]; then
  echo "Usage: $0 LAMBDA_SWEEP_DIR [SUMMARY_DIR]" >&2
  exit 2
fi
SUMMARY_DIR="${2:-${LAMBDA_SUMMARY_DIR:-${INPUT_ROOT}/summary}}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/../analysis/lambda_results.py" \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${SUMMARY_DIR}" \
  --expected-lambdas "${LAMBDA_VALUES:-0.25 0.5 1.0 2.0}" \
  --metric "${METRIC:-accuracy}"
