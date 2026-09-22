#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PLOT_PYTHON_BIN:-${PYTHON_BIN:-python3}}"
INPUT_ROOT="${1:-${LAMBDA_SWEEP_DIR:-}}"
if [[ -z "${INPUT_ROOT}" ]]; then
  echo "Usage: $0 LAMBDA_SWEEP_DIR [FIGURE_DIR]" >&2
  exit 2
fi
SUMMARY_JSON="${LAMBDA_SUMMARY_JSON:-${INPUT_ROOT}/summary/lambda_summary.json}"
if [[ ! -f "${SUMMARY_JSON}" ]]; then
  echo "Missing ${SUMMARY_JSON}; run aggregate_lambda_ablation.sh first" >&2
  exit 1
fi
FIGURE_DIR="${2:-${LAMBDA_FIGURE_DIR:-${INPUT_ROOT}/figures}}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/../plots/plot_lambda_ablation.py" \
  --summary-json "${SUMMARY_JSON}" \
  --output-dir "${FIGURE_DIR}"
