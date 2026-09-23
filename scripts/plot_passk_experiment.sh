#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PASSK_OUTPUT_ROOT="${PASSK_OUTPUT_ROOT:-${REPO_DIR}/results/passk}"
PASSK_PLOT_TAG="${PASSK_PLOT_TAG:-$(date +%Y%m%d_%H%M%S)}"
PASSK_FIGURE_DIR="${PASSK_FIGURE_DIR:-${PASSK_OUTPUT_ROOT}/figures}"

if (( $# == 0 )); then
  cat >&2 <<'EOF'
Usage:
  bash scripts/plot_passk_experiment.sh SUMMARY.json [MORE_SUMMARY.json ...]

This command only reads saved summary files; it never starts vLLM inference.
EOF
  exit 2
fi

SUMMARY_ARGS=()
for summary in "$@"; do
  if [[ ! -f "${summary}" ]]; then
    echo "Pass@K summary does not exist: ${summary}" >&2
    exit 1
  fi
  SUMMARY_ARGS+=(--summary "${summary}")
done

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" scripts/plot_passk.py \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${PASSK_FIGURE_DIR}" \
  --tag "${PASSK_PLOT_TAG}"
