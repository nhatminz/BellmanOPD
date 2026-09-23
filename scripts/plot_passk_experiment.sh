#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PASSK_OUTPUTS_ROOT="${PASSK_OUTPUTS_ROOT:-${REPO_DIR}/outputs}"
PASSK_RESULTS_ROOT="${PASSK_RESULTS_ROOT:-${REPO_DIR}/results/passk}"

if (( $# == 0 )); then
  cat >&2 <<'EOF'
Usage:
  bash scripts/plot_passk_experiment.sh RUN_NAME [MORE_RUN_NAME ...]
  bash scripts/plot_passk_experiment.sh SUMMARY.json [MORE_SUMMARY.json ...]

RUN_NAME is a directory name under outputs/. The command only reads saved
summaries; it never starts vLLM inference.
EOF
  exit 2
fi

SUMMARY_ARGS=()
RUN_LABELS=()
for input in "$@"; do
  if [[ -f "${input}" ]]; then
    SUMMARY_PATH="$(cd "$(dirname "${input}")" && pwd)/$(basename "${input}")"
    RUN_LABEL="$(basename "${input}" .json)"
  else
    if [[ -d "${input}" ]]; then
      RUN_DIR="$(cd "${input}" && pwd)"
    else
      RUN_DIR="${PASSK_OUTPUTS_ROOT%/}/${input}"
    fi
    if [[ ! -d "${RUN_DIR}" ]]; then
      echo "Pass@K run directory does not exist: ${RUN_DIR}" >&2
      exit 1
    fi
    mapfile -t CANDIDATES < <(find "${RUN_DIR}/summaries" -maxdepth 1 -type f -name 'passk_summary_*.json' -print 2>/dev/null | sort)
    if (( ${#CANDIDATES[@]} != 1 )); then
      echo "Expected exactly one Pass@K summary under ${RUN_DIR}/summaries, found ${#CANDIDATES[@]}" >&2
      exit 1
    fi
    SUMMARY_PATH="${CANDIDATES[0]}"
    RUN_LABEL="$(basename "${RUN_DIR}")"
  fi
  if [[ ! -f "${SUMMARY_PATH}" ]]; then
    echo "Pass@K summary does not exist: ${SUMMARY_PATH}" >&2
    exit 1
  fi
  SUMMARY_ARGS+=(--summary "${SUMMARY_PATH}")
  RUN_LABELS+=("${RUN_LABEL}")
done

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
COMPARISON_RAW="${RUN_LABELS[0]}"
for (( index = 1; index < ${#RUN_LABELS[@]}; index++ )); do
  COMPARISON_RAW+="_vs_${RUN_LABELS[index]}"
done
COMPARISON_NAME="$(
  "${PYTHON_BIN}" -c \
    'from b200_experiment.passk_plotting import _slug; import sys; print(_slug(sys.argv[1]))' \
    "${COMPARISON_RAW}"
)"
PASSK_PLOT_TAG="${PASSK_PLOT_TAG:-${COMPARISON_NAME}}"
PASSK_FIGURE_DIR="${PASSK_FIGURE_DIR:-${PASSK_RESULTS_ROOT%/}/${COMPARISON_NAME}}"

echo "Comparing Pass@K runs: ${RUN_LABELS[*]}"
echo "Figure output: ${PASSK_FIGURE_DIR}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" scripts/plot_passk.py \
  "${SUMMARY_ARGS[@]}" \
  --output-dir "${PASSK_FIGURE_DIR}" \
  --tag "${PASSK_PLOT_TAG}"
