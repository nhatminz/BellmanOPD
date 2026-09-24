#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if (( $# != 2 )); then
  cat >&2 <<'EOF'
Usage:
  bash scripts/rebuild_pass8_history_b200.sh METHOD METHOD_RUN_OUTPUT

Examples:
  bash scripts/rebuild_pass8_history_b200.sh cmt outputs/<run>/cmt_opd
  bash scripts/rebuild_pass8_history_b200.sh opd /other/repo/outputs/<run>/opd
EOF
  exit 2
fi

METHOD="$1"
RUN_OUTPUT="$2"
if [[ "${RUN_OUTPUT}" != /* ]]; then
  RUN_OUTPUT="${REPO_DIR}/${RUN_OUTPUT}"
fi

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" -m b200_experiment.rebuild_evaluation_history \
  --method "${METHOD}" \
  --metric pass@8 \
  --run-output "${RUN_OUTPUT}"
