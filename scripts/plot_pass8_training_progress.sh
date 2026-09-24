#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLOT_EVAL_METRIC=pass@8
exec bash "${SCRIPT_DIR}/plot_training_progress.sh" "$@"
