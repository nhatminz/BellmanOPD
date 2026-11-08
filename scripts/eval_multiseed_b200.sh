#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export PYTHONUNBUFFERED=1
# Deliberately do not source common_b200.sh: the old run's resolved config,
# rather than this checkout's default model/data paths, owns the evaluation.
cd "${REPO_DIR}"
exec "${PYTHON_BIN:-python3}" -m b200_experiment.multiseed_evaluation "$@"
