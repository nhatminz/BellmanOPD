#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ================= USER CONFIG: ALFWORLD EVAL =================
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${ALFWORLD_EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
# One quoted path per line, WITHOUT commas. Methods are read from each run.
CHECKPOINTS=(
  "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/alfworld_v2_opd_20261109_093150_093055241/opd_opd/checkpoint-000400"
  # "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/ANOTHER_RUN/cmt_opd/checkpoint-000400"
  # "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/ANOTHER_RUN/ta_opd/final"
)
SEEDS=(42 43 44)
EPISODES_PER_TASK="${EPISODES_PER_TASK:-8}"
EVAL_TAG="${EVAL_TAG:-alfworld_avg8_pass8_$(date +%Y%m%d_%H%M%S_%N)}"
# ==============================================================

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"
# Keep historical --run-output/--checkpoints commands (one episode by default).
for arg in "$@"; do
  case "${arg}" in
    --run-output|--run-output=*|--worker)
      exec "${PYTHON_BIN:-python3}" -m b200_experiment.alfworld_evaluation "$@"
      ;;
  esac
done
printf 'ALFWorld eval: tag=%s, GPUs=%s, seeds=%s, episodes/task/seed=%s\n' \
  "${EVAL_TAG}" "${CUDA_VISIBLE_DEVICES}" "${SEEDS[*]}" "${EPISODES_PER_TASK}"
exec "${PYTHON_BIN:-python3}" -m b200_experiment.alfworld_evaluation \
  --checkpoint-paths "${CHECKPOINTS[@]}" \
  --seeds "${SEEDS[@]}" --gpus "${CUDA_VISIBLE_DEVICES}" \
  --episodes-per-task "${EPISODES_PER_TASK}" --tag "${EVAL_TAG}" "$@"
