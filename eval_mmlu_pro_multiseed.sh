#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= USER CONFIG: MMLU-PRO =================
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${MMLU_PRO_GPUS:-0,1}"
MMLU_PRO_TEST="${MMLU_PRO_TEST:-${STORAGE_ROOT%/}/nlp/minhpn19/data/mmlu-pro/test-00000-of-00001.parquet}"
EVAL_TAG="${EVAL_TAG:-mmlu_pro_$(date +%Y%m%d_%H%M%S_%N)}"

# Add one checkpoint directory per line. Each checkpoint is evaluated in turn.
CHECKPOINTS=(
  "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/cmt_20260922_115452_504016128/cmt_opd/checkpoint-001500"
  # "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/ANOTHER_RUN/opd/checkpoint-001050"
  # "${STORAGE_ROOT%/}/nlp/minhpn19/BellmanOPD_analysis/outputs/ANOTHER_RUN/ta_opd/final"
)
SEEDS=(42 43 44)
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"  # questions per call; each has 8 completions
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"        # concurrent sequences on each GPU
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
# ========================================================

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"
printf 'MMLU-Pro: tag=%s, GPUs=%s, seeds=%s\n' "${EVAL_TAG}" "${CUDA_VISIBLE_DEVICES}" "${SEEDS[*]}"
exec "${PYTHON_BIN:-python3}" -m b200_experiment.mmlu_pro_evaluation \
  --storage-root "${STORAGE_ROOT}" \
  --data "${MMLU_PRO_TEST}" \
  --checkpoints "${CHECKPOINTS[@]}" \
  --seeds "${SEEDS[@]}" --gpus "${CUDA_VISIBLE_DEVICES}" --tag "${EVAL_TAG}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" --max-model-len "${MAX_MODEL_LEN}" \
  --batch-size "${EVAL_BATCH_SIZE}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-thinking "${ENABLE_THINKING}" "$@"
