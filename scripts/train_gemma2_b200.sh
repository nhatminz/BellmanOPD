#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 1 )); then
  echo "Usage: bash scripts/train_gemma2_b200.sh opd|ta|iw|cmt|all [extra training CLI args]" >&2
  exit 2
fi
METHOD="$1"
shift
case "${METHOD}" in opd|ta|iw|cmt|all) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac
if [[ "${METHOD}" == all && -n "${OUTPUT_DIR:-}${RESUME_FROM_CHECKPOINT:-}${RESUME:-}" ]]; then
  echo "Use OUTPUT_ROOT with all; select one method when setting OUTPUT_DIR or resuming." >&2
  exit 2
fi

# USER CONFIG: same model/data/generation/batch workload across four methods.
# CMT keeps its response-index-grouped allocation; IW keeps its sampled-action
# objective. This wrapper changes neither method's mathematical definition.
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export TEACHER_MODEL="${TEACHER_MODEL:-models/gemma-2-9b-it}"
export STUDENT_MODEL="${STUDENT_MODEL:-models/gemma-2-2b-it}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RUN_NAME="${RUN_NAME:-gemma2_$(date +%Y%m%d_%H%M%S_%N)}"
export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-64}}"
export NUM_RESPONSES="${NUM_RESPONSES:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LR="${LR:-5.0e-6}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-8192}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-4096}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-150}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-150}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export SEED="${SEED:-42}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export TOP_K="${TOP_K:-16}"
export VLLM_RETURN_LOG_PROBS="${VLLM_RETURN_LOG_PROBS:-false}"
# Detailed diagnostics are optional; keep the same instrumentation for the
# four methods when measuring end-to-end overhead.
export TOKEN_SCORE_STATS_ENABLED="${TOKEN_SCORE_STATS_ENABLED:-false}"
export COST_LOGGING_ENABLED="${COST_LOGGING_ENABLED:-true}"
export COST_WARMUP_ROLLOUTS="${COST_WARMUP_ROLLOUTS:-1}"

REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/${RUN_NAME}}"
declare -A SUBDIRS=([opd]=opd [ta]=ta_opd [iw]=iw [cmt]=cmt_opd)
METHODS=("${METHOD}")
if [[ "${METHOD}" == all ]]; then METHODS=(opd ta iw cmt); fi
for CURRENT in "${METHODS[@]}"; do
  OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${SUBDIRS[${CURRENT}]}}" \
  COST_OPD_BASELINE="${COST_OPD_BASELINE:-$(if [[ "${METHOD}" == all && "${CURRENT}" != opd ]]; then printf '%s' "${OUTPUT_ROOT}/opd"; fi)}" \
    bash "${SCRIPT_DIR}/train_${CURRENT}_b200.sh" \
      --overlay "${REPO_DIR}/configs/gemma2_b200.yaml" "$@"
done
