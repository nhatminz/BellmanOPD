#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD="${1:?Usage: train_alfworld_multiturn_14b.sh cmt|opd|ta|iw [CLI overrides]}"
shift
case "${METHOD}" in cmt|opd|ta|iw) ;; *) echo "Unknown method: ${METHOD}" >&2; exit 2 ;; esac

# ================= USER CONFIG: ALFWORLD 14B -> 4B =================
export B200_METHOD="${METHOD}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export STUDENT_MODEL="${STUDENT_MODEL:-${STUDENT_MODEL_PATH:-models/Qwen3-4B}}"
export TEACHER_MODEL="${TEACHER_MODEL:-${TEACHER_MODEL_PATH:-models/Qwen3-14B}}"
export TRAIN_DATASET=custom
export TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-nlp/minhpn19/data/ALFworld/train-00000-of-00001.parquet}"
export TRAIN_PROMPT_KEY=id
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RUN_NAME="${RUN_NAME:-alfworld_v2_${METHOD}_14b_$(date +%Y%m%d_%H%M%S_%N)}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-${SEED}}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"       # global tasks per rollout
export NUM_RESPONSES="${NUM_RESPONSES:-4}"               # episodes per task
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"    # global episodes per update
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
export SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-1}"
export MAX_TURNS="${MAX_TURNS:-30}"
export MAX_TURN_TOKENS="${MAX_TURN_TOKENS:-128}"
export MAX_EPISODE_TOKENS="${MAX_EPISODE_TOKENS:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-3840}"      # cumulative actor tokens
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-${MAX_EPISODE_TOKENS}}"
export ROLLOUT_VLLM_STARTUP_TIMEOUT="${ROLLOUT_VLLM_STARTUP_TIMEOUT:-1800}"
export ROLLOUT_VLLM_LOGGING_LEVEL="${ROLLOUT_VLLM_LOGGING_LEVEL:-INFO}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export MAX_STEPS="${MAX_STEPS:-1000}"                    # total target, also on resume
export LR="${LR:-5.0e-6}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
export TRAIN_EVAL_ENABLED=false
export ACTION_PROMPT_MODE="${ACTION_PROMPT_MODE:-admissible}"
export COLLAPSE_GUARD_MODE="${COLLAPSE_GUARD_MODE:-error}"
# Diagnostics are off by default; when enabled compare processed sampling LPs.
export VLLM_LOGPROB_SANITY_FAIL="${VLLM_LOGPROB_SANITY_FAIL:-true}"
export TOKEN_SCORE_INTERVAL="${TOKEN_SCORE_INTERVAL:-50}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME:-}}"
# ========================================================
source "${SCRIPT_DIR}/common_b200.sh"
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != auto && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname -- "${RESUME_FROM_CHECKPOINT}")"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/${RUN_NAME}/${METHOD}_opd}"
fi
build_training_args "${OUTPUT_DIR}"
print_asset_selection
echo "ALFWorld ${METHOD}: output=${OUTPUT_DIR}, max_turns=${MAX_TURNS}, actor cap=${MAX_RESPONSE_LEN}, context=${MAX_EPISODE_TOKENS}"
cd "${REPO_DIR}"
run_training_cli train --config "${REPO_DIR}/configs/alfworld_multiturn.yaml" \
  "${COMMON_TRAIN_ARGS[@]}" \
  --set "experiment.method=${METHOD}" \
  --set "alfworld.max_turns=${MAX_TURNS}" \
  --set "alfworld.max_turn_tokens=${MAX_TURN_TOKENS}" \
  --set "alfworld.max_episode_tokens=${MAX_EPISODE_TOKENS}" \
  --set "alfworld.action_prompt_mode=${ACTION_PROMPT_MODE}" \
  --set "alfworld.collapse_guard.mode=${COLLAPSE_GUARD_MODE}" \
  --set "alfworld.env_workers=${ENV_WORKERS:-4}" \
  --set "alfworld.env_timeout_sec=${ENV_TIMEOUT_SEC:-60}" "$@"
