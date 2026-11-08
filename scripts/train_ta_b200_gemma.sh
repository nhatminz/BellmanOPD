#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= USER CONFIG: TA-OPD / GEMMA-2 ==================
# Standalone launcher: edit parameters here, then run this file directly.
# This copies the supplied CMT experiment: same models, data, seed, LR,
# global batch=64, n=4, PPO batch=64, epochs=3 and response cap=4096.
# Only the backward micro-batch defaults to 4 instead of 16 after the
# observed loss.backward() OOM. Four GPUs accumulate four local micros/update.
# Change the SAME runtime controls in all four *_b200_gemma.sh launchers
# when reporting matched training cost. Scoring stays at 8 (not the OOM stage).
export TEACHER_MODEL="${TEACHER_MODEL:-nlp/minhpn19/models/gemma-2-9b-it}"
export STUDENT_MODEL="${STUDENT_MODEL:-nlp/minhpn19/models/gemma-2-2b-it}"
case "${TRAIN_DATASET:-competition_math}" in
  dapo_math|dapo-math|dapo)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/DAPO-Math-17k-Processed"
    _DEFAULT_PROMPT_KEY="prompt"
    _DEFAULT_TRAIN_SPLIT="all"
    _DEFAULT_NUM_EPOCHS="2"
    ;;
  *)
    _DEFAULT_TRAIN_DATA="nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
    _DEFAULT_PROMPT_KEY="problem"
    _DEFAULT_TRAIN_SPLIT="null"
    _DEFAULT_NUM_EPOCHS="3"
    ;;
esac
export TRAIN_DATA="${TRAIN_DATA:-${TRAIN_DATA_PATH:-${_DEFAULT_TRAIN_DATA}}}"
export PROMPT_KEY="${PROMPT_KEY:-${TRAIN_PROMPT_KEY:-${_DEFAULT_PROMPT_KEY}}}"
export TRAIN_DATA_SPLIT="${TRAIN_DATA_SPLIT:-${_DEFAULT_TRAIN_SPLIT}}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL}"
export TRAIN_DATA_PATH="${TRAIN_DATA}"
export TRAIN_PROMPT_KEY="${PROMPT_KEY}"

export RUN_NAME="${RUN_NAME:-ta_$(date +%Y%m%d_%H%M%S_%N)}"
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export GLOBAL_BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-${TRAIN_BATCH_SIZE:-64}}}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE}}"
export NUM_RESPONSES="${NUM_RESPONSES:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO:-${MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE:-4}}}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-auto}"
export NUM_EPOCHS="${NUM_EPOCHS:-${EPOCHS:-${_DEFAULT_NUM_EPOCHS}}}"
export MAX_STEPS="${MAX_STEPS:--1}"               # -1 = consume all configured epochs
export LR="${LR:-${LEARNING_RATE:-5.0e-6}}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LENGTH:-${MAX_PROMPT_LEN:-1024}}"
export OVERLONG_PROMPT_POLICY="${OVERLONG_PROMPT_POLICY:-filter}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LEN:-${MAX_NEW_TOKENS:-4096}}}"
export TOP_K="${TOP_K:-16}"
export SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-8}"
export TRIM_RESPONSE_PADDING="${TRIM_RESPONSE_PADDING:-true}"
export LENGTH_BUCKETED_SCORING="${LENGTH_BUCKETED_SCORING:-true}"
export LENGTH_BUCKETED_MICRO_BATCHES="${LENGTH_BUCKETED_MICRO_BATCHES:-true}"
export JOINT_CROSS_SCORING="${JOINT_CROSS_SCORING:-true}"
export SCORE_CHUNK_STEPS="${SCORE_CHUNK_STEPS:-128}"
export TA_VOCAB_CHUNK_TOKENS="${TA_VOCAB_CHUNK_TOKENS:-2048}"
export CMT_ALLOCATION_KL="${CMT_ALLOCATION_KL:-0.5}"
export CMT_ALLOCATION_MODE="${CMT_ALLOCATION_MODE:-direct_bounded_gibbs}"
export CMT_WEIGHT_MIN="${CMT_WEIGHT_MIN:-0.5}"
export CMT_WEIGHT_MAX="${CMT_WEIGHT_MAX:-2.0}"
export CMT_CORRECTION_MODE="${CMT_CORRECTION_MODE:-tanh_q99}"
export CMT_CORRECTION_QUANTILE="${CMT_CORRECTION_QUANTILE:-0.99}"
export CMT_FINAL_ALLOCATION_KL="${CMT_FINAL_ALLOCATION_KL:-0.02}"
export CMT_GAMMA="${CMT_GAMMA:-1.0}"
export CMT_SUCCESSOR_LAMBDA="${CMT_SUCCESSOR_LAMBDA:-1.0}"
export CMT_FULL_VOCAB_DIAGNOSTICS="${CMT_FULL_VOCAB_DIAGNOSTICS:-false}"
export CMT_TOKEN_AUDIT_ENABLED="${CMT_TOKEN_AUDIT_ENABLED:-false}"
export CMT_TOKEN_AUDIT_INTERVAL="${CMT_TOKEN_AUDIT_INTERVAL:-150}"
export CMT_TOKEN_AUDIT_TOP_K="${CMT_TOKEN_AUDIT_TOP_K:-50}"
export CMT_TOKEN_CONTEXT_RADIUS="${CMT_TOKEN_CONTEXT_RADIUS:-32}"
export CMT_GAIN_HEATMAP_ENABLED="${CMT_GAIN_HEATMAP_ENABLED:-false}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-150}"
export EVAL_INTERVAL="${EVAL_INTERVAL:-150}"
export LOG_INTERVAL="${LOG_INTERVAL:-1}"
# Compact all-valid-token histograms/quantiles.  This is independent of
# checkpoint/evaluation cadence and is enabled by default for CMT.
export TOKEN_SCORE_STATS_ENABLED="${TOKEN_SCORE_STATS_ENABLED:-true}"
export TOKEN_SCORE_INTERVAL="${TOKEN_SCORE_INTERVAL:-100}"
export ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"

# Keep enabled when accuracy-over-step plots are needed after training.
export TRAIN_EVAL_ENABLED="${TRAIN_EVAL_ENABLED:-false}"
export TRAIN_EVAL_BACKEND="${TRAIN_EVAL_BACKEND:-vllm}"
export TRAIN_EVAL_TEMPERATURE="${TRAIN_EVAL_TEMPERATURE:-0.7}"
export TRAIN_EVAL_TOP_P="${TRAIN_EVAL_TOP_P:-0.95}"
export TRAIN_EVAL_NUM_RESPONSES="${TRAIN_EVAL_NUM_RESPONSES:-8}"
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"
# Gemma2 supports 8192 context tokens. Periodic evaluation stays disabled,
# as in the supplied run. If enabled, prompt + response must fit this context.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# FSDP production defaults. GLOBAL_BATCH_SIZE is independent of GPU count.
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
export FSDP_TEACHER_CPU_OFFLOAD="${FSDP_TEACHER_CPU_OFFLOAD:-false}"
export FSDP_USE_NO_SYNC="${FSDP_USE_NO_SYNC:-false}"
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-100}"
export USE_BATCH_AUTOTUNE="${USE_BATCH_AUTOTUNE:-false}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
# 1024 prompt tokens + 4096 generated tokens fit within this context.
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-5200}"
export ROLLOUT_VLLM_WAKE_HEADROOM_GIB="${ROLLOUT_VLLM_WAKE_HEADROOM_GIB:-2}"

# Cost is implemented by common_b200.sh + trainer.py; it does not depend
# on using the older train_gemma2_b200.sh wrapper. All files live in OUTPUT_DIR.
export COST_LOGGING_ENABLED="${COST_LOGGING_ENABLED:-true}"
export COST_WARMUP_ROLLOUTS="${COST_WARMUP_ROLLOUTS:-1}"
export COST_MEMORY_SAMPLE_INTERVAL_SEC="${COST_MEMORY_SAMPLE_INTERVAL_SEC:-0.25}"
export COST_OPD_BASELINE="${COST_OPD_BASELINE:-}"
export TENSORBOARD_ENABLED="${TENSORBOARD_ENABLED:-true}"
export TENSORBOARD_LOG_INTERVAL="${TENSORBOARD_LOG_INTERVAL:-1}"
# Shared sampler controls and instrumentation across the four launchers.
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
export VLLM_RETURN_LOG_PROBS="${VLLM_RETURN_LOG_PROBS:-false}"

# Fresh train: leave empty. Resume: normally pass this on the launch command.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-${RESUME:-}}"
# ================================================================

source "${SCRIPT_DIR}/common_b200.sh"
export B200_METHOD="ta"

print_asset_selection
resolve_run_paths
if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_FROM_CHECKPOINT}" != "auto" && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="$(dirname -- "${RESUME_FROM_CHECKPOINT}")"
else
  OUTPUT_DIR="${OUTPUT_DIR:-${TA_RUN_OUTPUT}}"
fi
build_training_args "${OUTPUT_DIR}"
echo "TA-OPD / Gemma2 run: ${RUN_NAME}"
echo "TA-OPD / Gemma2 output: ${OUTPUT_DIR}"
echo "Gemma2 batch: prompts=${BATCH_SIZE}, responses=${NUM_RESPONSES}, PPO=${PPO_MINI_BATCH_SIZE}, micro/GPU=${MICRO_BATCH_SIZE_PER_GPU}, score micro/GPU=${SCORE_MICRO_BATCH_SIZE}"
echo "Cost logs: ${OUTPUT_DIR}/training_cost.{jsonl,csv} and training_cost_summary.json (enabled=${COST_LOGGING_ENABLED})"
echo "TensorBoard: ${OUTPUT_DIR}/tensorboard (enabled=${TENSORBOARD_ENABLED})"
echo "OPD cost baseline: ${COST_OPD_BASELINE:-not set; overhead is unavailable until a matched OPD baseline is supplied}"
echo "Token-score stats: enabled=${TOKEN_SCORE_STATS_ENABLED}, interval=${TOKEN_SCORE_INTERVAL} optimizer steps"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
else
  echo "Training mode: fresh"
fi
cd "${REPO_DIR}"
run_training_cli train \
  --config "${TA_CONFIG}" \
  "${COMMON_TRAIN_ARGS[@]}" "$@"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "TA-OPD / Gemma2 resume completed in: ${OUTPUT_DIR}"
else
  echo "TA-OPD / Gemma2 completed. Keep this identifier: TA_RUN_NAME=${RUN_NAME}"
fi
