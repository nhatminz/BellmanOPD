#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-random}"  # original | random | both (sequential train -> final eval)
if (( $# > 0 )); then shift; fi
case "${MODE}" in original|random|both) ;; *) echo "Usage: $0 [original|random|both] [CLI overrides]" >&2; exit 2 ;; esac

# ================= USER CONFIG: CONTROLLED CMT ABLATION =================
export STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
export B200_METHOD=cmt
export TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-4B}"
export STUDENT_MODEL="${STUDENT_MODEL:-models/Qwen3-1.7B}"
export TRAIN_DATASET=competition_math
export TRAIN_DATA="${STORAGE_ROOT%/}/nlp/minhpn19/data/competition_math/data/train-00000-of-00001.parquet"
export TRAIN_DATA_SPLIT=null
export PROMPT_KEY=problem
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SEED="${SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export BATCH_SIZE="${BATCH_SIZE:-${GLOBAL_BATCH_SIZE:-64}}"
export GLOBAL_BATCH_SIZE="${BATCH_SIZE}"
export NUM_RESPONSES="${NUM_RESPONSES:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-16}"
export SCORE_MICRO_BATCH_SIZE="${SCORE_MICRO_BATCH_SIZE:-8}"
export GRAD_ACCUM_STEPS=auto
export NUM_EPOCHS=1
export MAX_STEPS=-1
export LR="${LR:-5.0e-6}"
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-4096}"
export OVERLONG_PROMPT_POLICY="${OVERLONG_PROMPT_POLICY:-filter}"
export TOP_K="${TOP_K:-16}"
export ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
export ROLLOUT_VLLM_MAX_MODEL_LEN="${ROLLOUT_VLLM_MAX_MODEL_LEN:-5200}"
export CMT_ALLOCATION_MODE="${CMT_ALLOCATION_MODE:-direct_bounded_gibbs}"
export CMT_ALLOCATION_KL="${CMT_ALLOCATION_KL:-0.5}"
export CMT_FINAL_ALLOCATION_KL="${CMT_FINAL_ALLOCATION_KL:-0.02}"
export CMT_WEIGHT_MIN="${CMT_WEIGHT_MIN:-0.5}"
export CMT_WEIGHT_MAX="${CMT_WEIGHT_MAX:-2.0}"
export CMT_CORRECTION_MODE="${CMT_CORRECTION_MODE:-tanh_q99}"
export CMT_CORRECTION_QUANTILE="${CMT_CORRECTION_QUANTILE:-0.99}"
export CMT_GAMMA="${CMT_GAMMA:-1.0}"
export CMT_SUCCESSOR_LAMBDA="${CMT_SUCCESSOR_LAMBDA:-1.0}"
export CMT_PAIRED_WEIGHT_DIAGNOSTICS="${CMT_PAIRED_WEIGHT_DIAGNOSTICS:-false}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-150}"
export TRAIN_EVAL_ENABLED=false
export TRAIN_EVAL_TEMPERATURE="${TRAIN_EVAL_TEMPERATURE:-0.7}"
export TRAIN_EVAL_TOP_P="${TRAIN_EVAL_TOP_P:-0.95}"
export TRAIN_EVAL_NUM_RESPONSES=8
export TRAIN_EVAL_MAX_NEW_TOKENS="${TRAIN_EVAL_MAX_NEW_TOKENS:-7168}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9216}"
# Both arms always start from STUDENT_MODEL, never from the preceding arm.
export RESUME_FROM_CHECKPOINT=""
RUN_PREFIX="${RUN_PREFIX:-cmt_compe_math_4b_17b_1epoch_s${SEED}_$(date +%Y%m%d_%H%M%S_%N)}"
EVAL_TAG="${EVAL_TAG:-final_s42_43_44}"
# =======================================================================

source "${SCRIPT_DIR}/common_b200.sh"
cd "${REPO_DIR}"
MODES=("${MODE}")
if [[ "${MODE}" == both ]]; then MODES=(original random); fi
for ARM in "${MODES[@]}"; do
  OUTPUT_DIR="${REPO_DIR}/outputs/${RUN_PREFIX}_${ARM}/cmt_opd"
  if [[ "${ARM}" == random ]]; then
    export CMT_RANDOM_WEIGHT_CONTROL=true
  else
    export CMT_RANDOM_WEIGHT_CONTROL=false
  fi
  build_training_args "${OUTPUT_DIR}"
  print_asset_selection
  echo "CMT arm=${ARM}: one epoch, train-time eval disabled, output=${OUTPUT_DIR}"
  run_training_cli train --config "${CMT_CONFIG}" "${COMMON_TRAIN_ARGS[@]}" \
    --set experiment.method=cmt \
    --set training.epochs=1 --set training.max_steps=null \
    --set training.resume_from_checkpoint=null \
    --set training_evaluation.enabled=false \
    --set training_evaluation.eval_at_start=false \
    --set training_evaluation.eval_at_end=false "$@"
  echo "Training completed (${ARM}); evaluating final checkpoint on seeds 42,43,44."
  bash "${SCRIPT_DIR}/eval_multiseed_b200.sh" \
    --run-output "${OUTPUT_DIR}" --checkpoints final \
    --seeds 42 43 44 --fresh-seeds --tag "${EVAL_TAG}"
  echo "${ARM}: summary=${OUTPUT_DIR}/eval_summary_${EVAL_TAG}.json"
done
