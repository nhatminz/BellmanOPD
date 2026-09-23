#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ========================= PASS@K CONFIG =========================
# One TP=1 vLLM replica is launched per visible GPU when PASSK_WORLD_SIZE=0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
STORAGE_ROOT="${STORAGE_ROOT:-/workspace/storage-shared}"
# Every checkpoint gets one independently reusable run under outputs/. Set
# PASSK_RUN_NAME only when launching exactly one checkpoint to override the
# readable automatic name (e.g. cmt_checkpoint_000600_qwen3_4b_<source-run>).
PASSK_OUTPUTS_ROOT="${PASSK_OUTPUTS_ROOT:-${REPO_DIR}/outputs}"
PASSK_RUN_NAME="${PASSK_RUN_NAME:-}"

PASSK_BENCHMARKS="${PASSK_BENCHMARKS:-AIME24 AIME25 AMC23}"
AIME_K_VALUES="${AIME_K_VALUES:-8 16 32 64 128}"
AMC_K_VALUES="${AMC_K_VALUES:-4 8 16 32 64}"
AIME_NUM_SAMPLES="${AIME_NUM_SAMPLES:-128}"
AMC_NUM_SAMPLES="${AMC_NUM_SAMPLES:-64}"

PASSK_TEMPERATURE="${PASSK_TEMPERATURE:-1.0}"
PASSK_TOP_P="${PASSK_TOP_P:-0.8}"
PASSK_MAX_NEW_TOKENS="${PASSK_MAX_NEW_TOKENS:-7168}"
PASSK_MAX_MODEL_LEN="${PASSK_MAX_MODEL_LEN:-9216}"
PASSK_TENSOR_PARALLEL_SIZE="${PASSK_TENSOR_PARALLEL_SIZE:-1}"
PASSK_WORLD_SIZE="${PASSK_WORLD_SIZE:-0}"  # 0 = all visible GPUs for TP=1
PASSK_GPU_MEMORY_UTILIZATION="${PASSK_GPU_MEMORY_UTILIZATION:-auto}"
PASSK_GPU_HEADROOM_GIB="${PASSK_GPU_HEADROOM_GIB:-4}"
PASSK_GPU_WORKSPACE_HEADROOM_GIB="${PASSK_GPU_WORKSPACE_HEADROOM_GIB:-2}"
PASSK_MAX_NUM_SEQS="${PASSK_MAX_NUM_SEQS:-256}"
PASSK_SEED="${PASSK_SEED:-1234}"
PASSK_PLOT_AFTER_RUN="${PASSK_PLOT_AFTER_RUN:-false}"

# Format per line/item:
#   MODEL_GROUP|METHOD|LABEL|CHECKPOINT[|CONFIG]
#
# Edit/uncomment these examples on the B200 server. The final field is optional;
# method defaults select configs/qwen3_b200_{opd,cmt,ta,...}.yaml.
CHECKPOINT_SPECS=(
  # "qwen3_1.7b|opd|OPD|/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/<RUN_17B_OPD>/opd/final"
  # "qwen3_1.7b|cmt|CMT|/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/<RUN_17B_CMT>/cmt_opd/final"
  # "qwen3_4b|opd|OPD|/workspace/storage-shared/nlp/minhpn19/BellmanOPD/outputs/<RUN_4B_OPD>/opd/final"
  # "qwen3_4b|cmt|CMT|/workspace/storage-shared/nlp/minhpn19/BellmanOPD_analysis/outputs/<RUN_4B_CMT>/cmt_opd/final"
)
# ================================================================

# For automation, either pass specs as positional arguments or provide a
# newline-separated PASSK_CHECKPOINT_SPECS variable; both override the edited
# CHECKPOINT_SPECS array above.
if (( $# > 0 )); then
  CHECKPOINT_SPECS=("$@")
elif [[ -n "${PASSK_CHECKPOINT_SPECS:-}" ]]; then
  mapfile -t CHECKPOINT_SPECS < <(
    printf '%s\n' "${PASSK_CHECKPOINT_SPECS}" |
      sed -e '/^[[:space:]]*$/d' -e '/^[[:space:]]*#/d'
  )
fi

if (( ${#CHECKPOINT_SPECS[@]} == 0 )); then
  echo "No checkpoints configured." >&2
  echo "Edit CHECKPOINT_SPECS or set newline-separated PASSK_CHECKPOINT_SPECS." >&2
  exit 2
fi

BENCHMARK_INPUT="${PASSK_BENCHMARKS//,/ }"
read -r -a BENCHMARK_LIST <<< "${BENCHMARK_INPUT}"
if (( ${#BENCHMARK_LIST[@]} == 0 )); then
  echo "PASSK_BENCHMARKS must not be empty" >&2
  exit 2
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"

echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES} (world_size=${PASSK_WORLD_SIZE}, TP=${PASSK_TENSOR_PARALLEL_SIZE})"
echo "Benchmarks: ${BENCHMARK_LIST[*]}"
echo "AIME: N=${AIME_NUM_SAMPLES}, K={${AIME_K_VALUES}}"
echo "AMC23: N=${AMC_NUM_SAMPLES}, K={${AMC_K_VALUES}}"
echo "Sampling: temperature=${PASSK_TEMPERATURE}, top_p=${PASSK_TOP_P}, seed=${PASSK_SEED}"
if [[ -n "${PASSK_RUN_NAME}" ]] && (( ${#CHECKPOINT_SPECS[@]} != 1 )); then
  echo "PASSK_RUN_NAME can only be used with exactly one checkpoint spec" >&2
  exit 2
fi

cd "${REPO_DIR}"
CREATED_RUN_NAMES=()
for spec in "${CHECKPOINT_SPECS[@]}"; do
  IFS='|' read -r MODEL_GROUP METHOD LABEL CHECKPOINT CONFIG EXTRA <<< "${spec}"
  if [[ -n "${EXTRA:-}" || -z "${MODEL_GROUP:-}" || -z "${METHOD:-}" || -z "${LABEL:-}" || -z "${CHECKPOINT:-}" ]]; then
    echo "Invalid checkpoint spec: ${spec}" >&2
    echo "Expected MODEL_GROUP|METHOD|LABEL|CHECKPOINT[|CONFIG]" >&2
    exit 2
  fi

  AUTO_RUN_NAME="$(
    "${PYTHON_BIN}" -c \
      'from b200_experiment.passk_experiment import automatic_passk_run_name; import sys; print(automatic_passk_run_name(*sys.argv[1:]))' \
      "${METHOD}" "${MODEL_GROUP}" "${CHECKPOINT}"
  )"
  RUN_NAME="${PASSK_RUN_NAME:-${AUTO_RUN_NAME}}"
  if ! [[ "${RUN_NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid PASSK run name: ${RUN_NAME}" >&2
    exit 2
  fi
  RUN_OUTPUT="${PASSK_OUTPUTS_ROOT%/}/${RUN_NAME}"
  RUN_SPEC="${MODEL_GROUP}|${METHOD}|${LABEL}|${CHECKPOINT}"
  if [[ -n "${CONFIG:-}" ]]; then
    RUN_SPEC+="|${CONFIG}"
  fi

  echo
  echo "Pass@K run name: ${RUN_NAME}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Output: ${RUN_OUTPUT}"

  "${PYTHON_BIN}" -m b200_experiment.passk_experiment \
    --checkpoint-spec "${RUN_SPEC}" \
    --storage-root "${STORAGE_ROOT}" \
    --output-root "${RUN_OUTPUT}" \
    --tag "${RUN_NAME}" \
    --benchmarks "${BENCHMARK_LIST[@]}" \
    --aime-k-values "${AIME_K_VALUES}" \
    --amc-k-values "${AMC_K_VALUES}" \
    --aime-num-samples "${AIME_NUM_SAMPLES}" \
    --amc-num-samples "${AMC_NUM_SAMPLES}" \
    --temperature "${PASSK_TEMPERATURE}" \
    --top-p "${PASSK_TOP_P}" \
    --max-new-tokens "${PASSK_MAX_NEW_TOKENS}" \
    --max-model-len "${PASSK_MAX_MODEL_LEN}" \
    --tensor-parallel-size "${PASSK_TENSOR_PARALLEL_SIZE}" \
    --world-size "${PASSK_WORLD_SIZE}" \
    --gpu-memory-utilization "${PASSK_GPU_MEMORY_UTILIZATION}" \
    --gpu-headroom-gib "${PASSK_GPU_HEADROOM_GIB}" \
    --gpu-workspace-headroom-gib "${PASSK_GPU_WORKSPACE_HEADROOM_GIB}" \
    --max-num-seqs "${PASSK_MAX_NUM_SEQS}" \
    --seed "${PASSK_SEED}"
  CREATED_RUN_NAMES+=("${RUN_NAME}")
done

echo
echo "Created/reused Pass@K run(s): ${CREATED_RUN_NAMES[*]}"
echo "Compare them with: bash scripts/plot_passk_experiment.sh ${CREATED_RUN_NAMES[*]}"
if [[ "${PASSK_PLOT_AFTER_RUN,,}" =~ ^(1|true|yes)$ ]]; then
  bash "${SCRIPT_DIR}/plot_passk_experiment.sh" "${CREATED_RUN_NAMES[@]}"
fi
