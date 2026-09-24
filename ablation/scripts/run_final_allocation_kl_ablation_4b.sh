#!/usr/bin/env bash
set -euo pipefail

# Qwen3-14B teacher -> Qwen3-4B student specialization of the canonical final
# allocation-KL launcher.  Keep orchestration in the canonical scripts instead
# of maintaining divergent *_4b copies of every sweep implementation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export STUDENT_MODEL="${STUDENT_MODEL:-models/Qwen3-4B}"
export TEACHER_MODEL="${TEACHER_MODEL:-models/Qwen3-14B}"
export RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-analysis_qwen3_14b_to_4b}"

echo "Model pair: ${TEACHER_MODEL} -> ${STUDENT_MODEL}"
exec bash "${SCRIPT_DIR}/run_final_allocation_kl_ablation.sh" "$@"
