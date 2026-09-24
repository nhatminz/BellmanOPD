#!/usr/bin/env bash
set -euo pipefail

# D-only CMT ablation: L_t = D_t.  D_t keeps the canonical sequential
# construction, rollout-global tanh_q99 correction, and direct bounded Gibbs
# allocation.  Every other setting is inherited from train.sh or the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CMT_CORRECTION_MODE:-}" && "${CMT_CORRECTION_MODE}" != "tanh_q99" ]]; then
  echo "D-only ablation requires CMT_CORRECTION_MODE=tanh_q99" >&2
  exit 2
fi
if [[ -n "${CMT_ALLOCATION_MODE:-}" && "${CMT_ALLOCATION_MODE}" != "direct_bounded_gibbs" ]]; then
  echo "D-only ablation requires CMT_ALLOCATION_MODE=direct_bounded_gibbs" >&2
  exit 2
fi

export CMT_CORRECTION_MODE=tanh_q99
export CMT_ALLOCATION_MODE=direct_bounded_gibbs
exec bash "${SCRIPT_DIR}/train.sh" d_only "$@"
