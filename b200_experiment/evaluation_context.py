"""Model-specific context limits for evaluation of older resolved run configs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path


def resolve_evaluation_max_model_len(model_path: str | Path, requested: int | None):
    """Respect Gemma2's native context without changing generation/sampling caps.

    Some Gemma runs inherited Qwen's 9216-token evaluation reservation. Model
    weights and their config stay read-only; only the engine reservation is
    bounded here. Other families retain their existing RoPE/context handling.
    """
    path = Path(model_path) / "config.json"
    try:
        model_config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return requested
    if not isinstance(model_config, dict) or model_config.get("model_type") != "gemma2":
        return requested
    limit = int(model_config.get("max_position_embeddings", 8192))
    if limit <= 0:
        raise ValueError(f"Invalid Gemma2 max_position_embeddings={limit} in {path}")
    if requested is None:
        return limit
    if int(requested) <= 0:
        raise ValueError("Evaluation max_model_len must be positive")
    if int(requested) > limit:
        warnings.warn(
            f"Gemma2 checkpoint {model_path}: evaluation max_model_len={requested} "
            f"exceeds max_position_embeddings={limit}; using {limit}. "
            "max_new_tokens, sampling parameters and prompt text are unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
    return min(int(requested), limit)
