"""Shared local-LLM plumbing for the refinement passes (speaker correction and
fluff triage). Runs on-device via `mlx-lm` (Apple Silicon GPU) — nothing leaves
the machine. The heavy import is lazy so the web app and unit tests run without
the ML stack.
"""

from __future__ import annotations

import importlib.util
import json
import os
import threading
from collections.abc import Iterator

# A 7B 4-bit instruct model: strong enough for the reasoning, ~4.5 GB, comfortable
# on Apple Silicon. Override with AVID_LLM_MODEL.
DEFAULT_LLM_MODEL = os.environ.get("AVID_LLM_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")


def mlx_lm_available() -> bool:
    """True if the local-LLM passes can run (Apple Silicon + mlx-lm installed).

    `AVID_DISABLE_LLM` is a master kill-switch for every LLM feature. We check via
    `find_spec` rather than importing `mlx_lm`: importing it would initialise MLX
    on the calling thread, which must not happen off the dedicated MLX thread (see
    `backend/mlx_runtime`).
    """
    if os.environ.get("AVID_DISABLE_LLM"):
        return False
    return importlib.util.find_spec("mlx_lm") is not None


# Loaded models are cached for the life of the process (a server handles many
# jobs/requests) and guarded by a lock since mlx generation is not thread-safe.
_MODEL_CACHE: dict[str, tuple] = {}
_MODEL_LOCK = threading.Lock()


def load_model(model_repo: str) -> tuple:
    with _MODEL_LOCK:
        if model_repo not in _MODEL_CACHE:
            from mlx_lm import load

            _MODEL_CACHE[model_repo] = load(model_repo)
        return _MODEL_CACHE[model_repo]


def generate_chat(model, tokenizer, messages: list[dict[str, str]], max_tokens: int) -> str:
    from mlx_lm import generate

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    try:
        return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    except TypeError:
        # Older mlx-lm took the prompt positionally.
        return generate(model, tokenizer, prompt, max_tokens=max_tokens, verbose=False)


def extract_json_array(raw: str) -> list:
    """Pull the first JSON array out of model output (tolerating fences/prose)."""
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def iter_windows(n: int, window: int, context: int) -> Iterator[tuple[int, int, int]]:
    """Yield (win_start, core_start, core_end) tiles over `n` items.

    `core` is the editable span; it is preceded by up to `context` read-only
    lines so a window sees the lead-in across boundaries. `core_end` is exclusive.
    """
    core_start = 0
    while core_start < n:
        core_end = min(core_start + window, n)
        win_start = max(0, core_start - context)
        yield win_start, core_start, core_end
        core_start = core_end
