"""Local LLM pass that trims low-value lines so the Avid sequence isn't littered
with a marker for every utterance.

Transcription captures *everything*; an editor reviewing rushes only wants
markers on lines worth jumping to. This pass reads the grouped transcript and
returns the line indices to DROP. It is deliberately non-destructive at the
caller layer: dropped lines are un-ticked (excluded from export) but stay
visible and re-tickable. "Useful" is subjective, so a trim `level` and optional
free-text `guidance` steer how aggressive it is.

Runs on-device via `mlx-lm`; see `backend/llm.py` for the shared model plumbing.
"""

from __future__ import annotations

import re

from backend.llm import (
    DEFAULT_LLM_MODEL,
    extract_json_array,
    generate_chat,
    iter_windows,
    load_model,
)

# Trim aggressiveness. "dedup" is deterministic (no LLM); the rest add an LLM pass
# on top of the deterministic dedup. Unknown levels fall back to balanced.
LEVELS = ("dedup", "light", "balanced", "aggressive")
# How much the LLM may add on top of dedup at "light" before we treat it as
# over-reach and keep only the safe deterministic dedup (a fraction of all lines).
_LIGHT_LLM_CAP_FRACTION = 0.2
_LEVEL_POLICY = {
    "light": (
        "Remove ONLY lines that are clearly NOT part of the scene's spoken "
        "dialogue: stray or garbled transcription, hallucinated repetition, or an "
        "utterance unrelated to the conversation. KEEP every line of real "
        "dialogue, including short replies, one-word answers, and acknowledgements "
        "like \"okay\" or \"yeah\". Never remove a line merely for being short or "
        "being filler. When in any doubt, keep it."
    ),
    "balanced": (
        "Drop filler and backchannel, false starts, near-repeats, and small talk "
        "or bare acknowledgements that carry no information. Keep any line that "
        "advances the conversation or carries meaning."
    ),
    "aggressive": (
        "Keep ONLY lines that carry real information or story value: questions, "
        "substantive answers, decisions, revelations, named people/places, and "
        "clear emotional beats. Drop everything else."
    ),
}

_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for exact-match dedup."""
    return " ".join(_PUNCT_RE.sub("", text).lower().split())


def find_duplicate_indices(texts: list[str], lookback: int = 3, min_words: int = 2) -> list[int]:
    """Indices of lines that repeat a recent line — mlx-whisper's doubles.

    Two cases, both keeping the first occurrence and dropping the repeats:
    - **Adjacent** exact repeats (a line identical to the immediately preceding
      kept line) are always dropped, whatever their length — these are the classic
      Whisper hallucination loops (e.g. five "Okay." in a row over silence).
    - **Non-adjacent** repeats within `lookback` kept lines are dropped only when
      the line has at least `min_words` words, so distinct short replies that
      simply recur (two people each saying "yeah") are left alone.
    """
    drops: list[int] = []
    recent: list[str] = []  # normalized text of the last `lookback` kept lines
    prev_kept: str | None = None  # normalized text of the immediately preceding kept line
    for i, text in enumerate(texts):
        norm = _normalize(text)
        adjacent_repeat = bool(norm) and norm == prev_kept
        nearby_repeat = len(norm.split()) >= min_words and norm in recent
        if adjacent_repeat or nearby_repeat:
            drops.append(i)
            continue
        prev_kept = norm
        recent.append(norm)
        if len(recent) > lookback:
            recent.pop(0)
    return drops


def build_triage_messages(
    texts: list[str],
    speakers: list[str],
    win_start: int,
    win_end: int,
    core_start: int,
    core_end: int,
    level: str,
    guidance: str,
) -> list[dict[str, str]]:
    """Build chat messages for one window. Lines `win_start:win_end` are shown;
    only `core_start:core_end` (end-exclusive) are decided. Numbering is global."""
    rendered = "\n".join(
        f"{i}: ({speakers[i] or '?'}) {texts[i]}" for i in range(win_start, win_end)
    )
    policy = _LEVEL_POLICY.get(level, _LEVEL_POLICY["balanced"])
    steer = f"\nEditor's guidance (overrides the above where relevant): {guidance.strip()}" \
        if guidance and guidance.strip() else ""
    system = (
        "You are trimming a dialogue transcript that will become timeline markers "
        "for a video editor reviewing TV rushes. The editor wants markers only on "
        "lines worth jumping to — not every utterance. Decide which lines to DROP.\n"
        f"Trim level: {policy}{steer}\n"
        "Output STRICT JSON only: an array of the integer line numbers to drop, "
        "e.g. [3, 7, 12]. Use [] to keep everything."
    )
    user = (
        f"Transcript:\n{rendered}\n\n"
        f"Only decide lines {core_start} to {core_end - 1} inclusive; lines outside "
        f"that range are context only. Return the JSON array of line numbers to drop now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_drops(raw: str, min_line: int, max_line: int) -> list[int]:
    """Parse model output into validated line numbers to drop within range."""
    out: list[int] = []
    seen: set[int] = set()
    for item in extract_json_array(raw):
        if isinstance(item, dict):  # tolerate [{"line": 3}, ...]
            item = item.get("line", item.get("index"))
        if isinstance(item, bool):  # bool is an int subclass — reject explicitly
            continue
        if not isinstance(item, int):
            try:
                item = int(item)
            except (TypeError, ValueError):
                continue
        if item < min_line or item > max_line or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def window_drops_plausible(window_drops: list[int], core_count: int) -> bool:
    """False if a window dropped *every* one of its lines — almost always a
    malformed/echoed response rather than a genuine call to blank the section."""
    return not (core_count > 0 and len(window_drops) >= core_count)


def triage_segments(
    texts: list[str],
    speakers: list[str],
    *,
    level: str = "balanced",
    guidance: str = "",
    model_repo: str = DEFAULT_LLM_MODEL,
    window: int = 80,
    context: int = 6,
    max_tokens: int = 600,
) -> list[int]:
    """Return the sorted indices of lines to drop. `texts`/`speakers` are parallel
    lists over the grouped segments. Empty input or `level` unknown is handled
    gracefully (unknown level falls back to balanced)."""
    n = len(texts)
    if n == 0:
        return []

    # Deterministic dedup runs at every level and is always safe to drop.
    dup_drops = set(find_duplicate_indices(texts))
    if level == "dedup":
        return sorted(dup_drops)

    model, tokenizer = load_model(model_repo)
    llm_drops: set[int] = set()
    for win_start, core_start, core_end in iter_windows(n, window, context):
        messages = build_triage_messages(
            texts, speakers, win_start, core_end, core_start, core_end, level, guidance
        )
        raw = generate_chat(model, tokenizer, messages, max_tokens)
        window_drops = parse_drops(raw, core_start, core_end - 1)
        if not window_drops_plausible(window_drops, core_end - core_start):
            continue  # discard a "drop everything" window; let the editor trim by hand
        llm_drops.update(window_drops)

    # "Light" is meant to be a feather touch: if the LLM wants to cut more than a
    # small share on top of dedup, treat it as over-reach and keep dedup only.
    if level == "light":
        extra = llm_drops - dup_drops
        if len(extra) > max(2, int(_LIGHT_LLM_CAP_FRACTION * n)):
            llm_drops = set()

    return sorted(dup_drops | llm_drops)
