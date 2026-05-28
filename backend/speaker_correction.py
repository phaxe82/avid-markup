"""Local LLM pass that fixes obvious speaker-label mistakes from diarization.

Diarization (pyannote) gets *who said what* right roughly 70% of the time on our
material. Many of its errors are obvious from the dialogue alone: a line that is
clearly a continuation of the previous turn, a question/answer pair where the
answer got the questioner's label, or one stray line that breaks an otherwise
clean back-and-forth. A language model reading the ordered transcript can spot
and reassign those — *without hearing the audio*, so it only ever reshuffles
among the speakers diarization already found; it never invents or merges them.

If anything goes wrong the caller keeps the original labels — this stage must
never break a transcript. See `backend/llm.py` for the shared model plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.llm import (
    DEFAULT_LLM_MODEL,
    extract_json_array,
    generate_chat,
    iter_windows,
    load_model,
    mlx_lm_available,  # re-exported for callers/tests
)

__all__ = [
    "Correction",
    "apply_corrections",
    "build_prompt_messages",
    "correct_speaker_labels",
    "mlx_lm_available",
    "parse_corrections",
]


@dataclass
class Correction:
    line: int  # global index into the transcript
    speaker: str  # an existing diariser label, e.g. "SPEAKER_01"


_SYSTEM_PROMPT = (
    "You correct speaker labels in a diarized transcript. Each line is prefixed "
    "with the speaker the diarizer assigned, which is sometimes wrong. Using only "
    "the conversational flow and content, find lines whose speaker label is "
    "clearly wrong and give the correct speaker.\n"
    "Rules:\n"
    "- Use ONLY the speaker IDs listed; never invent, rename, or merge speakers.\n"
    "- Only flag a line when the dialogue makes the right speaker clear (e.g. an "
    "obvious continuation of the previous turn, or the answer side of a Q&A).\n"
    "- Leave a line alone if you are unsure — do not guess.\n"
    "- Output STRICT JSON only: an array of {\"line\": <int>, \"speaker\": "
    "\"<SPEAKER_ID>\"} objects, and nothing else. Empty array [] if all correct."
)


def build_prompt_messages(
    texts: list[str],
    speakers: list[str],
    speaker_ids: list[str],
    win_start: int,
    win_end: int,
    core_start: int,
    core_end: int,
) -> list[dict[str, str]]:
    """Build chat messages for one window.

    Lines `win_start:win_end` are shown for context; only `core_start:core_end`
    (the editable range, end-exclusive) may be corrected. Numbering is global so
    corrections refer to absolute transcript indices.
    """
    rendered = "\n".join(
        f"{i}: ({speakers[i]}) {texts[i]}" for i in range(win_start, win_end)
    )
    user = (
        f"Speakers present: {', '.join(speaker_ids)}\n\n"
        f"Transcript:\n{rendered}\n\n"
        f"Only output corrections for lines {core_start} to {core_end - 1} inclusive. "
        f"Lines outside that range are context only. Return the JSON array now."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_corrections(
    raw: str,
    valid_speakers: set[str],
    min_line: int,
    max_line: int,
) -> list[Correction]:
    """Parse model output into validated corrections within [min_line, max_line]."""
    out: list[Correction] = []
    seen: set[int] = set()
    for item in extract_json_array(raw):
        if not isinstance(item, dict):
            continue
        line = item.get("line")
        speaker = item.get("speaker")
        if not isinstance(line, int):
            try:
                line = int(line)  # tolerate "12"
            except (TypeError, ValueError):
                continue
        if line < min_line or line > max_line:
            continue
        if speaker not in valid_speakers:
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(Correction(line=line, speaker=speaker))
    return out


def apply_corrections(
    speakers: list[str],
    corrections: list[Correction],
) -> tuple[list[str], int]:
    """Apply corrections to a copy of `speakers`; return it and the number changed."""
    updated = list(speakers)
    changed = 0
    for c in corrections:
        if 0 <= c.line < len(updated) and updated[c.line] != c.speaker:
            updated[c.line] = c.speaker
            changed += 1
    return updated, changed


def correct_speaker_labels(
    texts: list[str],
    speakers: list[str],
    *,
    model_repo: str = DEFAULT_LLM_MODEL,
    window: int = 60,
    context: int = 8,
    max_tokens: int = 800,
) -> tuple[list[str], int]:
    """Re-attribute mislabeled lines with a local LLM.

    `texts` and `speakers` are parallel lists (one entry per transcript segment).
    Returns the corrected speaker list and how many labels changed. Long scenes
    are processed in windows; corrections feed forward so later windows see the
    already-fixed labels.
    """
    speaker_ids = sorted({s for s in speakers if s})
    if len(speaker_ids) < 2:
        return list(speakers), 0  # nothing to disambiguate

    model, tokenizer = load_model(model_repo)
    valid = set(speaker_ids)
    updated = list(speakers)
    total_changed = 0

    for win_start, core_start, core_end in iter_windows(len(texts), window, context):
        messages = build_prompt_messages(
            texts, updated, speaker_ids, win_start, core_end, core_start, core_end
        )
        raw = generate_chat(model, tokenizer, messages, max_tokens)
        corrections = parse_corrections(raw, valid, core_start, core_end - 1)
        updated, changed = apply_corrections(updated, corrections)
        total_changed += changed

    return updated, total_changed
