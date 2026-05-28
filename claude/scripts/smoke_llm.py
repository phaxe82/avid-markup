"""Smoke-test the local-LLM passes (speaker correction + fluff triage) end to end.

Uses a SMALL model by default so it's a quick download — this validates the
plumbing (model load, chat template, generation, JSON parsing, window loop), not
the quality you'll get from the production 7B model. Run quality checks on real
audio with the default model.

    AVID_LLM_MODEL=mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        python claude/scripts/smoke_llm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("AVID_LLM_MODEL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit")

from backend.speaker_correction import correct_speaker_labels  # noqa: E402
from backend.triage import triage_segments  # noqa: E402

# A tiny scripted exchange: line 3 is mislabeled (clearly B continuing), and a
# few lines are pure filler/backchannel that a trim should drop.
texts = [
    "So tell me what happened that night.",
    "I was at home, watching television.",
    "And then the phone rang around eleven.",  # spoken by B but mislabeled A
    "Mm-hmm.",
    "Right.",
    "It was my sister. She said the house was on fire.",
    "Okay.",
    "We need to find out who called the fire brigade.",
]
speakers = ["A", "B", "A", "A", "A", "B", "A", "A"]

print("MODEL:", os.environ["AVID_LLM_MODEL"])
print("\n-- speaker correction --")
fixed, changed = correct_speaker_labels(texts, speakers)
for i, (t, before, after) in enumerate(zip(texts, speakers, fixed)):
    flag = "  <-- changed" if before != after else ""
    print(f"{i}: {before}->{after} {t!r}{flag}")
print(f"changed: {changed}")

print("\n-- triage (balanced) --")
drop = triage_segments(texts, speakers, level="balanced")
for i, t in enumerate(texts):
    print(f"{i}: {'DROP' if i in drop else 'keep'}  {t!r}")
print(f"dropped: {len(drop)} of {len(texts)} -> {drop}")
