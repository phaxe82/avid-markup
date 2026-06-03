"""A/B benchmark: sherpa-onnx (token-free) vs pyannote community-1 (reference).

Sweeps sherpa's clustering threshold and reports, for each: number of speakers,
number of segments, runtime, and frame-level agreement with pyannote (after optimal
speaker-label matching — a rough 1-DER proxy, no overlap/collar handling).

Usage:  ./.venv/bin/python claude/scripts/bench_diarizers.py <audio> [thresholds...]
The HF token is read from .env so pyannote can run as the reference.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

load_dotenv()  # HF_TOKEN for pyannote

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/diar_clip_5min.wav"
THRESHOLDS = [float(x) for x in sys.argv[2:]] or [0.5, 0.6, 0.7, 0.8]

import whisperx  # noqa: E402

from backend.diarize_sherpa import diarize_samples  # noqa: E402

samples = whisperx.load_audio(AUDIO)
total_dur = len(samples) / 16000.0
print(f"audio: {AUDIO}  ({total_dur:.1f}s)\n")


def to_frames(intervals: list[tuple[float, float, str]], hop: float = 0.1) -> list[str | None]:
    n = int(total_dur / hop) + 1
    labels: list[str | None] = [None] * n
    for start, end, spk in intervals:
        for i in range(max(0, int(start / hop)), min(n, int(end / hop) + 1)):
            labels[i] = spk
    return labels


def agreement(ref: list[str | None], hyp: list[str | None]) -> tuple[float, int]:
    """Fraction of co-voiced frames where hyp matches ref under the best 1-to-1 label
    map (Hungarian). Penalises over-clustering — measures speaker-count/identity parity."""
    pairs = [(r, h) for r, h in zip(ref, hyp) if r is not None and h is not None]
    if not pairs:
        return 0.0, 0
    refs = sorted({r for r, _ in pairs})
    hyps = sorted({h for _, h in pairs})
    ri = {r: i for i, r in enumerate(refs)}
    hi = {h: i for i, h in enumerate(hyps)}
    m = np.zeros((len(hyps), len(refs)))
    for r, h in pairs:
        m[hi[h], ri[r]] += 1
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-m)
    return float(m[rows, cols].sum()) / len(pairs), len(pairs)


def purity(ref: list[str | None], hyp: list[str | None]) -> float:
    """Hypothesis purity: fraction of co-voiced frames whose hyp cluster majority-matches
    one ref speaker (many-to-one). High purity + over-clustering = recoverable by the
    label-then-merge UX (each split is a pure subset of a real speaker)."""
    from collections import Counter, defaultdict

    pairs = [(r, h) for r, h in zip(ref, hyp) if r is not None and h is not None]
    if not pairs:
        return 0.0
    by_h: dict[str, Counter] = defaultdict(Counter)
    for r, h in pairs:
        by_h[h][r] += 1
    correct = sum(c.most_common(1)[0][1] for c in by_h.values())
    return correct / len(pairs)


# --- Reference: pyannote community-1 ---
ref_frames = None
token = os.environ.get("HF_TOKEN")
if token:
    from backend.transcribe import DEFAULT_DIARIZE_MODEL, _load_diarization_pipeline, _pick_torch_device

    dev = _pick_torch_device()
    print(f"pyannote ({DEFAULT_DIARIZE_MODEL}) on {dev} …")
    t0 = time.monotonic()
    try:
        pipe = _load_diarization_pipeline(token, dev, DEFAULT_DIARIZE_MODEL)
        df = pipe(samples)
    except Exception as exc:
        print(f"  MPS failed ({type(exc).__name__}); retry on cpu")
        pipe = _load_diarization_pipeline(token, "cpu", DEFAULT_DIARIZE_MODEL)
        df = pipe(samples)
    ref = [(float(r.start), float(r.end), str(r.speaker)) for r in df[["start", "end", "speaker"]].itertuples(index=False)]
    ref_frames = to_frames(ref)
    print(f"  speakers={len({s for _,_,s in ref})}  segments={len(ref)}  {time.monotonic()-t0:.1f}s\n")
else:
    print("no HF_TOKEN — skipping pyannote reference\n")

# --- sherpa-onnx threshold sweep ---
print(f"{'thresh':>7} {'speakers':>9} {'segments':>9} {'runtime':>8} {'1to1':>7} {'purity':>7}")
for th in THRESHOLDS:
    os.environ["AVID_DIARIZE_THRESHOLD"] = str(th)
    t0 = time.monotonic()
    segs = diarize_samples(samples, 16000)
    dt = time.monotonic() - t0
    nspk = len({s for _, _, s in segs})
    a1 = pur = ""
    if ref_frames is not None:
        hyp_frames = to_frames(segs)
        a, _ = agreement(ref_frames, hyp_frames)
        a1 = f"{a*100:5.1f}%"
        pur = f"{purity(ref_frames, hyp_frames)*100:5.1f}%"
    print(f"{th:>7.2f} {nspk:>9} {len(segs):>9} {dt:>7.1f}s {a1:>7} {pur:>7}")
