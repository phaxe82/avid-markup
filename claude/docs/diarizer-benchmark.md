# Diarizer A/B benchmark — sherpa-onnx (token-free) vs pyannote community-1

**Goal:** find a token-free, redistributable diarizer good enough to be the *default*
for the public build, so speaker labelling works with no HuggingFace account/token.

**Method:** `claude/scripts/bench_diarizers.py`. Reference = pyannote
`speaker-diarization-community-1` (current production model, needs a token). Two real
5-min scene clips. Metrics over 100 ms co-voiced frames:
- **1to1**: agreement under best one-to-one speaker map (Hungarian) — penalises wrong
  speaker counts / identity swaps.
- **purity**: fraction of frames whose sherpa cluster majority-matches one pyannote
  speaker (many-to-one) — high purity + over-clustering is recoverable by the app's
  label-then-merge UX; impurity (mixing two people in one cluster) is not.

## Results (clip 1, noisy reality-TV; pyannote = 4 speakers)

| embedding | threshold | speakers | 1to1 | purity |
|---|---|---|---|---|
| CAM++ (3dspeaker campplus) | 0.5–0.9 | 8–20 | 27–42% | 53–63% |
| eres2net | 0.4–0.8 | 33–56 | 21–48% | 90–92% |
| **wespeaker ResNet34** | **0.60** | **4** | **72.5%** | 74.9% |
| wespeaker ResNet34 | 0.70 | 3 | 74.8% | 74.8% |

## Results (clip 2, different content; pyannote = 4 speakers)

| embedding | threshold | speakers | 1to1 | purity |
|---|---|---|---|---|
| wespeaker ResNet34 | 0.60 | 6 | 46.6% | 64.7% |
| **wespeaker ResNet34** | **0.65** | **4** | **67.2%** | 67.4% |
| wespeaker ResNet34 | 0.70 | 3 | 67.4% | 67.4% |

## Decision

**Default: WeSpeaker ResNet34 (VoxCeleb, Apache-2.0) @ threshold 0.65.**
- Matched pyannote's speaker count (4/4) on both clips; ~67–73% frame agreement.
- CAM++ over-clusters *impurely* (mixes speakers); eres2net is pure but fragments into
  30–56 clusters (unusable). ResNet34 is the only one that lands near pyannote.
- 0.65 over 0.60/0.70: 0.60 over-split clip 2 (6 spk), 0.70 under-split clip 1 (3 spk) —
  under-splitting is unrecoverable in the UI, so we lean to the slightly-higher-cluster
  side. Tunable via `AVID_DIARIZE_THRESHOLD`.

**Runtime:** ~30 s for 5 min on CPU (~0.1× realtime → ~9 min for a 47-min scene),
comparable to pyannote-on-MPS (~6 min) and with no MPS-op fragility. onnxruntime only.

**Caveats / residual gap:** pyannote is the *reference*, not ground truth, and these are
noisy clips — 67–73% agreement is a real but acceptable step down, absorbed downstream by
the LLM speaker-correction pass and manual reassignment. pyannote stays available as an
opt-in higher-accuracy path (`AVID_DIARIZER=pyannote`, needs a token).

## Follow-up: threshold clustering over-splits on long scenes

The 5-min benchmarks above hid a problem that only shows on full scenes. On a real
**63-minute, 2-speaker** scene, threshold clustering badly over-splits as the same two
voices drift over the hour:

| mode (12-min chunk of that scene) | speakers |
|---|---|
| threshold 0.65 (default) | 12 |
| threshold 0.72 | 8 |
| threshold 0.80 | 7 |
| threshold 0.88 | 4 |
| **force num_clusters = 2** | **2** ✓ |

No single threshold fixes it (even 0.88 → 4 for 2 speakers), and raising it globally risks
*merging* distinct speakers (unrecoverable). **Fix:** `_clustering_params` now uses any
speaker-count hint (UI Min/**Max** speakers) as the exact `num_clusters`, and the UI nudges
users to enter the cast size. With no hint we keep threshold 0.65 (over-clustering is
recoverable via label-then-merge; under-clustering is not).
