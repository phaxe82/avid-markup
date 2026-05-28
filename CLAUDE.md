# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline web app that transcribes a TV scene's audio, separates and labels the speakers, and exports a tab-delimited marker file that Avid Media Composer imports onto a sequence. Built for 30–60 minute scenes. Everything runs on-machine (no audio leaves the box) — important because the source is often embargoed broadcast rushes.

## Commands

```bash
# Setup (Python 3.10–3.12; the ML stack does not support 3.13+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"        # core + WhisperX/mlx + tests
pip install -e ".[dev]"           # logic work only — skips the heavy ML stack
brew install ffmpeg               # required by WhisperX to decode audio

# Run the app
export HF_TOKEN=hf_...             # required for diarization (see Gotchas)
uvicorn backend.app:app --port 8000

# Tests
pytest                                                   # full suite (no models loaded)
pytest tests/test_avid_markers.py::test_format_matches_reference_bytes   # single test

# Exercise the real ML pipeline end-to-end on an audio file
HF_TOKEN=... python claude/scripts/smoke_transcribe.py <audio.mp3> [model_size]

# Force the slow path when debugging GPU issues
AVID_DISABLE_MLX=1        # CPU faster-whisper instead of GPU mlx-whisper
AVID_TORCH_DEVICE=cpu     # CPU alignment + diarization instead of MPS
AVID_DISABLE_LLM=1                    # master kill-switch for ALL LLM passes (correction + triage)
AVID_DISABLE_LLM_CORRECT=1            # skip only the speaker-correction pass
AVID_LLM_MODEL=mlx-community/...      # override the LLM both passes use (default Qwen2.5-7B-Instruct-4bit)
```

## Architecture

The flow is a 4-stage pipeline orchestrated by `backend/transcribe.transcribe_and_diarize`, then turned into markers:

1. **Transcribe** — `mlx-whisper` on the GPU when available (`_mlx_available()`), else `faster-whisper` on CPU. mlx is preferred because faster-whisper (CTranslate2) has no Metal backend.
2. **Align** (`whisperx.align`) and **diarize** (`pyannote`) — both run on `_pick_torch_device()` (MPS if available, else CPU). **Diarization dominates runtime on long files**, so getting it onto MPS is the single biggest speedup (~47 min → ~6 min on a 47-min file).
2b. **Correct speakers (optional)** — `backend/speaker_correction.correct_speaker_labels` runs a local `mlx-lm` LLM over the diarized fragments and reassigns obvious mislabels (pyannote is ~70% accurate; many errors are clear from dialogue flow). It only reshuffles among existing speaker IDs, never invents/merges them, runs in overlapping windows, and is best-effort — any failure silently keeps the original labels. Gated by the `llm_correct` flag (UI checkbox) + `mlx_lm_available()`. Operates on **raw fragments before client-side grouping**, so grouping then merges any newly-agreeing neighbours.
3. **Group** — done **client-side** in `frontend/app.js` (`groupSegments`): the server returns raw WhisperX fragments; the browser merges them into "whole speaker turn" (default) / "complete sentences" / "raw" markers live, with no re-transcription. Merging keys on the *resolved* speaker (`speakerKey` = the user's label, else the raw `SPEAKER_xx` id), so a person diarization split across several ids merges into one turn once they're labelled the same name — committing a name (`change` event) re-runs `applyGrouping`. Export sends the grouped+edited segments.
3b. **Trim low-value lines (optional)** — `backend/triage.triage_segments` (via `POST /api/triage`, the "Trim low-value lines (AI)" button in step 3) returns the indices of **grouped** segments to drop; the client *un-ticks* those (`include=false`) — non-destructive, reviewable, re-runnable after a grouping change. **Deterministic dedup runs at every level** (`find_duplicate_indices`: exact multi-word line repeats within a lookback window — catches mlx-whisper's doubles, never removes a unique line). Levels: `dedup` (dedup only, **no LLM** — needs no model, allowed even when the LLM is unavailable), `light` (dedup + LLM removes only out-of-context/non-dialogue, capped at `_LIGHT_LLM_CAP_FRACTION` or it falls back to dedup-only), `balanced`, `aggressive`. Triage never re-includes a line (manual keeps / crew exclusions survive), and `window_drops_plausible` discards any window that tries to drop *everything*.
4. **Render markers** — `backend/avid_markers.build_markers` produces the Avid file.

`backend/app.py` is a FastAPI server: upload → background-thread transcription with `/api/jobs/{id}` polling → client-side labeling/grouping/editing → export. Job state and prepared exports live in in-memory dicts, so a server restart drops them.

### The marker file format is the core contract — do not change casually

8 tab-separated fields, LF line endings, no header row, reverse-engineered byte-for-byte from a real Avid export:

```
Name <TAB> Timecode <TAB> Track <TAB> Color <TAB> Comment <TAB> Duration <TAB> (empty) <TAB> Color
```

Color is capitalized and appears **twice** (fields 4 and 8); field 7 is empty. `tests/test_avid_markers.py::test_format_matches_reference_bytes` asserts the exact bytes — it is the regression guard. Marker timecode = scene start TC + segment offset at 25 fps non-drop (`backend/timecode_utils.py`, via the `timecode` lib).

### Non-obvious things that will bite you

- **Whisper mis-detects language → alignment can have no model.** On music/ambiguous audio Whisper often auto-detects the wrong language (e.g. Welsh `cy` on English), and `whisperx.load_align_model` then raises `ValueError: No default align-model for language: <x>`. Mitigations in place: the UI **Language** selector defaults to English and passes `force_language` (skips detection, also fixes the wrong transcription); and `load_align_model`/`align` are wrapped so a missing model **skips alignment gracefully** (keeps Whisper segment-level timestamps — `assign_word_speakers` still labels each segment by overlap; only sub-segment word timing is lost, which markers don't use).
- **HF token + gated model.** Diarization needs `HF_TOKEN` set *and* the user to have accepted the conditions for `pyannote/speaker-diarization-community-1` on huggingface.co. A valid token still 403s until the model terms are accepted. Without a token, transcription still works but every line is one unlabeled speaker.
- **Model pinning.** The diariser is pinned to `community-1` (`DEFAULT_DIARIZE_MODEL`). The older `speaker-diarization-3.1` does **not** load cleanly on the installed pyannote 4.x. The auth kwarg (`token` vs `use_auth_token`) is auto-detected by inspecting the signature, since it changed across whisperx versions.
- **Downloads must put the filename in the URL path.** Export is two-step: POST `/api/export_prepare` stashes the text and returns a GET URL ending in `..._markers.txt`; the browser navigates there. This is because Safari ignores blob `download` and `Content-Disposition` filename hints and saves files with a UUID and no extension. `/api/export` (JSON → PlainTextResponse) still exists for tests.
- **Bump the asset version when editing the frontend.** `index.html` references `app.js?v=N` / `styles.css?v=N`. Increment `N` on changes or Safari serves stale JS (this previously masked working fixes as "not working").
- **All MLX work must run on one thread** — mlx-whisper transcription *and* the mlx-lm passes. MLX binds its GPU stream to the thread that first touches it; using a model/array on another thread throws `RuntimeError: There is no Stream(gpu, 0) in current thread`. The server otherwise touches MLX from the transcription job's daemon thread (whisper + correction) and FastAPI's threadpool workers (triage), so every MLX call is funnelled through `backend.mlx_runtime.run_on_mlx_thread` (a `max_workers=1` executor): `_transcribe_mlx`, `correct_speaker_labels`, `triage_segments`. Two corollaries: (1) wrap the *whole* top-level call, never an inner helper, and never call it from the MLX thread itself (deadlocks the single worker); (2) availability checks (`mlx_lm_available`, `_mlx_available`) use `importlib.util.find_spec` and must **not** `import mlx*`, since importing initialises MLX on the calling (wrong) thread.
- **`mlx-lm` is version-capped (`<0.30`).** mlx-lm 0.30+ requires `transformers>=5` and `huggingface-hub>=1`, but whisperx pins `huggingface-hub<1.0.0` — installing the latest mlx-lm silently upgrades both and breaks model loading. If you bump the cap, run `pip check` and confirm whisperx still loads.
- **Lazy ML imports.** `transcribe.py` and `backend/llm.py` import `whisperx`/`torch`/`mlx_whisper`/`mlx_lm` only inside functions so the app and the test suite run without the `.[ml]` stack. Keep it that way: tests cover pure logic (timecode, marker bytes, export/download endpoints, grouping, device selection, correction + triage parse/apply/windowing/guards) and must never load models. `speaker_correction.py` and `triage.py` keep their prompt-building, JSON parsing, window math, and guards as plain functions so they stay unit-testable without the LLM; only the `*_segments` orchestrators touch the model.

Throwaway dev scripts and experiments go in `claude/` (e.g. `claude/scripts/smoke_transcribe.py`), never the project root.
