# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline web app that transcribes a TV scene's audio, separates and labels the speakers, and exports a tab-delimited marker file that Avid Media Composer imports onto a sequence. Built for 30–60 minute scenes. Everything runs on-machine (no audio leaves the box) — important because the source is often embargoed broadcast rushes.

## Commands

```bash
# Setup (Python 3.10–3.12; the ML stack does not support 3.13+)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]"        # core + WhisperX/mlx/sherpa-onnx + tests
pip install -e ".[dev]"           # logic work only — skips the heavy ML stack
./scripts/fetch_diarization_models.sh   # token-free diarizer models (~34 MB; for the [ml] path)
# No ffmpeg needed: audio is decoded in-process by PyAV. (Only the experimental Parakeet
# engine still shells out to the ffmpeg CLI — `brew install ffmpeg` for that.)

# Run the app — diarization works with NO HF token by default (sherpa-onnx)
uvicorn backend.app:app --port 8000

# Tests
pytest                                                   # full suite (no models loaded)
pytest tests/test_avid_markers.py::test_format_matches_reference_bytes   # single test

# Exercise the real ML pipeline end-to-end on an audio file (no token needed)
python claude/scripts/smoke_transcribe.py <audio.mp3> [model_size]
# A/B the diarizers (sherpa vs pyannote, needs HF_TOKEN for the reference)
python claude/scripts/bench_diarizers.py <audio> [thresholds...]

# Force the slow path when debugging GPU issues
AVID_DISABLE_MLX=1        # CPU faster-whisper instead of GPU mlx-whisper
AVID_TORCH_DEVICE=cpu     # CPU alignment (+ pyannote diarization) instead of MPS
AVID_DISABLE_LLM=1                    # master kill-switch for ALL LLM passes (correction + triage)
AVID_DISABLE_LLM_CORRECT=1            # skip only the speaker-correction pass
AVID_LLM_MODEL=mlx-community/...      # override the LLM both passes use (default Qwen2.5-7B-Instruct-4bit)

# Diarizer selection (default: sherpa-onnx, token-free — see Architecture stage 2)
AVID_DIARIZER=pyannote               # opt into pyannote instead (needs HF_TOKEN + accepted gated terms)
AVID_DIARIZE_THRESHOLD=0.65          # sherpa clustering threshold (lower = more speakers)
AVID_DIARIZE_MODEL_DIR=...           # override the diarization model dir (default: bundled / repo models/)

# Alternative ASR engine (English-only) — A/B Parakeet against the default Whisper
AVID_ASR_ENGINE=parakeet             # use NVIDIA Parakeet instead of Whisper (default: whisper)
AVID_PARAKEET_MODEL=mlx-community/... # override the Parakeet model (default parakeet-tdt-0.6b-v2)
AVID_DISABLE_PARAKEET=1              # force-disable Parakeet even if AVID_ASR_ENGINE=parakeet

# Build the one-download macOS .app (Apple Silicon)
pip install -e ".[ml,app,build]"
./packaging/build_app.sh             # ad-hoc local build (validate the freeze on this Mac)
./packaging/build_app.sh --release   # Developer-ID sign + notarize + .dmg (needs Apple Developer acct)
```

## Architecture

The flow is a 4-stage pipeline orchestrated by `backend/transcribe.transcribe_and_diarize`, then turned into markers. Audio is decoded once up front by `backend/audio.load_audio` (PyAV / **LGPL** FFmpeg libraries, **in-process — no ffmpeg CLI**) and the resulting 16 kHz mono float32 array is passed to every stage; mlx-whisper accepts the array directly (skipping its own ffmpeg subprocess), which is what keeps the frozen app ffmpeg-free.

1. **Transcribe** — `mlx-whisper` on the GPU when available (`_mlx_available()`), else `faster-whisper` on CPU. mlx is preferred because faster-whisper (CTranslate2) has no Metal backend. **Optional alternative engine:** set `AVID_ASR_ENGINE=parakeet` to use NVIDIA Parakeet (`_transcribe_parakeet`, English-only) instead — it supplies its own timestamps (so the align step is skipped) and rarely hallucinates over music/silence, but loses Whisper's multi-language support. Off by default; both engines feed the same diarization + grouping + export downstream. Parakeet is another MLX model, so it runs through `run_on_mlx_thread` like mlx-whisper, and `_parakeet_available()` checks via `find_spec` (no import off the MLX thread).
2. **Align** (`whisperx.align`, Whisper path only, on `_pick_torch_device()` — MPS if available) and **diarize**. The diarizer is chosen by `_diarizer()` (`AVID_DIARIZER`, default `sherpa`):
   - **Default — sherpa-onnx** (`backend/diarize_sherpa.py`, `_sherpa_available()`): **token-free**, runs on `onnxruntime`/CPU (no MLX thread, no MPS), with bundled redistributable models (MIT pyannote segmentation + Apache WeSpeaker ResNet34, chosen by A/B benchmark — see `claude/docs/diarizer-benchmark.md`). Returns `(start, end, SPEAKER_xx)` intervals fed to `_assign_speakers_by_overlap` for **both** ASR engines. ~9 min on a 47-min file (CPU).
   - **Opt-in — pyannote** (`AVID_DIARIZER=pyannote`): needs `HF_TOKEN` + accepted gated terms; runs on `_pick_torch_device()` (MPS, ~6 min on a 47-min file), uses `assign_word_speakers` (Whisper) or overlap (Parakeet). Also the automatic fallback when sherpa is unavailable *and* a token is present.
   - If neither is available, diarization is skipped (every line one unlabeled speaker — still a usable transcript).
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
- **Diarization is token-free by default (sherpa-onnx); HF token only for the opt-in pyannote path.** The default needs no token and no account. The `AVID_DIARIZER=pyannote` path needs `HF_TOKEN` set *and* the user to have accepted the conditions for `pyannote/speaker-diarization-community-1` on huggingface.co — a valid token still 403s until the terms are accepted. `/api/config` reports `diarization_enabled`/`diarizer`/`token_set` by mirroring `transcribe`'s selection, so keep the two in sync if you change the gating.
- **Sherpa models must be present on disk.** `_sherpa_available()` checks `find_spec("sherpa_onnx")` **and** that the models exist (`_resolve_models`). They're fetched by `scripts/fetch_diarization_models.sh` into `models/` (gitignored) and bundled into the .app; `_model_dir()` resolves them via `backend.paths.resource_dir()` (the bundle when frozen, repo root in dev). Sherpa runs on `onnxruntime`/CPU — **not** the MLX thread — so it needs no `run_on_mlx_thread`.
- **PyAV decode — pass arrays, not paths.** `backend/audio.load_audio` decodes via PyAV's LGPL FFmpeg libs (no ffmpeg CLI). mlx-whisper and faster-whisper are given the **array** so they skip their own ffmpeg subprocess; if you add a stage, feed it the array too. (The prebuilt `imageio-ffmpeg` binary is GPL — deliberately not used. The experimental Parakeet engine still `shutil.which("ffmpeg")`s, so it alone needs the CLI.)
- **Frozen app = read-only bundle + writable Application Support.** `backend/paths.py`: resources (frontend, models) come from `resource_dir()` (the bundle / `sys._MEIPASS`); writable state (uploads, the `.env` holding any token) goes to `data_dir()` (`~/Library/Application Support/AvidMarkup` when frozen, repo root in dev). Writing inside the signed bundle fails. The frozen binary can't `python -m uvicorn`, so `launcher.py` re-execs itself with `--serve` (see `packaging/` for the PyInstaller spec + entitlements; UPX is off — it breaks notarization).
- **pyannote model pinning (opt-in path).** Pinned to `community-1` (`DEFAULT_DIARIZE_MODEL`); the older `speaker-diarization-3.1` does **not** load cleanly on pyannote 4.x. The auth kwarg (`token` vs `use_auth_token`) is auto-detected by inspecting the signature, since it changed across whisperx versions.
- **Downloads must put the filename in the URL path.** Export is two-step: POST `/api/export_prepare` stashes the text and returns a GET URL ending in `..._markers.txt`; the browser navigates there. This is because Safari ignores blob `download` and `Content-Disposition` filename hints and saves files with a UUID and no extension. `/api/export` (JSON → PlainTextResponse) still exists for tests.
- **Bump the asset version when editing the frontend.** `index.html` references `app.js?v=N` / `styles.css?v=N`. Increment `N` on changes or Safari serves stale JS (this previously masked working fixes as "not working").
- **All MLX work must run on one thread** — mlx-whisper transcription *and* the mlx-lm passes. MLX binds its GPU stream to the thread that first touches it; using a model/array on another thread throws `RuntimeError: There is no Stream(gpu, 0) in current thread`. The server otherwise touches MLX from the transcription job's daemon thread (whisper + correction) and FastAPI's threadpool workers (triage), so every MLX call is funnelled through `backend.mlx_runtime.run_on_mlx_thread` (a `max_workers=1` executor): `_transcribe_mlx`, `correct_speaker_labels`, `triage_segments`. Two corollaries: (1) wrap the *whole* top-level call, never an inner helper, and never call it from the MLX thread itself (deadlocks the single worker); (2) availability checks (`mlx_lm_available`, `_mlx_available`) use `importlib.util.find_spec` and must **not** `import mlx*`, since importing initialises MLX on the calling (wrong) thread.
- **`mlx-lm` is version-capped (`<0.30`).** mlx-lm 0.30+ requires `transformers>=5` and `huggingface-hub>=1`, but whisperx pins `huggingface-hub<1.0.0` — installing the latest mlx-lm silently upgrades both and breaks model loading. If you bump the cap, run `pip check` and confirm whisperx still loads.
- **Lazy ML imports.** `transcribe.py`, `backend/llm.py`, `backend/diarize_sherpa.py`, and `backend/audio.py` import `whisperx`/`torch`/`mlx_whisper`/`mlx_lm`/`sherpa_onnx`/`av` only inside functions so the app and the test suite run without the `.[ml]` stack. Keep it that way: tests cover pure logic (timecode, marker bytes, export/download endpoints, grouping, device selection, diarizer selection + sherpa model-path/clustering logic, correction + triage parse/apply/windowing/guards) and must never load models. `speaker_correction.py`, `triage.py`, and `diarize_sherpa.py` keep their prompt-building / JSON parsing / window math / guards / path + clustering helpers as plain functions so they stay unit-testable without the heavy stack; only the orchestrators touch the model.

Throwaway dev scripts and experiments go in `claude/` (e.g. `claude/scripts/smoke_transcribe.py`), never the project root.
