# Avid Markup

Transcribe a scene's audio, separate and label the speakers, and export a
marker file that **Avid Media Composer** imports directly onto a sequence —
instead of watching the scene and typing locators by hand.

Everything runs **locally and offline** (Whisper transcription + sherpa-onnx speaker
diarization). No audio leaves your machine, so it's safe for unreleased / embargoed
rushes — and there's no account or token to set up.

## How it works

1. Drop in the scene audio (**mp3** or wav). Pick the **language** (defaults to
   English) — leaving it on auto-detect risks Whisper mis-reading the language on
   music or ambiguous intros.
2. It transcribes and works out who's speaking (speaker diarization).
3. Play a sample of each voice and give them a name (`Sam`, `Deb`, …).
4. Tidy the transcript if you like — fix wording, reassign a line, drop a line.
5. Enter the scene's **start timecode** (25 fps PAL) and download the `.txt`.
6. Import it in Media Composer.

Each line of dialogue becomes one marker: `Speaker - the line they said`, at the
right timecode, on the track you choose.

## Requirements

- **Python 3.10–3.12** (the ML stack doesn't support 3.13+ yet).

That's it. **Speaker separation works out of the box with no account and no token** —
the default diarizer is `sherpa-onnx` with small, redistributable models (fetched by
`scripts/fetch_diarization_models.sh`, bundled into the .app). Audio is decoded in-process
by PyAV, so **no `ffmpeg` install is needed** either.

### Optional: higher-accuracy diarization with pyannote

For a marginal accuracy gain you can switch to pyannote, which needs a free
**HuggingFace token** (one-time):

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Open [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
   and click **Agree and access repository** to accept its conditions.
3. Run with `AVID_DIARIZER=pyannote HF_TOKEN=hf_xxx …` (or paste the token in the UI).

> The experimental Parakeet ASR engine (`AVID_ASR_ENGINE=parakeet`) still needs the
> `ffmpeg` CLI (`brew install ffmpeg`); the default Whisper path does not.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,dev]"      # core + WhisperX + test deps
```

The first transcription downloads the Whisper and pyannote models (~1–3 GB), and
the first run with AI speaker-correction enabled downloads the correction LLM
(~4.5 GB). All cached after that.

## Run

### As a Mac app (recommended)

Run the one-time setup, then launch by double-clicking:

```bash
./setup.command          # one time: builds the venv, installs everything, asks for your HF token
```

It creates **`AvidMarkup.app`** — double-click it (or drag it to the Dock) to open
the app in its own window. No terminal, no `uvicorn` command. Closing the window
shuts the server down.

`AvidMarkup.app` is a thin launcher: it starts the same local server as below and
points a native window (`launcher.py`, via [pywebview](https://pywebview.flowrl.com))
at it. The server, models, and your audio all stay on your machine exactly as before.

### From the terminal

```bash
uvicorn backend.app:app --port 8000
```

Open <http://127.0.0.1:8000> and follow the four steps.

The server reads your HuggingFace token from a gitignored `.env` at startup
(`cp .env.example .env`, then put your token after `HF_TOKEN=`). An `export
HF_TOKEN=hf_xxx` in the shell still works and takes precedence if set.

## Distribute it (one-download .app)

The app can be frozen into a **single signed, notarized `.dmg`** that anyone on an
Apple-Silicon Mac downloads, drags to Applications, and launches — no Python, no pip,
no `brew`, **no HuggingFace token**. Speaker separation works out of the box because the
(token-free) diarization models are bundled.

```bash
pip install -e ".[ml,app,build]"
./packaging/build_app.sh              # local build + ad-hoc sign (validate on this Mac)
./packaging/build_app.sh --release    # Developer-ID sign + notarize + .dmg (for sharing)
```

Release mode needs an Apple Developer account ($99/yr) and two env vars
(`DEVELOPER_ID`, `AC_PROFILE`) — without notarization, macOS Gatekeeper blocks a
downloaded app. The frozen bundle is ~1.3 GB; the first transcription still downloads
the Whisper speech model (~1.5 GB), then it's cached. See `packaging/` for the spec,
entitlements, and build script.

> Per-machine dev setup (no freeze) still works too: `git clone`, install
> **Python 3.10–3.12**, run **`./setup.command`**, double-click **`AvidMarkup.app`**.

### Privacy / fully offline

All transcription, alignment, and diarization run **on your machine** — there is no
transcription/AI API and **no audio ever leaves the box**, which is the point for
embargoed rushes. The default diarizer needs no token at all. (`HF_TOKEN`, if you opt
into the pyannote diarizer, is only a one-time model-download key — not a metered/paid
API token.)

After the models are cached (first run), the only remaining network activity is a
HuggingFace "is there a newer version?" version check at load time — no audio is
involved. To make it **fully air-gapped** (zero outbound connections), set these
once the models are downloaded:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The server is localhost-only and there is no telemetry.

### Performance (Apple Silicon)

Both heavy stages run on the **GPU**: transcription via **mlx-whisper** (Metal),
and diarization via **pyannote on MPS**. On a 47-minute scene:

| 47-minute scene | Time |
| --- | --- |
| CPU only (faster-whisper + CPU diarization) | ~47 min |
| GPU: mlx-whisper `medium` + MPS diarization | **~5.5 min** |

So a full 30–60 min scene is a few-minute job, not an hour — and `medium` (more
accurate) is now as fast as the old CPU `tiny`. mlx-whisper is used automatically
when installed; otherwise it falls back to CPU faster-whisper. Overrides:

- `AVID_DISABLE_MLX=1` — force the CPU faster-whisper path.
- `AVID_TORCH_DEVICE=cpu|mps` — force the device for alignment + diarization
  (diarization also falls back to CPU automatically if MPS hits an unsupported op).
- `AVID_DISABLE_LLM=1` — turn off **all** AI passes (speaker correction + trim).
- `AVID_DISABLE_LLM_CORRECT=1` — skip only the AI speaker-correction pass.
- `AVID_LLM_MODEL=...` — use a different local model for the AI passes (default
  `mlx-community/Qwen2.5-7B-Instruct-4bit`; a 3B model is lighter on RAM).

mlx-whisper sets `condition_on_previous_text=False`, which suppresses Whisper's
repeated/looped-line hallucinations over silence and music.

### Alternative engine: Parakeet (English-only, experimental)

Whisper is the default. To A/B against **NVIDIA Parakeet** — which is trained to emit
nothing over non-speech, so it rarely produces the phantom lines Whisper does on music
and silence — launch with:

```bash
AVID_ASR_ENGINE=parakeet uvicorn backend.app:app --port 8000
# or, as the native app:  AVID_ASR_ENGINE=parakeet python launcher.py
```

Same diarization, grouping, and export — only the transcription engine changes, so you
can compare the two transcripts on the same scene. Trade-offs: Parakeet is **English-only**
(no Welsh/Irish/Gaelic), and since diarization dominates runtime on long files, total time
barely changes — the win is cleaner transcripts (fewer junk markers), not speed. Requires
the `ml` extra (which now includes `parakeet-mlx`).

### Speaker accuracy (local AI correction)

Diarization gets *who said what* right most of the time, but not always. With
**Refine speaker labels with local AI** ticked (step 1, on by default), a local
LLM re-reads the finished transcript and fixes the obvious mis-attributions —
a line that clearly continues the previous turn, the answer side of a question,
a stray line that breaks an otherwise clean back-and-forth.

It runs fully on-device (no audio or text leaves the box) and only ever reshuffles
between the speakers diarization already found — it never invents or merges them.
Because it can't hear the audio, it won't catch every error, only the ones the
dialogue itself makes clear. You can still reassign any line by hand in step 3.
The status line reports how many labels it changed. Untick the box to skip it.

### Trimming the fluff (fewer, better markers)

Transcription captures *every* utterance, but you rarely want a marker on every
"um" and "yeah". In step 3, **Trim low-value lines (AI)** runs the local LLM over
the grouped lines and **un-ticks** the ones not worth a marker — it never deletes
them, so you can see what went and re-tick anything you want back. It only ever
removes lines, so your manual keeps and crew exclusions are preserved.

"Useful" is subjective, so you steer it:

- **Trim level**:
  - *Duplicates only* (default, safest) — mechanically removes mlx's repeated
    lines with **no AI judgement**, so it can never drop a unique line.
  - *Light* — duplicates plus only clearly out-of-context / non-dialogue lines;
    keeps all real dialogue, even short replies. Capped so it stays light.
  - *Balanced* — also drops filler, backchannel, and small talk.
  - *Aggressive* — keeps only the key beats.
- **What matters in this scene?** — optional free text, e.g. *"keep anything
  about the wedding, drop small talk"*.

It works on whatever grouping you've chosen, so re-run it if you change the
grouping. The status line reports how many lines it trimmed and how many markers
remain. (A safety guard ignores any attempt to drop a whole section at once.)

## Import into Avid Media Composer

1. **Tools → Markers** to open the marker window.
2. Right-click the marker list → **Import**.
3. Choose the downloaded `*_markers.txt`.

The markers land at their timecodes on the chosen track (default `V1`). The file
format is byte-matched to Avid's own marker export (tab-delimited:
`Name · Timecode · Track · Colour · Comment · Duration · · Colour`).

## Options

- **Colour** — one colour for all dialogue, or one colour per speaker.
- **Marker length** — point (1 frame) or span the spoken line.
- **Track / Author** — configurable (default `V1` / `Transcriber`).

## Develop / test

```bash
pip install -e ".[dev]"          # core + tests, skips the heavy ML stack
pytest                            # timecode, marker format, export endpoint
```

The marker-format test asserts the output **byte-for-byte** against a real Avid
export, so format regressions fail loudly.

## Layout

```
backend/
  app.py             FastAPI: upload, job status, audio streaming, export
  transcribe.py      WhisperX wrapper (transcribe + align + diarize)
  timecode_utils.py  start TC + offset -> HH:MM:SS:FF (25 fps)
  avid_markers.py    segments + labels -> Avid marker text
frontend/            single-page UI (no build step)
tests/               pytest
```
