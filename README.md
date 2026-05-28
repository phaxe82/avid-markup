# Avid Markup

Transcribe a scene's audio, separate and label the speakers, and export a
marker file that **Avid Media Composer** imports directly onto a sequence —
instead of watching the scene and typing locators by hand.

Everything runs **locally and offline** (WhisperX + pyannote). No audio leaves
your machine, so it's safe for unreleased / embargoed rushes.

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
- **ffmpeg**: `brew install ffmpeg`
- A **HuggingFace token** for speaker diarization (one-time, free):
  1. Create a token at <https://huggingface.co/settings/tokens>.
  2. Open [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
     and click **Agree and access repository** to accept its conditions. (It's a
     self-contained pipeline — no other models need accepting.)
  3. Set it before running: `export HF_TOKEN=hf_xxx`

Without `HF_TOKEN` the app still transcribes, but every line is one unlabelled
speaker (no diarization).

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

## Share with a colleague (another Apple Silicon Mac)

There's no single installer — the GPU speed needs each Mac's own Metal, so the app
isn't a frozen bundle. Sharing means a one-time setup on their machine:

1. Push this project to a **private GitHub repo** (see below) and have them
   `git clone` it. (Cloning avoids macOS quarantining the files.)
2. They install **Python 3.10–3.12** and **ffmpeg** (`brew install ffmpeg`).
3. They run **`./setup.command`** once. It installs everything and asks for *their
   own* free HuggingFace token (each person uses their own — never share yours).
4. They double-click **`AvidMarkup.app`**.

First transcription on a new machine downloads the speech models (~1–6 GB); that
run is slower, then it's cached.

### Privacy / fully offline

All transcription, alignment, and diarization run **on your machine** — there is no
transcription/AI API and **no audio ever leaves the box**, which is the point for
embargoed rushes. `HF_TOKEN` is only a one-time download key for the gated pyannote
model; it is not a metered/paid API token.

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
