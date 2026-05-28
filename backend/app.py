"""FastAPI app: upload audio, transcribe + diarize locally, export Avid markers.

Run with:  uvicorn backend.app:app --reload
"""

from __future__ import annotations

import os
import re
import threading
import uuid
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.avid_markers import MarkerSettings, Segment, build_markers
from backend.timecode_utils import DEFAULT_FPS, validate_timecode

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Load local secrets (HF_TOKEN) from a gitignored .env. A value already exported
# in the shell takes precedence (load_dotenv doesn't override existing env vars).
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")

app = FastAPI(title="Avid Markup")

# job_id -> {status, result, error, audio_filename}
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _run_transcription(
    job_id: str,
    audio_path: Path,
    model_size: str,
    min_speakers: int | None,
    max_speakers: int | None,
    llm_correct: bool,
    force_language: str | None,
) -> None:
    _set_job(job_id, status="running")

    def _progress(msg: str) -> None:
        _set_job(job_id, stage=msg)

    try:
        from backend.transcribe import transcribe_and_diarize

        result = transcribe_and_diarize(
            str(audio_path),
            model_size=model_size,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            llm_correct=llm_correct,
            force_language=force_language,
            progress_cb=_progress,
        )
        _set_job(
            job_id,
            status="done",
            stage="",
            result={
                "language": result.language,
                "segments": [asdict(s) | {"include": True} for s in result.segments],
                "speakers": [asdict(s) for s in result.speakers],
                "corrected_speakers": result.corrected_speakers,
            },
        )
    except Exception as exc:  # surface any ML/setup failure to the UI
        _set_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")


@app.get("/api/config")
def config() -> dict:
    from backend.llm import mlx_lm_available
    from backend.transcribe import _asr_engine, _parakeet_available

    return {
        "diarization_enabled": bool(os.environ.get("HF_TOKEN")),
        "llm_available": mlx_lm_available(),
        "asr_engine": _asr_engine(),
        "parakeet_available": _parakeet_available(),
        "model_size": DEFAULT_MODEL_SIZE,
        "default_fps": DEFAULT_FPS,
    }


# Real HuggingFace tokens are "hf_" followed by alphanumerics. Validating the shape
# also blocks newline/injection into the .env file written below.
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]+")


class TokenRequest(BaseModel):
    token: str


def _write_env_token(token: str, env_path: Path) -> None:
    """Persist HF_TOKEN to the gitignored .env, replacing any existing line so the
    setting survives a restart. Other lines (e.g. comments) are preserved."""
    lines = []
    if env_path.exists():
        lines = [ln for ln in env_path.read_text().splitlines() if not ln.startswith("HF_TOKEN=")]
    lines.append(f"HF_TOKEN={token}")
    env_path.write_text("\n".join(lines) + "\n")


@app.post("/api/token")
def set_token(req: TokenRequest) -> dict:
    """Save the user's HuggingFace token from the UI so they never touch a terminal.

    Writes it to the local gitignored .env and applies it to the running server
    immediately (the next transcription picks it up — no restart). The token never
    leaves this machine."""
    token = req.token.strip()
    if not _HF_TOKEN_RE.fullmatch(token):
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like a HuggingFace token — it should start with 'hf_'.",
        )
    _write_env_token(token, ENV_PATH)
    os.environ["HF_TOKEN"] = token
    return {"diarization_enabled": True}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form(DEFAULT_MODEL_SIZE),
    min_speakers: int | None = Form(None),
    max_speakers: int | None = Form(None),
    llm_correct: bool = Form(True),
    language: str = Form(""),
) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex
    job_dir = UPLOADS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / f"audio{ext}"
    audio_path.write_bytes(await file.read())

    _set_job(job_id, status="pending", audio_filename=audio_path.name, original_name=file.filename)

    thread = threading.Thread(
        target=_run_transcription,
        args=(job_id, audio_path, model_size, min_speakers, max_speakers, llm_correct,
              language.strip() or None),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "audio_url": f"/media/{job_id}/{audio_path.name}"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage", ""),
        "result": job.get("result"),
        "error": job.get("error"),
    }


class TriageSegment(BaseModel):
    speaker: str = ""
    text: str


class TriageRequest(BaseModel):
    segments: list[TriageSegment]
    level: str = "balanced"
    guidance: str = ""


@app.post("/api/triage")
def triage(req: TriageRequest) -> dict:
    """Decide which grouped lines are low-value and should be dropped from the
    markers. Returns the indices to drop (into the submitted list); the client
    un-ticks them so the choice stays visible and reversible."""
    from backend.llm import mlx_lm_available

    # "dedup" is deterministic and needs no model; everything else needs the LLM.
    if req.level != "dedup" and not mlx_lm_available():
        raise HTTPException(status_code=503, detail="Local LLM unavailable (install the ML extra).")
    from backend.mlx_runtime import run_on_mlx_thread
    from backend.triage import triage_segments

    texts = [s.text for s in req.segments]
    speakers = [s.speaker for s in req.segments]
    try:
        drop = run_on_mlx_thread(
            triage_segments, texts, speakers, level=req.level, guidance=req.guidance
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Triage failed: {type(exc).__name__}: {exc}") from exc
    return {"drop": drop, "total": len(texts), "dropped": len(drop), "kept": len(texts) - len(drop)}


class ExportSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = ""
    include: bool = True


class ExportRequest(BaseModel):
    start_tc: str
    fps: int = DEFAULT_FPS
    author: str = "Transcriber"
    track: str = "V1"
    label_separator: str = " - "
    speaker_labels: dict[str, str] = {}
    color_mode: str = "single"
    single_color: str = "Yellow"
    speaker_colors: dict[str, str] = {}
    duration_mode: str = "point"
    segments: list[ExportSegment]
    filename: str = "markers"


def _render_markers(req: ExportRequest) -> tuple[str, str]:
    """Build marker text and the download filename. Raises ValueError on bad TC."""
    start_tc = validate_timecode(req.start_tc, req.fps)
    settings = MarkerSettings(
        start_tc=start_tc,
        fps=req.fps,
        author=req.author,
        track=req.track,
        label_separator=req.label_separator,
        speaker_labels=req.speaker_labels,
        color_mode="per_speaker" if req.color_mode == "per_speaker" else "single",
        single_color=req.single_color,
        speaker_colors=req.speaker_colors,
        duration_mode="span" if req.duration_mode == "span" else "point",
    )
    segments = [
        Segment(start=s.start, end=s.end, text=s.text, speaker=s.speaker, include=s.include)
        for s in req.segments
    ]
    text = build_markers(segments, settings)
    safe_name = Path(req.filename).stem or "markers"
    return text, f"{safe_name}_markers.txt"


# Prepared exports awaiting download (token -> {text, filename}).
_export_cache: dict[str, dict[str, str]] = {}


@app.post("/api/export")
def export_markers(req: ExportRequest) -> PlainTextResponse:
    try:
        text, filename = _render_markers(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PlainTextResponse(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/export_prepare")
def export_prepare(req: ExportRequest) -> dict:
    """Build the file, stash it, and return a GET URL ending in the filename.

    Downloading via a GET whose path ends in `..._markers.txt` guarantees the
    saved filename in every browser — Safari ignores blob/Content-Disposition
    filename hints and names downloads with a UUID otherwise."""
    try:
        text, filename = _render_markers(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    token = uuid.uuid4().hex
    _export_cache[token] = {"text": text, "filename": filename}
    return {"download_url": f"/api/download/{token}/{filename}", "filename": filename}


@app.get("/api/download/{token}/{filename}")
def download_export(token: str, filename: str) -> PlainTextResponse:
    entry = _export_cache.pop(token, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Export expired — generate it again")
    return PlainTextResponse(
        content=entry["text"],
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{entry["filename"]}"'},
    )


# Range-capable media serving (StaticFiles supports HTTP Range for audio seeking).
app.mount("/media", StaticFiles(directory=UPLOADS_DIR), name="media")


@app.get("/")
def index() -> FileResponse:
    # Never cache the HTML shell, or the browser keeps an old page after an update
    # (the ?v= query only busts the linked JS/CSS, not the document itself).
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# SPA assets (app.js, styles.css). Mounted last so it doesn't shadow /api or /media.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
