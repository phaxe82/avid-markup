"""Local transcription + speaker diarization via WhisperX.

Everything runs on-machine — nothing is uploaded. On Apple Silicon WhisperX runs
on CPU (CTranslate2 has no Metal backend), which is fine for a single scene.

`whisperx` (and its torch/pyannote stack) is imported lazily so the web app and
unit tests don't need the heavy ML dependencies present.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str  # diariser label e.g. "SPEAKER_00", or "" if diarization skipped


@dataclass
class SpeakerSample:
    """A representative clip of one speaker, for voice identification in the UI."""

    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    language: str
    segments: list[TranscriptSegment]
    speakers: list[SpeakerSample]
    corrected_speakers: int = 0  # labels the local-LLM pass reassigned


def _representative_samples(segments: list[TranscriptSegment]) -> list[SpeakerSample]:
    """Pick the longest segment per speaker — the easiest clip to recognise a voice from."""
    longest: dict[str, TranscriptSegment] = {}
    for seg in segments:
        if not seg.speaker:
            continue
        current = longest.get(seg.speaker)
        if current is None or (seg.end - seg.start) > (current.end - current.start):
            longest[seg.speaker] = seg
    samples = [
        SpeakerSample(speaker=s.speaker, start=s.start, end=s.end, text=s.text)
        for s in longest.values()
    ]
    samples.sort(key=lambda s: s.speaker)
    return samples


# whisperx 3.8 + pyannote.audio 4.x use this single self-contained pipeline
# (segmentation + embedding + PLDA bundled in one gated repo). The older
# speaker-diarization-3.1 does not load cleanly on pyannote 4.x.
DEFAULT_DIARIZE_MODEL = "pyannote/speaker-diarization-community-1"


def _load_diarization_pipeline(hf_token: str, device: str, model_name: str):
    """Import the diarization pipeline, tolerating whisperx API changes.

    The auth kwarg was renamed `use_auth_token` -> `token` around whisperx 3.8.
    """
    import inspect

    try:
        from whisperx.diarize import DiarizationPipeline  # whisperx >= 3.2
    except ImportError:
        from whisperx import DiarizationPipeline  # older layout

    params = inspect.signature(DiarizationPipeline.__init__).parameters
    token_kw = "token" if "token" in params else "use_auth_token"
    kwargs = {token_kw: hf_token, "device": device}
    if "model_name" in params and model_name:
        kwargs["model_name"] = model_name
    return DiarizationPipeline(**kwargs)


# Apple-Silicon GPU transcription via MLX. Maps our UI model sizes to the
# mlx-community model repos.
MLX_MODEL_REPOS = {
    "tiny": "mlx-community/whisper-tiny",
    "base": "mlx-community/whisper-base",
    "small": "mlx-community/whisper-small",
    "medium": "mlx-community/whisper-medium",
    "large": "mlx-community/whisper-large-v3",
    "large-v3": "mlx-community/whisper-large-v3",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def _mlx_available() -> bool:
    """True if MLX transcription should be used (Apple Silicon GPU).

    Checked via `find_spec` rather than importing `mlx_whisper`: importing it
    would initialise MLX on the calling thread, but all MLX work must happen on
    the dedicated thread (see `backend/mlx_runtime`)."""
    if os.environ.get("AVID_DISABLE_MLX"):
        return False
    import importlib.util

    return importlib.util.find_spec("mlx_whisper") is not None


def _transcribe_mlx(audio_path: str, model_size: str, language: str | None = None) -> dict:
    """Transcribe on the GPU via MLX. `condition_on_previous_text=False` curbs the
    repeated/looped-line hallucinations Whisper produces over silence and music.
    Pass `language` (e.g. "en") to skip auto-detection, which Whisper often gets
    wrong on music/ambiguous intros.
    """
    import mlx_whisper

    repo = MLX_MODEL_REPOS.get(model_size, f"mlx-community/whisper-{model_size}")
    kwargs = {"path_or_hf_repo": repo, "condition_on_previous_text": False}
    if language:
        kwargs["language"] = language
    return mlx_whisper.transcribe(audio_path, **kwargs)


# Optional alternative ASR engine: NVIDIA Parakeet on Apple Silicon (English-only).
# Off by default; enable with AVID_ASR_ENGINE=parakeet. Parakeet barely hallucinates
# over music/silence (it's trained to emit nothing on non-speech), which yields fewer
# phantom marker lines than Whisper — at the cost of Whisper's multi-language support.
DEFAULT_PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"


def _asr_engine() -> str:
    """Selected transcription engine: 'whisper' (default) or 'parakeet'."""
    return os.environ.get("AVID_ASR_ENGINE", "whisper").strip().lower()


def _parakeet_available() -> bool:
    """True if the Parakeet engine can be used. Checked via find_spec (not import) so
    MLX isn't initialised on the calling thread — see backend/mlx_runtime."""
    if os.environ.get("AVID_DISABLE_PARAKEET"):
        return False
    import importlib.util

    return importlib.util.find_spec("parakeet_mlx") is not None


def _transcribe_parakeet(audio_path: str, model_repo: str) -> dict:
    """Transcribe on the GPU via parakeet-mlx (English). Parakeet emits sentence-level
    timestamps directly, so the whisperx alignment step is skipped for this path. Long
    scenes are chunked to bound memory. Returns a whisperx-shaped dict so the rest of
    the pipeline (diarization, grouping, export) is unchanged."""
    from parakeet_mlx import from_pretrained

    model = from_pretrained(model_repo)
    out = model.transcribe(audio_path, chunk_duration=120.0, overlap_duration=15.0)
    segments = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
        for s in out.sentences
    ]
    return {"segments": segments, "language": "en"}


def _assign_speakers_by_overlap(
    segments: list[dict], diarized: list[tuple[float, float, str]]
) -> list[dict]:
    """Tag each transcript segment with the diarized speaker it overlaps most.

    Used for the Parakeet path: whisperx.assign_word_speakers expects word-level data
    Parakeet doesn't expose, and markers only need a speaker per line anyway.
    `diarized` is a list of (start, end, speaker) intervals from pyannote.
    """
    for seg in segments:
        best_speaker, best_overlap = "", 0.0
        for d_start, d_end, speaker in diarized:
            overlap = min(seg["end"], d_end) - max(seg["start"], d_start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, speaker
        seg["speaker"] = best_speaker
    return segments


def _pick_torch_device() -> str:
    """GPU device for the torch stages (alignment + diarization).

    faster-whisper itself can't use Metal, but wav2vec2 alignment and pyannote
    diarization can — and diarization dominates runtime on long files. Override
    with AVID_TORCH_DEVICE if needed.
    """
    override = os.environ.get("AVID_TORCH_DEVICE")
    if override:
        return override
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def transcribe_and_diarize(
    audio_path: str,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    batch_size: int = 16,
    diarize_model: str = DEFAULT_DIARIZE_MODEL,
    torch_device: str | None = None,
    llm_correct: bool = True,
    force_language: str | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> TranscriptionResult:
    """Transcribe, align word timestamps, and diarize an audio file.

    `device` is for faster-whisper (CPU on Apple Silicon). `torch_device` runs
    the alignment + diarization on the GPU (MPS) when available. If MPS chokes on
    an unsupported op, diarization retries on CPU.

    `force_language` (e.g. "en") skips Whisper's language auto-detection, which it
    frequently gets wrong on music/ambiguous audio (e.g. detecting Welsh "cy" on
    English). If alignment has no model for the language, it's skipped and we fall
    back to Whisper's segment-level timestamps — diarization still labels each
    segment, only sub-segment word timing is lost (markers don't use it).

    If `hf_token` is missing, diarization is skipped and every segment gets an
    empty speaker label (still a usable plain transcript).

    When `llm_correct` is set and a local LLM is available, a final pass re-reads
    the diarized transcript and fixes obvious speaker-label mistakes.
    """
    import whisperx  # lazy: pulls in torch/pyannote

    _progress = progress_cb or (lambda _: None)

    token = hf_token or os.environ.get("HF_TOKEN")
    torch_device = torch_device or _pick_torch_device()
    use_parakeet = _asr_engine() == "parakeet" and _parakeet_available()

    audio = whisperx.load_audio(audio_path)

    _progress("Transcribing audio…")
    if use_parakeet:
        # Parakeet (English) supplies its own timestamps — skip whisperx alignment.
        from backend.mlx_runtime import run_on_mlx_thread

        parakeet_model = os.environ.get("AVID_PARAKEET_MODEL", DEFAULT_PARAKEET_MODEL)
        result = run_on_mlx_thread(_transcribe_parakeet, audio_path, parakeet_model)
        language = "en"
    else:
        if _mlx_available():
            from backend.mlx_runtime import run_on_mlx_thread

            base = run_on_mlx_thread(_transcribe_mlx, audio_path, model_size, force_language)
        else:
            model = whisperx.load_model(model_size, device, compute_type=compute_type)
            base = model.transcribe(audio, batch_size=batch_size, language=force_language)
        language = force_language or base["language"]

        _progress("Aligning word timestamps…")
        try:
            align_model, metadata = whisperx.load_align_model(language_code=language, device=torch_device)
            result = whisperx.align(
                base["segments"], align_model, metadata, audio, torch_device, return_char_alignments=False
            )
        except Exception:
            # No alignment model for this language (e.g. a mis-detected "cy"), or
            # alignment failed — keep Whisper's segment-level timestamps and carry on.
            result = {"segments": base["segments"]}

    if token:
        _progress("Identifying speakers…")
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        try:
            diarize_pipeline = _load_diarization_pipeline(token, torch_device, diarize_model)
            diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
        except Exception:
            if torch_device == "cpu":
                raise
            # MPS hit an unsupported op — fall back to CPU diarization.
            diarize_pipeline = _load_diarization_pipeline(token, "cpu", diarize_model)
            diarize_segments = diarize_pipeline(audio, **diarize_kwargs)
        if use_parakeet:
            diarized = [
                (float(r.start), float(r.end), str(r.speaker))
                for r in diarize_segments[["start", "end", "speaker"]].itertuples(index=False)
            ]
            _assign_speakers_by_overlap(result["segments"], diarized)
        else:
            result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = [
        TranscriptSegment(
            start=float(seg["start"]),
            end=float(seg["end"]),
            text=str(seg.get("text", "")).strip(),
            speaker=str(seg.get("speaker", "") or ""),
        )
        for seg in result["segments"]
        if seg.get("text", "").strip()
    ]

    corrected = 0
    if llm_correct and not os.environ.get("AVID_DISABLE_LLM_CORRECT") and any(
        seg.speaker for seg in segments
    ):
        _progress("Refining speaker labels…")
        try:
            from backend.llm import mlx_lm_available
            from backend.mlx_runtime import run_on_mlx_thread
            from backend.speaker_correction import correct_speaker_labels

            if mlx_lm_available():
                new_speakers, corrected = run_on_mlx_thread(
                    correct_speaker_labels,
                    [seg.text for seg in segments],
                    [seg.speaker for seg in segments],
                )
                for seg, speaker in zip(segments, new_speakers):
                    seg.speaker = speaker
        except Exception:
            corrected = 0  # correction is best-effort; never break the transcript

    return TranscriptionResult(
        language=language,
        segments=segments,
        speakers=_representative_samples(segments),  # samples reflect corrected labels
        corrected_speakers=corrected,
    )
