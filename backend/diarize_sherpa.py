"""Token-free, fully offline speaker diarization via sherpa-onnx.

This is the **default** diarizer for the public build: it needs no HuggingFace token
and no gated-model terms. The models — an MIT-licensed pyannote segmentation model and
an Apache-licensed 3D-Speaker embedding model, both exported to ONNX and redistributed
*ungated* by the k2-fsa team — are small (~34 MB total), so they ship inside the app and
run on `onnxruntime` (already a whisperx dependency) on CPU. No MLX thread involved.

It returns `(start, end, speaker)` intervals shaped exactly like the pyannote path's
output, so `backend.transcribe._assign_speakers_by_overlap` tags transcript segments
unchanged. Speaker labels are formatted `SPEAKER_00`, `SPEAKER_01`, … to match pyannote
so the correction / grouping / UI layers treat both diarizers identically.

`sherpa_onnx` and `whisperx` are imported lazily so the web app and unit tests run
without the heavy ML stack present.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from pathlib import Path

# Default on-disk layout (also what the .app bundles). Overridable for the frozen app
# or A/B testing via the AVID_DIARIZE_* env vars below.
_DEFAULT_MODEL_SUBDIR = "models/diarization"
_SEG_REL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
# WeSpeaker ResNet34 (VoxCeleb, Apache-2.0) — chosen by A/B benchmark over CAM++ and
# eres2net: it matched pyannote's speaker count (4/4) on two real scenes with ~67-73%
# frame agreement, where CAM++ over-clustered impurely and eres2net fragmented into 30+.
_EMB_REL = "wespeaker_en_voxceleb_resnet34_LM.onnx"

# sherpa's diarizer ingests 16 kHz mono float32 — the same format whisperx.load_audio
# already produces, so no resampling is needed on the normal pipeline path.
_TARGET_SAMPLE_RATE = 16000


def _model_dir() -> Path:
    """Directory holding the diarization models (env override → bundled/repo default)."""
    override = os.environ.get("AVID_DIARIZE_MODEL_DIR")
    if override:
        return Path(override)
    from backend.paths import resource_dir

    return resource_dir() / _DEFAULT_MODEL_SUBDIR


def _resolve_models() -> tuple[str, str]:
    """Return (segmentation_model, embedding_model) paths, honouring per-model overrides.

    Raises FileNotFoundError with an actionable message if a model is missing."""
    base = _model_dir()
    seg = Path(os.environ.get("AVID_DIARIZE_SEG_MODEL") or base / _SEG_REL)
    emb = Path(os.environ.get("AVID_DIARIZE_EMB_MODEL") or base / _EMB_REL)
    missing = [str(p) for p in (seg, emb) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "sherpa-onnx diarization model(s) not found: "
            + ", ".join(missing)
            + ". Fetch them with scripts/fetch_diarization_models.sh or set "
            "AVID_DIARIZE_MODEL_DIR."
        )
    return str(seg), str(emb)


def available() -> bool:
    """True if sherpa-onnx is installed *and* the models are present on disk.

    Uses find_spec (no import) to stay cheap and side-effect-free, mirroring the other
    availability checks in backend.transcribe."""
    if importlib.util.find_spec("sherpa_onnx") is None:
        return False
    try:
        _resolve_models()
    except FileNotFoundError:
        return False
    return True


def _clustering_params(min_speakers: int | None, max_speakers: int | None) -> tuple[int, float]:
    """Map the UI's min/max speaker hints onto sherpa's clustering.

    sherpa supports either an *exact* cluster count or a distance threshold — not a
    range. **Any** speaker-count hint is therefore used as the exact `num_clusters`
    (max preferred, else min). This is the reliable fix for long scenes: with no count,
    pure threshold clustering over-splits badly — the same two voices drift over a
    30–60 min scene and keep spawning clusters (e.g. ~12 clusters for a 2-speaker hour),
    so passing the cast size collapses it to exactly that many. With no hint at all we
    fall back to the distance `threshold` (tunable via AVID_DIARIZE_THRESHOLD): safe
    (over-clustering is recoverable via the label-then-merge UX) but imprecise on long
    scenes — which is why the UI nudges the user to enter a speaker count.
    """
    threshold = float(os.environ.get("AVID_DIARIZE_THRESHOLD", "0.65"))
    target = max_speakers or min_speakers
    if target:
        return int(target), threshold
    return -1, threshold


def _build_config(seg: str, emb: str, num_clusters: int, threshold: float, num_threads: int):
    """Construct the OfflineSpeakerDiarizationConfig (sherpa_onnx imported lazily)."""
    import sherpa_onnx as so

    return so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(model=seg),
            num_threads=num_threads,
            provider="cpu",
        ),
        embedding=so.SpeakerEmbeddingExtractorConfig(
            model=emb, num_threads=num_threads, provider="cpu"
        ),
        clustering=so.FastClusteringConfig(num_clusters=num_clusters, threshold=threshold),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )


def diarize_samples(
    samples,
    sample_rate: int,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[tuple[float, float, str]]:
    """Diarize a 16 kHz mono float32 sample array → sorted (start, end, speaker) intervals.

    `samples` is a numpy float32 array (e.g. from whisperx.load_audio). `progress_cb`,
    if given, receives a 0.0–1.0 fraction as chunks are processed.
    """
    import numpy as np
    import sherpa_onnx as so

    if sample_rate != _TARGET_SAMPLE_RATE:
        raise ValueError(
            f"sherpa diarization needs {_TARGET_SAMPLE_RATE} Hz audio, got {sample_rate}; "
            "load via whisperx.load_audio (which returns 16 kHz)."
        )

    seg, emb = _resolve_models()
    num_clusters, threshold = _clustering_params(min_speakers, max_speakers)
    num_threads = int(os.environ.get("AVID_DIARIZE_THREADS", "4"))

    config = _build_config(seg, emb, num_clusters, threshold, num_threads)
    if not config.validate():
        raise RuntimeError("sherpa-onnx diarization config failed to validate (check model paths).")
    sd = so.OfflineSpeakerDiarization(config)

    audio = np.ascontiguousarray(samples, dtype=np.float32)

    if progress_cb is not None:
        def _cb(*args: int) -> int:  # sherpa passes (processed, total[, arg])
            processed, total = (args + (0, 0))[:2]
            progress_cb(processed / total if total else 0.0)
            return 0

        result = sd.process(audio, callback=_cb)
    else:
        result = sd.process(audio)

    return [
        (float(s.start), float(s.end), f"SPEAKER_{int(s.speaker):02d}")
        for s in result.sort_by_start_time()
    ]


def diarize(
    audio_path: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> list[tuple[float, float, str]]:
    """Convenience: decode `audio_path` to 16 kHz mono via whisperx, then diarize.

    On the normal pipeline path, backend.transcribe already holds the decoded audio and
    calls `diarize_samples` directly — this loader is for standalone use (smoke tests,
    the A/B benchmark)."""
    import whisperx

    samples = whisperx.load_audio(audio_path)
    return diarize_samples(samples, _TARGET_SAMPLE_RATE, min_speakers, max_speakers, progress_cb)
