"""Audio decoding via PyAV (LGPL FFmpeg libraries bundled in the `av` wheel).

Replaces the ffmpeg-CLI subprocess that whisperx / mlx-whisper otherwise shell out to,
so the frozen .app needs **no external ffmpeg binary and no GPL ffmpeg build** — PyAV's
wheels bundle LGPL FFmpeg *libraries* and decode in-process. The pipeline decodes once
here and passes the resulting 16 kHz mono float32 array to every stage (transcription,
alignment, diarization), none of which then touch the CLI.

`av` and `numpy` are imported lazily so the web app and unit tests run without them.
"""

from __future__ import annotations

SAMPLE_RATE = 16000


def load_audio(path: str, sr: int = SAMPLE_RATE):
    """Decode an audio file to a mono float32 numpy array at `sr` Hz.

    Mirrors whisperx.load_audio's output (16 kHz mono float32) byte-for-byte in length,
    but uses PyAV instead of an ffmpeg subprocess — so no external ffmpeg binary is
    required, and the bundled FFmpeg is LGPL (PyAV) rather than GPL.
    """
    import av
    import numpy as np

    with av.open(path) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream found in {path!r}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks: list = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):  # flush the resampler's tail
            chunks.append(resampled.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)
