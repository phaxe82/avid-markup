# Third-party notices

Avid Markup is distributed as a frozen macOS app that bundles the following third-party
software and models. Each is used under its own licence; copies of the licence texts are
included in the app bundle under `Contents/Resources/licenses/`.

## Runtime libraries

| Component | Licence | Notes |
|---|---|---|
| PyTorch (`torch`) | BSD-3-Clause | alignment + diarization runtime |
| MLX, `mlx-whisper`, `mlx-lm` | MIT / Apache-2.0 | Apple-Silicon GPU transcription + correction LLM |
| WhisperX (`whisperx`) | BSD-2-Clause | transcription/alignment orchestration |
| `pyannote.audio` | MIT | optional diarizer (library only; gated weights **not** bundled) |
| `sherpa-onnx` | Apache-2.0 | default speaker diarizer |
| ONNX Runtime (`onnxruntime`) | MIT | runs the diarization models |
| **PyAV (`av`)** | BSD-3-Clause | audio decoding |
| ↳ **FFmpeg libraries** (bundled in the PyAV wheel) | **LGPL v2.1+** | see LGPL note below |
| FastAPI / Starlette / Uvicorn | MIT / BSD-3-Clause | local web server |
| `transformers`, `huggingface-hub` | Apache-2.0 | model loading |
| `timecode` | MIT | timecode arithmetic |

### FFmpeg / LGPL note

Audio is decoded **in-process** by PyAV, which links the **LGPL v2.1** FFmpeg libraries
bundled in its wheel. No GPL-configured FFmpeg binary is shipped, and the app does **not**
invoke an external `ffmpeg` CLI on its default (Whisper) path. To comply with the LGPL:
the FFmpeg libraries are dynamically loaded (replaceable), the LGPL licence text is
included in the bundle, and the corresponding FFmpeg source for the bundled version is
available on request / from the PyAV release it ships with.

> The optional, off-by-default **Parakeet** ASR engine (`AVID_ASR_ENGINE=parakeet`) loads
> audio via the external `ffmpeg` CLI and therefore is **not** part of the bundled app's
> functionality unless the user has ffmpeg installed separately.

## Bundled models (token-free, redistributable)

| Model | Licence | Source |
|---|---|---|
| pyannote `segmentation-3.0` (ONNX export) | MIT (© 2022 CNRS) | redistributed by k2-fsa / sherpa-onnx |
| WeSpeaker ResNet34 (VoxCeleb) embedding | Apache-2.0 | redistributed by k2-fsa / sherpa-onnx |
| OpenAI Whisper weights (`mlx-community/whisper-*`) | MIT | downloaded on first run (not bundled by default) |
| Qwen2.5-7B-Instruct (4-bit, correction LLM) | Apache-2.0 | downloaded on first run only if AI correction is enabled |

The gated `pyannote/speaker-diarization-community-1` model is **not** bundled or
redistributed; it is used only on the opt-in `AVID_DIARIZER=pyannote` path, downloaded by
the user with their own HuggingFace token.

---

**The application's own licence is recorded in `LICENSE`.** (Choose one before public
release — e.g. MIT for a permissive open-source tool. This file documents only the
third-party components Avid Markup redistributes.)
