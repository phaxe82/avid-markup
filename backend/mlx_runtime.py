"""One dedicated thread for ALL MLX work — mlx-whisper transcription and the
mlx-lm refinement passes (speaker correction + fluff triage).

MLX binds its GPU stream to the thread that first touches it, and a model/array
created on one thread cannot be evaluated on another ("There is no Stream(gpu, 0)
in current thread"). The server otherwise hits MLX from several threads — the
transcription job's daemon thread and FastAPI's threadpool workers — so every MLX
operation is funnelled through this single worker thread to keep the stream
consistent. max_workers=1 also serialises GPU work, which avoids contention.

Rules:
- Wrap the whole top-level call (`_transcribe_mlx`, `correct_speaker_labels`,
  `triage_segments`) — never an inner helper — so its model load and every op run
  on this thread.
- Never call this from the MLX thread itself; the single worker would deadlock.
- Availability checks must NOT import mlx/mlx_lm/mlx_whisper (use
  `importlib.util.find_spec`), or they would initialise MLX off this thread.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_T = TypeVar("_T")


def run_on_mlx_thread(fn: Callable[..., _T], /, *args, **kwargs) -> _T:
    """Run `fn` on the single dedicated MLX thread and return its result."""
    return _MLX_EXECUTOR.submit(fn, *args, **kwargs).result()
