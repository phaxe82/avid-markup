import threading
from concurrent.futures import ThreadPoolExecutor

from backend.llm import extract_json_array, iter_windows
from backend.mlx_runtime import run_on_mlx_thread


def test_run_on_mlx_thread_returns_result_and_passes_args():
    assert run_on_mlx_thread(lambda a, b: a + b, 2, 3) == 5
    assert run_on_mlx_thread(sorted, [3, 1, 2], reverse=True) == [3, 2, 1]


def test_run_on_mlx_thread_uses_one_dedicated_thread_across_callers():
    # All MLX work must land on a single thread regardless of which thread calls,
    # since MLX can't share a model/stream across threads.
    def which_thread() -> str:
        return threading.current_thread().name

    # Call from two different caller threads (mimicking the transcription daemon
    # thread and a FastAPI threadpool worker).
    with ThreadPoolExecutor(max_workers=2) as callers:
        names = list(callers.map(lambda _: run_on_mlx_thread(which_thread), range(4)))

    assert all(n.startswith("mlx") for n in names)
    assert len(set(names)) == 1  # the same single executor thread every time
    assert names[0] != threading.current_thread().name  # not the caller's thread


def test_shared_helpers_still_importable():
    assert extract_json_array("x [1,2] y") == [1, 2]
    assert list(iter_windows(5, window=2, context=1)) == [(0, 0, 2), (1, 2, 4), (3, 4, 5)]
