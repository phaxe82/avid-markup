"""Offset transcript timestamps by a scene's start timecode.

The transcript gives each segment a start time in seconds relative to the audio
file. Avid needs an absolute timecode (HH:MM:SS:FF). We add the scene's start
timecode to each offset at the project frame rate (25 fps PAL, non-drop).
"""

from __future__ import annotations

from timecode import Timecode

DEFAULT_FPS = 25


def seconds_to_frames(seconds: float, fps: int = DEFAULT_FPS) -> int:
    """Convert a duration in seconds to whole frames at the given rate."""
    if seconds < 0:
        raise ValueError(f"seconds must be non-negative, got {seconds}")
    return round(seconds * fps)


def validate_timecode(start_tc: str, fps: int = DEFAULT_FPS) -> str:
    """Return start_tc normalised to HH:MM:SS:FF, raising ValueError if invalid."""
    try:
        return str(Timecode(str(fps), start_tc))
    except Exception as exc:  # timecode raises bare ValueError/AttributeError
        raise ValueError(f"invalid timecode {start_tc!r} at {fps} fps: {exc}") from exc


def offset_timecode(start_tc: str, offset_seconds: float, fps: int = DEFAULT_FPS) -> str:
    """Return start_tc advanced by offset_seconds, as HH:MM:SS:FF.

    25 fps non-drop, so no drop-frame compensation is needed.
    """
    base = Timecode(str(fps), start_tc)
    result = base + seconds_to_frames(offset_seconds, fps)
    return str(result)
