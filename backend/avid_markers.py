"""Build an Avid Media Composer marker (locator) import file.

Format reverse-engineered byte-for-byte from a real Avid export (`od -c`):
8 tab-separated fields per line, LF line endings, plain text, no header row:

    Name <TAB> Timecode <TAB> Track <TAB> Color <TAB> Comment <TAB> Duration <TAB> <TAB> Color

The Color appears twice (Avid's COLOR_EXTENDED and COLOR properties); field 7
between Duration and the trailing Color is empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from backend.timecode_utils import DEFAULT_FPS, offset_timecode, seconds_to_frames

# Avid's classic marker colours, capitalised as Avid writes them.
AVID_COLORS: list[str] = [
    "Red",
    "Green",
    "Blue",
    "Cyan",
    "Magenta",
    "Yellow",
    "White",
    "Black",
]


@dataclass
class Segment:
    """One diarised transcript segment."""

    start: float  # seconds from audio start
    end: float
    text: str
    speaker: str  # diariser label, e.g. "SPEAKER_00", or "" if none
    include: bool = True


@dataclass
class MarkerSettings:
    start_tc: str
    fps: int = DEFAULT_FPS
    author: str = "Transcriber"
    track: str = "V1"
    label_separator: str = " - "
    # Map diariser label -> human label, e.g. {"SPEAKER_00": "Sam"}.
    speaker_labels: dict[str, str] = field(default_factory=dict)
    color_mode: Literal["single", "per_speaker"] = "single"
    single_color: str = "Yellow"
    # Optional explicit diariser-label -> colour; otherwise auto-cycled.
    speaker_colors: dict[str, str] = field(default_factory=dict)
    duration_mode: Literal["point", "span"] = "point"


def sanitize_comment(text: str) -> str:
    """Collapse whitespace and strip tabs/newlines so a comment stays one line."""
    return " ".join(text.split())


def format_marker_line(
    name: str,
    timecode: str,
    track: str,
    color: str,
    comment: str,
    duration: int,
) -> str:
    """Build a single tab-delimited marker line (no trailing newline)."""
    return "\t".join(
        [name, timecode, track, color, sanitize_comment(comment), str(duration), "", color]
    )


def _assign_speaker_colors(
    segments: list[Segment], settings: MarkerSettings
) -> dict[str, str]:
    """Return a diariser-label -> colour map, cycling AVID_COLORS by first appearance."""
    colors: dict[str, str] = dict(settings.speaker_colors)
    next_index = 0
    for seg in segments:
        if seg.speaker and seg.speaker not in colors:
            colors[seg.speaker] = AVID_COLORS[next_index % len(AVID_COLORS)]
            next_index += 1
    return colors


def _comment_for(seg: Segment, settings: MarkerSettings) -> str:
    label = settings.speaker_labels.get(seg.speaker, seg.speaker)
    if label:
        return f"{label}{settings.label_separator}{seg.text}"
    return seg.text


def _color_for(seg: Segment, settings: MarkerSettings, speaker_colors: dict[str, str]) -> str:
    if settings.color_mode == "per_speaker" and seg.speaker:
        return speaker_colors.get(seg.speaker, settings.single_color)
    return settings.single_color


def _duration_for(seg: Segment, settings: MarkerSettings) -> int:
    if settings.duration_mode == "span":
        return max(1, seconds_to_frames(seg.end - seg.start, settings.fps))
    return 1


def build_markers(segments: list[Segment], settings: MarkerSettings) -> str:
    """Build the full marker file text (LF line endings, trailing newline)."""
    speaker_colors = _assign_speaker_colors(segments, settings)
    lines: list[str] = []
    for seg in segments:
        if not seg.include:
            continue
        timecode = offset_timecode(settings.start_tc, seg.start, settings.fps)
        lines.append(
            format_marker_line(
                name=settings.author,
                timecode=timecode,
                track=settings.track,
                color=_color_for(seg, settings, speaker_colors),
                comment=_comment_for(seg, settings),
                duration=_duration_for(seg, settings),
            )
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
