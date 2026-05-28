import pytest

from backend.timecode_utils import (
    offset_timecode,
    seconds_to_frames,
    validate_timecode,
)


def test_seconds_to_frames_basic():
    assert seconds_to_frames(0, 25) == 0
    assert seconds_to_frames(1, 25) == 25
    assert seconds_to_frames(4.48, 25) == 112
    assert seconds_to_frames(0.5, 25) == 12  # 12.5 -> banker's rounding to 12


def test_seconds_to_frames_negative_raises():
    with pytest.raises(ValueError):
        seconds_to_frames(-1)


def test_offset_zero_returns_start():
    assert offset_timecode("13:23:06:03", 0.0) == "13:23:06:03"


def test_offset_one_second():
    assert offset_timecode("13:23:06:03", 1.0) == "13:23:07:03"


def test_offset_known_value():
    # 4.48s * 25 = 112 frames
    assert offset_timecode("10:00:00:00", 4.48) == "10:00:04:12"


def test_offset_wraps_seconds_and_minutes():
    # 13:23:59:24 + 1 frame -> 13:24:00:00
    assert offset_timecode("13:23:59:24", 1 / 25) == "13:24:00:00"
    # cross a minute: 13:23:59:00 + 1s + 1 frame
    assert offset_timecode("13:23:59:00", 1.04) == "13:24:00:01"


def test_offset_wraps_hours():
    assert offset_timecode("13:59:59:24", 1 / 25) == "14:00:00:00"


def test_validate_timecode_ok():
    assert validate_timecode("13:23:06:03") == "13:23:06:03"


def test_validate_timecode_bad():
    with pytest.raises(ValueError):
        validate_timecode("not-a-timecode")
