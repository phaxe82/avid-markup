from backend.avid_markers import (
    AVID_COLORS,
    MarkerSettings,
    Segment,
    build_markers,
    format_marker_line,
    sanitize_comment,
)

# Byte-exact first line from the real Avid export (verified via `od -c`):
# Tom \t 13:23:06:03 \t V1 \t Yellow \t START \t 1 \t \t Yellow \n
REFERENCE_FIRST_LINE = "Tom\t13:23:06:03\tV1\tYellow\tSTART\t1\t\tYellow"


def test_format_matches_reference_bytes():
    line = format_marker_line("Tom", "13:23:06:03", "V1", "Yellow", "START", 1)
    assert line == REFERENCE_FIRST_LINE
    # 8 fields, field 7 (index 6) empty, colour repeated in fields 4 and 8.
    fields = line.split("\t")
    assert len(fields) == 8
    assert fields[6] == ""
    assert fields[3] == fields[7] == "Yellow"


def test_sanitize_strips_tabs_and_newlines():
    assert sanitize_comment("a\tb\nc  d") == "a b c d"


def test_build_markers_lf_and_trailing_newline():
    segs = [Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00")]
    settings = MarkerSettings(start_tc="10:00:00:00", speaker_labels={"SPEAKER_00": "Sam"})
    out = build_markers(segs, settings)
    assert "\r" not in out
    assert out.endswith("\n")
    assert out == "Transcriber\t10:00:00:00\tV1\tYellow\tSam - hello\t1\t\tYellow\n"


def test_speaker_prefix_and_offset():
    segs = [Segment(start=4.48, end=6.0, text="Where were you?", speaker="SPEAKER_01")]
    settings = MarkerSettings(
        start_tc="10:00:00:00",
        author="Tom",
        speaker_labels={"SPEAKER_01": "DETECTIVE"},
    )
    out = build_markers(segs, settings).strip()
    assert out == "Tom\t10:00:04:12\tV1\tYellow\tDETECTIVE - Where were you?\t1\t\tYellow"


def test_per_speaker_color_cycles_by_first_appearance():
    segs = [
        Segment(start=0.0, end=1.0, text="a", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.0, text="b", speaker="SPEAKER_01"),
        Segment(start=2.0, end=3.0, text="c", speaker="SPEAKER_00"),
    ]
    settings = MarkerSettings(start_tc="10:00:00:00", color_mode="per_speaker")
    lines = build_markers(segs, settings).strip().split("\n")
    colors = [ln.split("\t")[3] for ln in lines]
    assert colors == [AVID_COLORS[0], AVID_COLORS[1], AVID_COLORS[0]]


def test_excluded_segments_dropped():
    segs = [
        Segment(start=0.0, end=1.0, text="keep", speaker="SPEAKER_00"),
        Segment(start=1.0, end=2.0, text="drop", speaker="SPEAKER_00", include=False),
    ]
    settings = MarkerSettings(start_tc="10:00:00:00")
    lines = build_markers(segs, settings).strip().split("\n")
    assert len(lines) == 1
    assert "keep" in lines[0]


def test_span_duration_in_frames():
    segs = [Segment(start=0.0, end=2.0, text="x", speaker="")]
    settings = MarkerSettings(start_tc="10:00:00:00", duration_mode="span")
    out = build_markers(segs, settings).strip()
    # 2.0s * 25 = 50 frames; no speaker -> no prefix
    assert out == "Transcriber\t10:00:00:00\tV1\tYellow\tx\t50\t\tYellow"
