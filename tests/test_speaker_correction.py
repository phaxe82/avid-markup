from backend.speaker_correction import (
    Correction,
    apply_corrections,
    build_prompt_messages,
    mlx_lm_available,
    parse_corrections,
)

VALID = {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


def test_parse_corrections_basic():
    raw = '[{"line": 2, "speaker": "SPEAKER_01"}, {"line": 5, "speaker": "SPEAKER_00"}]'
    got = parse_corrections(raw, VALID, min_line=0, max_line=10)
    assert got == [Correction(2, "SPEAKER_01"), Correction(5, "SPEAKER_00")]


def test_parse_corrections_tolerates_prose_and_fences():
    raw = "Sure! Here are the fixes:\n```json\n[{\"line\": 1, \"speaker\": \"SPEAKER_02\"}]\n```\nDone."
    assert parse_corrections(raw, VALID, 0, 10) == [Correction(1, "SPEAKER_02")]


def test_parse_corrections_filters_out_of_range_and_unknown_speaker():
    raw = (
        '[{"line": 99, "speaker": "SPEAKER_00"},'  # out of editable range
        ' {"line": 3, "speaker": "SPEAKER_09"},'  # speaker not present
        ' {"line": 4, "speaker": "SPEAKER_01"}]'  # the only keeper
    )
    assert parse_corrections(raw, VALID, min_line=2, max_line=6) == [Correction(4, "SPEAKER_01")]


def test_parse_corrections_coerces_string_line_and_dedupes():
    raw = '[{"line": "3", "speaker": "SPEAKER_00"}, {"line": 3, "speaker": "SPEAKER_01"}]'
    # first valid entry for a line wins; duplicate line ignored
    assert parse_corrections(raw, VALID, 0, 10) == [Correction(3, "SPEAKER_00")]


def test_parse_corrections_malformed_returns_empty():
    assert parse_corrections("no json here", VALID, 0, 10) == []
    assert parse_corrections("[not valid json", VALID, 0, 10) == []


def test_apply_corrections_changes_and_counts():
    speakers = ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    updated, changed = apply_corrections(
        speakers, [Correction(1, "SPEAKER_01"), Correction(2, "SPEAKER_00")]
    )
    assert updated == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    assert changed == 1  # line 2 was already SPEAKER_00, only line 1 changed
    assert speakers[1] == "SPEAKER_00"  # original list untouched


def test_apply_corrections_ignores_out_of_bounds():
    updated, changed = apply_corrections(["SPEAKER_00"], [Correction(5, "SPEAKER_01")])
    assert updated == ["SPEAKER_00"]
    assert changed == 0


def test_build_prompt_messages_numbers_globally_and_marks_editable_range():
    texts = ["a", "b", "c", "d", "e"]
    speakers = ["SPEAKER_00"] * 5
    messages = build_prompt_messages(
        texts, speakers, ["SPEAKER_00", "SPEAKER_01"],
        win_start=1, win_end=5, core_start=2, core_end=5,
    )
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "1: (SPEAKER_00) b" in user  # context line rendered with global index
    assert "0: (SPEAKER_00) a" not in user  # before the window, not shown
    assert "lines 2 to 4 inclusive" in user  # editable range advertised


def test_mlx_lm_available_respects_master_disable_env(monkeypatch):
    monkeypatch.setenv("AVID_DISABLE_LLM", "1")
    assert mlx_lm_available() is False
