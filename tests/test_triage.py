from backend.triage import (
    build_triage_messages,
    find_duplicate_indices,
    parse_drops,
    triage_segments,
    window_drops_plausible,
)


def test_parse_drops_basic():
    assert parse_drops("[1, 4, 9]", min_line=0, max_line=10) == [1, 4, 9]


def test_parse_drops_tolerates_objects_and_strings():
    raw = '[{"line": 2}, {"index": 5}, "7"]'
    assert parse_drops(raw, 0, 10) == [2, 5, 7]


def test_parse_drops_filters_range_and_dedupes():
    raw = "[3, 3, 99, -1, 4]"
    assert parse_drops(raw, min_line=2, max_line=6) == [3, 4]


def test_parse_drops_rejects_booleans():
    # JSON true would otherwise sneak through as int(1)
    assert parse_drops("[true, 2]", 0, 10) == [2]


def test_parse_drops_malformed_returns_empty():
    assert parse_drops("nothing here", 0, 10) == []
    assert parse_drops("[oops", 0, 10) == []


def test_build_triage_messages_includes_level_guidance_and_global_indices():
    texts = ["hello", "um yeah", "the body was found at noon", "right", "who did it"]
    speakers = ["A", "A", "B", "A", "B"]
    messages = build_triage_messages(
        texts, speakers, win_start=1, win_end=5, core_start=2, core_end=5,
        level="aggressive", guidance="keep anything about the investigation",
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "Keep ONLY lines" in system  # aggressive policy text present
    assert "keep anything about the investigation" in system  # guidance injected
    assert "2: (B) the body was found at noon" in user  # global index + speaker
    assert "0:" not in user  # before window, not shown
    assert "lines 2 to 4 inclusive" in user


def test_build_triage_messages_unknown_level_falls_back_to_balanced():
    msg = build_triage_messages(["x"], ["A"], 0, 1, 0, 1, level="bogus", guidance="")
    assert "Drop filler and backchannel" in msg[0]["content"]


def test_window_drops_plausible_rejects_dropping_everything():
    assert window_drops_plausible([0, 1], core_count=5) is True
    assert window_drops_plausible([0, 1, 2, 3, 4], core_count=5) is False  # all dropped
    assert window_drops_plausible([], core_count=0) is True  # empty window is fine


def test_find_duplicate_indices_drops_repeats_keeps_first():
    texts = ["I think so.", "I think so.", "Let's go.", "I think so."]
    # index 1 repeats 0; index 3 repeats within lookback of 1 — both dropped, first kept
    assert find_duplicate_indices(texts) == [1, 3]


def test_find_duplicate_indices_normalizes_case_and_punctuation():
    assert find_duplicate_indices(["The house was on fire!", "the house was on fire"]) == [1]


def test_find_duplicate_indices_drops_consecutive_single_word_loops():
    # five "Okay." in a row is a Whisper loop — keep the first, drop the rest
    assert find_duplicate_indices(["Okay.", "Okay.", "Okay.", "Done"]) == [1, 2]


def test_find_duplicate_indices_keeps_nonadjacent_single_word_replies():
    # two people each saying "yeah" a line apart are distinct — keep both
    assert find_duplicate_indices(["Yeah", "No", "Yeah"]) == []


def test_find_duplicate_indices_respects_lookback_window():
    texts = ["good morning", "a", "b", "c", "good morning"]
    # the repeat is 4 lines later, beyond the default lookback of 3 kept lines -> kept
    assert find_duplicate_indices(texts) == []


def test_triage_dedup_level_needs_no_model():
    # level="dedup" returns before any model load, so this runs without the ML stack
    drops = triage_segments(["Don't run.", "Don't run.", "Keep going."], ["A", "A", "A"], level="dedup")
    assert drops == [1]
