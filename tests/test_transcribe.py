from backend.transcribe import (
    TranscriptSegment,
    _asr_engine,
    _assign_speakers_by_overlap,
    _diarizer,
    _parakeet_available,
    _pick_torch_device,
    _representative_samples,
)


def test_asr_engine_default_and_override(monkeypatch):
    monkeypatch.delenv("AVID_ASR_ENGINE", raising=False)
    assert _asr_engine() == "whisper"
    monkeypatch.setenv("AVID_ASR_ENGINE", "Parakeet")  # case-insensitive
    assert _asr_engine() == "parakeet"


def test_diarizer_default_is_sherpa_and_override(monkeypatch):
    monkeypatch.delenv("AVID_DIARIZER", raising=False)
    assert _diarizer() == "sherpa"  # token-free default
    monkeypatch.setenv("AVID_DIARIZER", "PyAnnote")  # case-insensitive
    assert _diarizer() == "pyannote"


def test_parakeet_disabled_by_env(monkeypatch):
    monkeypatch.setenv("AVID_DISABLE_PARAKEET", "1")
    assert _parakeet_available() is False


def test_assign_speakers_by_overlap():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "a"},    # mostly SPEAKER_00
        {"start": 5.0, "end": 7.0, "text": "b"},    # SPEAKER_01
        {"start": 20.0, "end": 21.0, "text": "c"},  # no diarized overlap
    ]
    diarized = [
        (0.0, 1.8, "SPEAKER_00"),
        (1.8, 2.5, "SPEAKER_01"),
        (4.5, 7.5, "SPEAKER_01"),
    ]
    _assign_speakers_by_overlap(segments, diarized)
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[1]["speaker"] == "SPEAKER_01"
    assert segments[2]["speaker"] == ""


def test_pick_torch_device_env_override(monkeypatch):
    monkeypatch.setenv("AVID_TORCH_DEVICE", "cpu")
    assert _pick_torch_device() == "cpu"
    monkeypatch.setenv("AVID_TORCH_DEVICE", "mps")
    assert _pick_torch_device() == "mps"


def test_representative_picks_longest_per_speaker():
    segs = [
        TranscriptSegment(0.0, 1.0, "short A", "SPEAKER_00"),
        TranscriptSegment(1.0, 5.0, "long A", "SPEAKER_00"),
        TranscriptSegment(5.0, 6.0, "B", "SPEAKER_01"),
    ]
    samples = _representative_samples(segs)
    by_speaker = {s.speaker: s for s in samples}
    assert by_speaker["SPEAKER_00"].text == "long A"
    assert by_speaker["SPEAKER_00"].start == 1.0
    assert set(by_speaker) == {"SPEAKER_00", "SPEAKER_01"}


def test_representative_ignores_empty_speaker():
    segs = [TranscriptSegment(0.0, 1.0, "no speaker", "")]
    assert _representative_samples(segs) == []
