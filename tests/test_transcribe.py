from backend.transcribe import (
    TranscriptSegment,
    _pick_torch_device,
    _representative_samples,
)


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
