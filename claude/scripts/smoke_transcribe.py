"""One-off smoke test: run the full pipeline on the synthetic test clip."""

import sys

from backend.avid_markers import MarkerSettings, Segment, build_markers
from backend.transcribe import transcribe_and_diarize

audio = sys.argv[1] if len(sys.argv) > 1 else "/tmp/scene_test.mp3"
model = sys.argv[2] if len(sys.argv) > 2 else "tiny"

result = transcribe_and_diarize(audio, model_size=model)

print("LANGUAGE:", result.language)
print("SPEAKERS:", [s.speaker for s in result.speakers])
print("SEGMENTS:")
for s in result.segments:
    print(f"  [{s.start:6.2f}-{s.end:6.2f}] {s.speaker or '(none)':12} {s.text}")

settings = MarkerSettings(start_tc="10:00:00:00", author="Tom", color_mode="per_speaker")
segs = [Segment(start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in result.segments]
print("\n--- MARKER FILE ---")
print(build_markers(segs, settings), end="")
