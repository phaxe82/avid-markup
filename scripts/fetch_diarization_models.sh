#!/bin/bash
# Fetch the token-free, redistributable speaker-diarization models used by the default
# (sherpa-onnx) diarizer. Run once after cloning; the .app build bundles these.
#
#   - segmentation: pyannote segmentation-3.0 exported to ONNX (MIT, CNRS)
#   - embedding:    WeSpeaker ResNet34 trained on VoxCeleb (Apache-2.0)
#
# Both are redistributed *ungated* by the k2-fsa team (no HuggingFace token, no gated
# terms). They live in models/diarization/, overridable via AVID_DIARIZE_MODEL_DIR.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
DEST="models/diarization"
mkdir -p "$DEST"

SEG_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
EMB_DEST="$DEST/wespeaker_en_voxceleb_resnet34_LM.onnx"

if [ ! -f "$DEST/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  echo "Downloading segmentation model…"
  curl -fSL -o "$DEST/seg.tar.bz2" "$SEG_URL"
  tar xjf "$DEST/seg.tar.bz2" -C "$DEST"
  rm -f "$DEST/seg.tar.bz2"
else
  echo "Segmentation model already present — skipping."
fi

if [ ! -f "$EMB_DEST" ]; then
  echo "Downloading speaker-embedding model…"
  curl -fSL -o "$EMB_DEST" "$EMB_URL"
else
  echo "Embedding model already present — skipping."
fi

echo "Done. Models in $DEST:"
du -sh "$DEST/sherpa-onnx-pyannote-segmentation-3-0" "$EMB_DEST"
