"""Pure-logic tests for the token-free sherpa-onnx diarizer.

These never import sherpa_onnx or load a model — only the path resolution, clustering
parameter mapping, and availability gating, all of which are plain functions.
"""

import pytest

from backend.diarize_sherpa import (
    _clustering_params,
    _model_dir,
    _resolve_models,
    available,
)


def test_clustering_params_default(monkeypatch):
    monkeypatch.delenv("AVID_DIARIZE_THRESHOLD", raising=False)
    # No speaker hints → threshold mode (num_clusters = -1), default 0.65.
    assert _clustering_params(None, None) == (-1, 0.65)


def test_clustering_params_uses_any_count_hint(monkeypatch):
    monkeypatch.delenv("AVID_DIARIZE_THRESHOLD", raising=False)
    # An exact count pins num_clusters (the over-clustering fix for long scenes).
    assert _clustering_params(3, 3) == (3, 0.65)
    # Max alone is honoured (previously ignored — the bug behind 16 clusters/2 speakers).
    assert _clustering_params(None, 2) == (2, 0.65)
    # Min alone is honoured too.
    assert _clustering_params(2, None) == (2, 0.65)
    # A range can't map to sherpa's exact-count API → prefer max.
    assert _clustering_params(2, 5) == (5, 0.65)


def test_clustering_params_threshold_override(monkeypatch):
    monkeypatch.setenv("AVID_DIARIZE_THRESHOLD", "0.5")
    assert _clustering_params(None, None) == (-1, 0.5)


def test_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AVID_DIARIZE_MODEL_DIR", str(tmp_path))
    assert _model_dir() == tmp_path


def test_resolve_models_with_overrides(monkeypatch, tmp_path):
    seg = tmp_path / "seg.onnx"
    emb = tmp_path / "emb.onnx"
    seg.write_bytes(b"x")
    emb.write_bytes(b"x")
    monkeypatch.setenv("AVID_DIARIZE_SEG_MODEL", str(seg))
    monkeypatch.setenv("AVID_DIARIZE_EMB_MODEL", str(emb))
    assert _resolve_models() == (str(seg), str(emb))


def test_resolve_models_missing_raises(monkeypatch, tmp_path):
    # Point at an empty dir so neither model exists.
    monkeypatch.setenv("AVID_DIARIZE_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("AVID_DIARIZE_SEG_MODEL", raising=False)
    monkeypatch.delenv("AVID_DIARIZE_EMB_MODEL", raising=False)
    with pytest.raises(FileNotFoundError):
        _resolve_models()


def test_available_false_when_models_missing(monkeypatch, tmp_path):
    # Even if sherpa_onnx is installed, missing models => not available.
    monkeypatch.setenv("AVID_DIARIZE_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("AVID_DIARIZE_SEG_MODEL", raising=False)
    monkeypatch.delenv("AVID_DIARIZE_EMB_MODEL", raising=False)
    assert available() is False
