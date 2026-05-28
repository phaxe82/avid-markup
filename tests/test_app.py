from fastapi.testclient import TestClient

import backend.app as app_module
from backend.app import _write_env_token, app

client = TestClient(app)


def test_config_endpoint():
    r = client.get("/api/config")
    assert r.status_code == 200
    assert r.json()["default_fps"] == 25


def test_export_produces_marker_file():
    payload = {
        "start_tc": "10:00:00:00",
        "author": "Tom",
        "speaker_labels": {"SPEAKER_00": "Sam", "SPEAKER_01": "Deb"},
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Come on in", "speaker": "SPEAKER_00"},
            {"start": 4.48, "end": 6.0, "text": "It was gorgeous", "speaker": "SPEAKER_01"},
        ],
    }
    r = client.post("/api/export", json=payload)
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="markers_markers.txt"'
    lines = r.text.strip().split("\n")
    assert lines[0] == "Tom\t10:00:00:00\tV1\tYellow\tSam - Come on in\t1\t\tYellow"
    assert lines[1] == "Tom\t10:00:04:12\tV1\tYellow\tDeb - It was gorgeous\t1\t\tYellow"


def test_export_prepare_and_download_filename_in_url():
    payload = {
        "start_tc": "10:00:00:00",
        "author": "Tom",
        "filename": "TGB2_040_EP09",
        "speaker_labels": {"SPEAKER_00": "Sam"},
        "segments": [{"start": 0.0, "end": 2.0, "text": "Come on in", "speaker": "SPEAKER_00"}],
    }
    prep = client.post("/api/export_prepare", json=payload)
    assert prep.status_code == 200
    url = prep.json()["download_url"]
    # URL path ends in the filename so the browser can't mis-name it.
    assert url.endswith("/TGB2_040_EP09_markers.txt")

    dl = client.get(url)
    assert dl.status_code == 200
    assert dl.headers["content-disposition"] == 'attachment; filename="TGB2_040_EP09_markers.txt"'
    assert dl.text.strip() == "Tom\t10:00:00:00\tV1\tYellow\tSam - Come on in\t1\t\tYellow"

    # token is single-use
    assert client.get(url).status_code == 404


def test_export_rejects_bad_timecode():
    r = client.post(
        "/api/export",
        json={"start_tc": "nope", "segments": [{"start": 0, "end": 1, "text": "x"}]},
    )
    assert r.status_code == 400


def test_job_status_unknown():
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_write_env_token_creates_replaces_and_preserves(tmp_path):
    env = tmp_path / ".env"
    # creates the file
    _write_env_token("hf_aaa", env)
    assert env.read_text() == "HF_TOKEN=hf_aaa\n"
    # replaces the existing line, keeps other content
    env.write_text("# comment\nHF_TOKEN=hf_old\nOTHER=1\n")
    _write_env_token("hf_new", env)
    body = env.read_text()
    assert "HF_TOKEN=hf_new" in body
    assert "hf_old" not in body
    assert "# comment" in body and "OTHER=1" in body


def test_set_token_rejects_garbage():
    assert client.post("/api/token", json={"token": "not-a-token"}).status_code == 400
    assert client.post("/api/token", json={"token": ""}).status_code == 400


def test_set_token_saves_and_enables(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv("HF_TOKEN", raising=False)  # restored at teardown
    r = client.post("/api/token", json={"token": "hf_TESTtoken123"})
    assert r.status_code == 200
    assert r.json()["diarization_enabled"] is True
    assert (tmp_path / ".env").read_text() == "HF_TOKEN=hf_TESTtoken123\n"
    assert client.get("/api/config").json()["diarization_enabled"] is True
