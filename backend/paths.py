"""Filesystem paths that differ between dev (repo checkout) and the frozen .app.

In the frozen app the bundle is read-only and code lives under `sys._MEIPASS`, so
read-only resources (the frontend, the bundled diarization models) are read from the
bundle, while writable state (uploaded audio, the `.env` that stores the optional HF
token) goes to `~/Library/Application Support/AvidMarkup`. In dev, everything stays in
the repo root exactly as before.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_NAME = "AvidMarkup"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    """True when running inside a PyInstaller-frozen bundle."""
    return bool(getattr(sys, "frozen", False))


def resource_dir() -> Path:
    """Read-only resources: the frontend and the bundled diarization models.

    PyInstaller unpacks bundled data under `sys._MEIPASS`; in dev it's the repo root."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return _REPO_ROOT


def data_dir() -> Path:
    """User-writable app data (uploads, saved .env). Created if missing.

    The frozen bundle is read-only, so writable state lives in Application Support;
    in dev it's the repo root (unchanged behaviour)."""
    d = Path.home() / "Library" / "Application Support" / _APP_NAME if is_frozen() else _REPO_ROOT
    d.mkdir(parents=True, exist_ok=True)
    return d
