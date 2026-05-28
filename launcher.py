"""Native macOS window for Avid Markup.

Spawns the existing FastAPI server (`uvicorn backend.app:app`) as a subprocess and
points a pywebview/WKWebView window at it. The server runs exactly as it does from
the terminal — same `.env` loading, same MLX single-thread executor — so wrapping it
this way changes none of the working machinery; it only removes the terminal launch.

Run:  python launcher.py
"""

from __future__ import annotations

import atexit
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import webview

BASE_DIR = Path(__file__).resolve().parent

# How long to wait for the server to answer /api/config before giving up.
STARTUP_TIMEOUT_S = 30.0

LOADING_HTML = """
<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  html,body{height:100%;margin:0}
  body{display:flex;align-items:center;justify-content:center;
       font:16px -apple-system,system-ui,sans-serif;color:#e7eaf0;background:#0c0d10}
  .box{text-align:center}
  .spin{width:28px;height:28px;margin:0 auto 16px;border:3px solid #2a2d36;
        border-top-color:#f2c14e;border-radius:50%;animation:r .8s linear infinite}
  @keyframes r{to{transform:rotate(360deg)}}
  .small{color:#8a8f9c;font-size:13px;margin-top:8px}
</style></head><body><div class="box">
  <div class="spin"></div>
  <div>Starting Avid Markup…</div>
  <div class="small">First run downloads the speech models (~1–6&nbsp;GB) — that one's slower.</div>
</div></body></html>
"""


def _free_port() -> int:
    """Grab a free TCP port by binding to 0 and reading what the OS assigned."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(port: int) -> subprocess.Popen:
    """Launch the uvicorn server in its own process group (so we can reap the whole
    group on quit) with the full parent environment (so an exported HF_TOKEN wins,
    matching terminal behaviour)."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", str(port)],
        cwd=str(BASE_DIR),
        env=os.environ.copy(),
        start_new_session=True,
        stderr=subprocess.PIPE,
    )


def _wait_until_ready(port: int, proc: subprocess.Popen) -> str | None:
    """Poll the cheap /api/config endpoint until the server answers. Returns an
    error string if the server died or never came up, else None."""
    url = f"http://localhost:{port}/api/config"
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:  # server exited before becoming ready
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            return f"The server stopped before it was ready.\n\n{err[-2000:]}"
        try:
            with urllib.request.urlopen(url, timeout=1) as resp:
                if resp.status == 200:
                    return None
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.3)
    return f"Timed out after {STARTUP_TIMEOUT_S:.0f}s waiting for the server to start."


def _stop_server(proc: subprocess.Popen) -> None:
    """Terminate the whole process group; fall back to killing the lone process."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def main() -> None:
    port = _free_port()
    proc = _start_server(port)
    atexit.register(_stop_server, proc)

    window = webview.create_window(
        "Avid Markup", html=LOADING_HTML, width=1180, height=860,
        min_size=(900, 640),
    )

    def _boot() -> None:
        error = _wait_until_ready(port, proc)
        if error:
            safe = error.replace("<", "&lt;").replace(">", "&gt;")
            window.load_html(
                f'<body style="font:14px -apple-system,sans-serif;color:#e06c75;'
                f'background:#0c0d10;padding:24px;white-space:pre-wrap">{safe}</body>'
            )
            return
        window.load_url(f"http://localhost:{port}")

    # ALLOW_DOWNLOADS makes WKWebView honour the export's Content-Disposition
    # download and show a native save panel — so the export flow needs no changes.
    webview.settings["ALLOW_DOWNLOADS"] = True
    try:
        webview.start(_boot)  # blocks on the main thread until the window closes
    finally:
        _stop_server(proc)


if __name__ == "__main__":
    main()
