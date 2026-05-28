#!/bin/bash
# One-time setup for Avid Markup on an Apple Silicon Mac.
# Double-click this file (or run it in Terminal). It is safe to re-run.
#
# It creates the Python virtualenv, installs the app + ML stack, checks ffmpeg,
# stores YOUR OWN HuggingFace token, and builds a double-click AvidMarkup.app.

set -u
cd "$(dirname "$0")" || exit 1
REPO="$(pwd)"

say()  { printf '\n\033[1m%s\033[0m\n' "$1"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$1"; exit 1; }

say "Avid Markup setup — $REPO"

# --- 1. Find a supported Python (3.10–3.12; the ML stack doesn't support 3.13+) ---
PYBIN=""
for cand in python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ]; then
  die "Need Python 3.10, 3.11, or 3.12. Install one (e.g. 'brew install python@3.12') and re-run."
fi
ok "Using $($PYBIN --version) ($PYBIN)"

# --- 2. Create / reuse the virtualenv ---
if [ ! -d .venv ]; then
  say "Creating virtualenv (.venv)…"
  "$PYBIN" -m venv .venv || die "Could not create .venv"
fi
ok "Virtualenv ready"

# --- 3. Install the app + ML stack ---
say "Installing dependencies (this pulls the ML stack — a few minutes the first time)…"
./.venv/bin/pip install --upgrade pip >/dev/null 2>&1
./.venv/bin/pip install -e ".[ml,app]" || die "pip install failed — see the output above."
if ./.venv/bin/pip check >/dev/null 2>&1; then
  ok "Dependencies installed and consistent"
else
  warn "pip check reported a conflict — the app may still run, but flag this:"
  ./.venv/bin/pip check
fi

# --- 4. ffmpeg (decodes the audio; system dependency) ---
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg found"
else
  warn "ffmpeg is not installed (required to read audio)."
  if command -v brew >/dev/null 2>&1; then
    read -r -p "  Install it now with 'brew install ffmpeg'? [y/N] " a
    case "$a" in [yY]*) brew install ffmpeg && ok "ffmpeg installed" || warn "ffmpeg install failed — install it manually." ;;
                 *) warn "Skipped. Install ffmpeg before transcribing." ;; esac
  else
    warn "Homebrew not found. Install it from https://brew.sh then run 'brew install ffmpeg'."
  fi
fi

# --- 5. HuggingFace token (for speaker diarization) ---
# Each person uses their OWN free token. We never ship or share a token.
say "HuggingFace token (enables speaker separation)"
if [ -f .env ] && grep -q '^HF_TOKEN=..' .env; then
  ok ".env already has a token — leaving it as-is."
else
  echo "  1. Create a free token at https://huggingface.co/settings/tokens"
  echo "  2. Open https://huggingface.co/pyannote/speaker-diarization-community-1"
  echo "     and click 'Agree and access repository' to accept its terms."
  echo "  (You can skip this — the app still transcribes, but every line becomes one"
  echo "   unlabelled speaker.)"
  read -r -p "  Paste your token (starts with hf_), or press Enter to skip: " HF
  if [ -n "$HF" ]; then
    [ -f .env ] || cp .env.example .env 2>/dev/null || touch .env
    # Replace any existing HF_TOKEN line, else append.
    if grep -q '^HF_TOKEN=' .env; then
      tmp="$(mktemp)"; grep -v '^HF_TOKEN=' .env > "$tmp"; mv "$tmp" .env
    fi
    printf 'HF_TOKEN=%s\n' "$HF" >> .env
    ok "Saved your token to .env (gitignored — it stays on this machine)."
  else
    warn "Skipped. Diarization is off until you add HF_TOKEN to .env."
  fi
fi

# --- 6. Build the double-click launcher app (built locally → no Gatekeeper warning) ---
say "Building AvidMarkup.app…"
rm -rf AvidMarkup.app
SCRIPT="do shell script \"cd $(printf %q "$REPO") && nohup ./.venv/bin/python launcher.py >/dev/null 2>&1 &\""
if osacompile -o AvidMarkup.app -e "$SCRIPT" 2>/dev/null; then
  ok "Built AvidMarkup.app — double-click it to launch the app."
else
  warn "Could not build AvidMarkup.app. You can still launch with: ./.venv/bin/python launcher.py"
fi

say "Done. Double-click AvidMarkup.app to start (drag it to the Dock if you like)."
echo "  First transcription downloads the speech models (~1–6 GB) — that run is slower."
