#!/bin/bash
# Build (and optionally sign + notarize) Avid Markup into a one-download .app/.dmg.
#
#   ./packaging/build_app.sh              # build + ad-hoc sign (local validation)
#   ./packaging/build_app.sh --release    # build + Developer-ID sign + notarize + dmg
#
# Release mode needs an Apple Developer account and these env vars:
#   DEVELOPER_ID   "Developer ID Application: Your Name (TEAMID)"
#   AC_PROFILE     a notarytool keychain profile (xcrun notarytool store-credentials)
#
# Apple-Silicon only. Run from the repo root with the .venv active or present.

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
RELEASE=0
[ "${1:-}" = "--release" ] && RELEASE=1

say() { printf '\n\033[1m%s\033[0m\n' "$1"; }

# 1. Models must be present so the spec can bundle them.
if [ ! -f "$ROOT/models/diarization/wespeaker_en_voxceleb_resnet34_LM.onnx" ]; then
  say "Fetching diarization models…"
  "$ROOT/scripts/fetch_diarization_models.sh"
fi

# 2. Freeze.
say "Running PyInstaller (this takes several minutes and writes a multi-GB bundle)…"
rm -rf "$ROOT/build" "$ROOT/dist/AvidMarkup" "$ROOT/dist/AvidMarkup.app"
"$PY" -m PyInstaller --noconfirm --clean packaging/AvidMarkup.spec
APP="$ROOT/dist/AvidMarkup.app"
[ -d "$APP" ] || { echo "Build failed: $APP not produced"; exit 1; }

# 3. Bundle the license texts required by LGPL/third-party notices.
mkdir -p "$APP/Contents/Resources/licenses"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$APP/Contents/Resources/licenses/" 2>/dev/null || true
[ -f "$ROOT/LICENSE" ] && cp "$ROOT/LICENSE" "$APP/Contents/Resources/licenses/" || true

ENT="$ROOT/packaging/entitlements.plist"

if [ "$RELEASE" -eq 0 ]; then
  # 4a. Ad-hoc sign for local validation (won't pass Gatekeeper on other Macs, but lets
  #     you launch and test the freeze on this machine).
  say "Ad-hoc signing (local validation build)…"
  codesign --force --deep -s - --entitlements "$ENT" --options runtime "$APP" 2>/dev/null \
    || codesign --force --deep -s - "$APP"
  say "Done (ad-hoc): $APP"
  echo "Launch with:  open \"$APP\"   (or ./dist/AvidMarkup.app/Contents/MacOS/AvidMarkup for logs)"
  exit 0
fi

# 4b. Release: Developer-ID sign every nested binary, then the app, with hardened runtime.
: "${DEVELOPER_ID:?set DEVELOPER_ID to your 'Developer ID Application: …' identity}"
say "Signing nested binaries with Developer ID…"
find "$APP" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 \
  | xargs -0 -I{} codesign --force --timestamp --options runtime \
      --entitlements "$ENT" -s "$DEVELOPER_ID" "{}"
codesign --force --deep --timestamp --options runtime \
  --entitlements "$ENT" -s "$DEVELOPER_ID" "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

# 5. Package a .dmg and notarize it.
say "Building .dmg…"
DMG="$ROOT/dist/AvidMarkup.dmg"
rm -f "$DMG"
"$PY" -m dmgbuild -s packaging/dmg_settings.py "Avid Markup" "$DMG" 2>/dev/null \
  || hdiutil create -volname "Avid Markup" -srcfolder "$APP" -ov -format UDZO "$DMG"

say "Notarizing (submits to Apple, waits)…"
: "${AC_PROFILE:?set AC_PROFILE to a notarytool keychain profile}"
xcrun notarytool submit "$DMG" --keychain-profile "$AC_PROFILE" --wait
xcrun stapler staple "$DMG"
xcrun stapler staple "$APP"
say "Done (release): $DMG"
