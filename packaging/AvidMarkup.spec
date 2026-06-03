# PyInstaller spec for Avid Markup — Apple-Silicon one-download .app.
#
# Invoke from the repo root:  pyinstaller packaging/AvidMarkup.spec
# Produces dist/AvidMarkup.app (onedir). Signing/notarization is done afterwards by
# packaging/build_app.sh (entitlements in packaging/entitlements.plist).
#
# The heavy ML stack (torch, mlx, whisperx, pyannote, sherpa-onnx, onnxruntime, PyAV)
# is collected wholesale; audio is decoded in-process by PyAV so no ffmpeg binary ships.

import os

from PyInstaller.utils.hooks import collect_all, copy_metadata

ROOT = os.path.dirname(SPECPATH)  # packaging/ -> repo root  # noqa: F821

datas = [
    (os.path.join(ROOT, "frontend"), "frontend"),
    (os.path.join(ROOT, "models", "diarization"), "models/diarization"),
]
binaries = []
hiddenimports = []

# Wholesale-collect the packages that carry native libs and/or data files. Wrapped so a
# package absent from a given install (e.g. an optional engine) doesn't abort the build.
_COLLECT = [
    "torch", "mlx", "mlx_whisper", "mlx_lm", "parakeet_mlx",
    "whisperx", "pyannote", "speechbrain", "asteroid_filterbanks",
    "lightning", "pytorch_lightning", "lightning_fabric",
    "sherpa_onnx", "onnxruntime", "av",
    "transformers", "huggingface_hub", "tokenizers", "safetensors",
    "numba", "llvmlite", "ctranslate2", "faster_whisper",
    "librosa", "soundfile", "audioread", "soxr", "lazy_loader",
    "scipy", "sklearn", "pandas",
    "uvicorn", "fastapi", "starlette", "pydantic", "pydantic_core",
    "python_multipart", "multipart", "dotenv",
]
for pkg in _COLLECT:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] skip collect_all({pkg!r}): {exc}")

# Packages that read their own dist metadata at runtime (importlib.metadata.version);
# PyInstaller drops it unless copied explicitly.
for pkg in ["torch", "tqdm", "transformers", "huggingface_hub", "lightning",
            "pytorch_lightning", "speechbrain", "whisperx", "numpy", "sherpa_onnx"]:
    try:
        datas += copy_metadata(pkg)
    except Exception as exc:  # noqa: BLE001
        print(f"[spec] skip copy_metadata({pkg!r}): {exc}")

hiddenimports += ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                  "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on",
                  "backend.app", "backend.transcribe", "backend.diarize_sherpa",
                  "backend.speaker_correction", "backend.triage", "backend.audio"]

a = Analysis(  # noqa: F821
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide6", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AvidMarkup",
    debug=False,
    strip=False,
    upx=False,  # UPX corrupts dylibs / breaks notarization — never enable.
    console=False,
)
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AvidMarkup",
)
app = BUNDLE(  # noqa: F821
    coll,
    name="AvidMarkup.app",
    icon=None,
    bundle_identifier="com.tomandrews.avidmarkup",
    info_plist={
        "CFBundleName": "Avid Markup",
        "CFBundleDisplayName": "Avid Markup",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
