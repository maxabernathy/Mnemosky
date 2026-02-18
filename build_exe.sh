#!/usr/bin/env bash
#
# Build Mnemosky as a portable Windows executable via Nuitka.
#
# Usage (Git Bash / MSYS2 / WSL on Windows):
#   ./build_exe.sh              # Minimal build (classical detection)
#   ./build_exe.sh --full       # Include scipy + onnxruntime
#   ./build_exe.sh --full --include-torch   # Include ultralytics + PyTorch
#   ./build_exe.sh --output-name my_detector
#
# Prerequisites:
#   pip install nuitka ordered-set opencv-python numpy
#

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────
FULL=0
INCLUDE_TORCH=0
OUTPUT_NAME="mnemosky"

# ── Parse arguments ─────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)          FULL=1; shift ;;
        --include-torch) INCLUDE_TORCH=1; FULL=1; shift ;;
        --output-name)   OUTPUT_NAME="$2"; shift 2 ;;
        --output-name=*) OUTPUT_NAME="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: $0 [--full] [--include-torch] [--output-name NAME]"
            echo ""
            echo "  --full            Include optional deps (scipy, onnxruntime)"
            echo "  --include-torch   Include ultralytics + PyTorch (implies --full, very large)"
            echo "  --output-name     Base name for the exe (default: mnemosky)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────
check_python_pkg() {
    python -c "import $1" 2>/dev/null
}

ok()   { echo "[OK] $1"; }
skip() { echo "[--] $1: not installed (skipped)"; }

# ── Preflight checks ───────────────────────────────────────────────────
echo "=== Mnemosky Nuitka Build ==="
echo ""

if ! command -v python &>/dev/null; then
    echo "ERROR: python not found on PATH."
    exit 1
fi

if ! check_python_pkg nuitka; then
    echo "ERROR: Nuitka is not installed."
    echo "  pip install nuitka ordered-set"
    exit 1
fi
ok "nuitka"

for dep in cv2 numpy; do
    if ! check_python_pkg "$dep"; then
        echo "ERROR: Required dependency '$dep' is not installed."
        echo "  pip install opencv-python numpy"
        exit 1
    fi
done
ok "opencv-python, numpy"

# Check optional deps
declare -A OPT_AVAILABLE
for dep in scipy onnxruntime ultralytics; do
    if check_python_pkg "$dep"; then
        OPT_AVAILABLE[$dep]=1
        ok "$dep"
    else
        OPT_AVAILABLE[$dep]=0
        skip "$dep"
    fi
done

echo ""

# ── Build Nuitka command ────────────────────────────────────────────────
CMD=(
    python -m nuitka
    --onefile
    --standalone
    "--output-filename=${OUTPUT_NAME}.exe"
    --output-dir=build

    # Keep console for CLI output
    --windows-console-mode=force

    # Exe metadata
    --product-name=Mnemosky
    --product-version=1.0.0
    "--file-description=Satellite and Airplane Trail Detector"

    # Core dependencies
    --include-package=cv2
    --include-package-data=cv2
    --include-package=numpy

    # Multiprocessing support
    --enable-plugin=multiprocessing
)

# ── Optional dependencies ──────────────────────────────────────────────
if [[ $FULL -eq 1 && ${OPT_AVAILABLE[scipy]} -eq 1 ]]; then
    CMD+=(--include-package=scipy)
    echo "Including: scipy"
fi

if [[ $FULL -eq 1 && ${OPT_AVAILABLE[onnxruntime]} -eq 1 ]]; then
    CMD+=(--include-package=onnxruntime --include-package-data=onnxruntime)
    echo "Including: onnxruntime"
fi

if [[ $INCLUDE_TORCH -eq 1 && ${OPT_AVAILABLE[ultralytics]} -eq 1 ]]; then
    CMD+=(
        --include-package=ultralytics --include-package-data=ultralytics
        --include-package=torch --include-package-data=torch
    )
    echo "Including: ultralytics + torch (this will be large)"
else
    CMD+=(
        --nofollow-import-to=torch
        --nofollow-import-to=ultralytics
        --nofollow-import-to=torchvision
        --nofollow-import-to=torchaudio
    )
fi

# Exclude unnecessary bloat
CMD+=(
    --nofollow-import-to=pytest
    --nofollow-import-to=setuptools
    --nofollow-import-to=pip
    --nofollow-import-to=tkinter
)

# Source file
CMD+=(satellite_trail_detector.py)

# ── Run the build ───────────────────────────────────────────────────────
echo ""
echo "Build command:"
echo "  ${CMD[*]}"
echo ""
echo "Starting Nuitka compilation (this takes several minutes)..."
echo ""

if ! "${CMD[@]}"; then
    echo ""
    echo "Build FAILED. Check the output above for errors."
    echo ""
    echo "Common fixes:"
    echo "  - Install a C compiler: Visual Studio Build Tools or MinGW64"
    echo "  - Nuitka can auto-download MinGW64 — answer 'yes' if prompted"
    echo "  - Ensure all deps are installed in the active Python environment"
    exit 1
fi

# ── Copy output ─────────────────────────────────────────────────────────
EXE_SRC="build/${OUTPUT_NAME}.exe"
EXE_DST="${OUTPUT_NAME}.exe"

if [[ -f "$EXE_SRC" ]]; then
    cp "$EXE_SRC" "$EXE_DST"
elif [[ -f "build/${OUTPUT_NAME}.dist/${OUTPUT_NAME}.exe" ]]; then
    cp "build/${OUTPUT_NAME}.dist/${OUTPUT_NAME}.exe" "$EXE_DST"
fi

if [[ -f "$EXE_DST" ]]; then
    SIZE=$(du -h "$EXE_DST" | cut -f1)
    echo ""
    echo "Build SUCCESS: ${EXE_DST} (${SIZE})"
    echo ""
    echo "Test it:  ./${OUTPUT_NAME}.exe --help"
else
    echo ""
    echo "Build completed but could not locate output exe."
    echo "Check the build/ directory."
fi
