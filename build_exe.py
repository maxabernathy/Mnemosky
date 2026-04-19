#!/usr/bin/env python3
"""
Nuitka build script for Mnemosky — produces a portable Windows executable.

Usage (on Windows):
    # Minimal build (classical detection only, ~80-120 MB)
    python build_exe.py

    # Full build with neural network support (~300+ MB without PyTorch)
    python build_exe.py --full

    # Include PyTorch/Ultralytics (very large, ~1.5 GB+)
    python build_exe.py --full --include-torch

    # Custom output name
    python build_exe.py --output-name my_detector

Requirements:
    pip install nuitka ordered-set   # ordered-set speeds up Nuitka
    pip install opencv-python numpy  # runtime deps (must be installed)

    # For --full builds, also install the optional deps you want bundled:
    pip install scipy
    pip install onnxruntime          # lightweight NN backend
    pip install ultralytics          # YOLOv8/v11 backend (pulls PyTorch)

Notes:
    - Must be run on Windows (or cross-compiled, but native is recommended)
    - Requires a C compiler: MinGW64 or MSVC (Visual Studio Build Tools)
    - Nuitka will download MinGW64 automatically if neither is found
    - The output exe is placed in the current directory
    - Build artifacts go into build/ (gitignored)
"""

import argparse
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path(__file__).parent / "satellite_trail_detector.py"


def read_source_version():
    """Pull __version__ from satellite_trail_detector.py."""
    try:
        text = SOURCE.read_text(encoding="utf-8")
    except OSError:
        return "0.0.0"
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def check_dependency(name, package=None):
    """Check if a Python package is importable. Returns True/False."""
    try:
        importlib.import_module(package or name)
        return True
    except ImportError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build Mnemosky as a portable Windows executable via Nuitka"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include optional deps (scipy, onnxruntime, rawpy, exifread). "
             "Without this flag, only classical detection (OpenCV + NumPy) is bundled.",
    )
    parser.add_argument(
        "--include-torch",
        action="store_true",
        help="Include PyTorch and ultralytics (very large). Implies --full.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the output executable name. Defaults to "
             "'mnemosky-<tag>' when --tag is set, else 'mnemosky'.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag to embed in the exe filename and product version "
             "metadata (e.g. 'alpha-april'). Also accepted as part of the "
             "exe filename if --output-name is not explicitly provided.",
    )
    args = parser.parse_args()

    source_version = read_source_version()
    # Resolve the actual output name: explicit --output-name wins, else
    # mnemosky-<tag> if --tag given, else just mnemosky.
    if args.output_name is None:
        args.output_name = f"mnemosky-{args.tag}" if args.tag else "mnemosky"

    if args.include_torch:
        args.full = True

    # ── Preflight checks ────────────────────────────────────────────────
    print("=== Mnemosky Nuitka Build ===\n")

    if not check_dependency("nuitka"):
        print("ERROR: Nuitka is not installed.")
        print("  pip install nuitka ordered-set")
        sys.exit(1)

    missing = []
    for dep in ["cv2", "numpy"]:
        if not check_dependency(dep):
            missing.append(dep)
    if missing:
        print(f"ERROR: Required dependencies not installed: {', '.join(missing)}")
        print("  pip install opencv-python numpy")
        sys.exit(1)

    print("[OK] nuitka")
    print("[OK] opencv-python, numpy")

    # Check optional deps
    optional_available = {}
    for name, pkg in [("scipy", "scipy"), ("onnxruntime", "onnxruntime"),
                      ("ultralytics", "ultralytics"), ("rawpy", "rawpy"),
                      ("exifread", "exifread")]:
        optional_available[name] = check_dependency(name, pkg)
        status = "OK" if optional_available[name] else "not installed (skipped)"
        print(f"[{'OK' if optional_available[name] else '--'}] {name}: {status}")

    print()
    print(f"Source version: {source_version}")
    if args.tag:
        print(f"Build tag:      {args.tag}")
    print(f"Output name:    {args.output_name}.exe")
    print()

    # ── Build Nuitka command ────────────────────────────────────────────
    # Nuitka --product-version must be digits-and-dots only.  Strip any
    # non-numeric characters from the source version and append .0 to pad
    # to four fields.  Keep the full human-readable string in
    # --file-version / file-description instead.
    numeric_version = re.sub(r"[^0-9.]", "", source_version) or "0.0.0"
    while numeric_version.count(".") < 3:
        numeric_version += ".0"
    descriptor = f"Satellite and Airplane Trail Detector (v{source_version}"
    descriptor += f", {args.tag})" if args.tag else ")"

    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",
        "--standalone",
        f"--output-filename={args.output_name}.exe",
        "--output-dir=build",

        # Keep console window for CLI output
        "--windows-console-mode=force",

        # Product metadata (embedded in exe properties)
        "--product-name=Mnemosky",
        f"--product-version={numeric_version}",
        f"--file-description={descriptor}",

        # ── Core dependencies ───────────────────────────────────────
        # OpenCV: include the full package (ships ffmpeg DLLs, data files)
        "--include-package=cv2",
        "--include-package-data=cv2",

        # NumPy
        "--include-package=numpy",

        # ── Multiprocessing ─────────────────────────────────────────
        # Nuitka handles multiprocessing.freeze_support() automatically
        # when it detects multiprocessing imports, but be explicit:
        "--enable-plugin=multiprocessing",
    ]

    # ── Optional dependencies ───────────────────────────────────────
    # Only include what's actually installed AND requested

    # scipy — improves Radon NMS quality
    if args.full and optional_available["scipy"]:
        cmd += ["--include-package=scipy"]
        print("Including: scipy")

    # onnxruntime — lightweight NN backend
    if args.full and optional_available["onnxruntime"]:
        cmd += [
            "--include-package=onnxruntime",
            "--include-package-data=onnxruntime",
        ]
        print("Including: onnxruntime")

    # ultralytics + torch — heavy NN backend
    if args.include_torch and optional_available["ultralytics"]:
        cmd += [
            "--include-package=ultralytics",
            "--include-package=torch",
            "--include-package-data=ultralytics",
            "--include-package-data=torch",
        ]
        print("Including: ultralytics + torch (this will be large)")
    elif not args.include_torch:
        # Explicitly exclude torch to prevent accidental inclusion
        cmd += [
            "--nofollow-import-to=torch",
            "--nofollow-import-to=ultralytics",
            "--nofollow-import-to=torchvision",
            "--nofollow-import-to=torchaudio",
        ]

    # rawpy — RAW image decoding (ARW, CR2, NEF, DNG)
    if args.full and optional_available["rawpy"]:
        cmd += [
            "--include-package=rawpy",
            "--include-package-data=rawpy",
        ]
        print("Including: rawpy")

    # exifread — EXIF metadata for RAW files
    if args.full and optional_available["exifread"]:
        cmd += ["--include-package=exifread"]
        print("Including: exifread")

    # Always exclude test/debug bloat
    cmd += [
        "--nofollow-import-to=pytest",
        "--nofollow-import-to=setuptools",
        "--nofollow-import-to=pip",
        "--nofollow-import-to=tkinter",
    ]

    # The source file
    cmd.append("satellite_trail_detector.py")

    # ── Run the build ───────────────────────────────────────────────
    print(f"\nBuild command:\n  {' '.join(cmd)}\n")
    print("Starting Nuitka compilation (this takes several minutes)...\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nBuild FAILED. Check the output above for errors.")
        print("\nCommon fixes:")
        print("  - Install a C compiler: Visual Studio Build Tools or MinGW64")
        print("  - Nuitka can auto-download MinGW64 — answer 'yes' if prompted")
        print("  - Ensure all deps are installed in the active Python environment")
        sys.exit(1)

    # Move the exe from build/ to the project root for convenience
    import os
    src = os.path.join("build", f"{args.output_name}.exe")
    dst = f"{args.output_name}.exe"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"\nBuild SUCCESS: {dst} ({size_mb:.1f} MB)")
        print(f"\nTest it:  .\\{args.output_name}.exe --help")
    else:
        # onefile output location can vary — check alternative paths
        alt = os.path.join("build", f"{args.output_name}.dist",
                           f"{args.output_name}.exe")
        if os.path.exists(alt):
            shutil.copy2(alt, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            print(f"\nBuild SUCCESS: {dst} ({size_mb:.1f} MB)")
        else:
            print(f"\nBuild completed but could not locate output exe.")
            print(f"Check the build/ directory.")


if __name__ == "__main__":
    main()
