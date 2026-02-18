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
import shutil
import subprocess
import sys


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
        help="Include optional deps (scipy, onnxruntime). Without this flag, "
             "only classical detection (OpenCV + NumPy) is bundled.",
    )
    parser.add_argument(
        "--include-torch",
        action="store_true",
        help="Include PyTorch and ultralytics (very large). Implies --full.",
    )
    parser.add_argument(
        "--output-name",
        default="mnemosky",
        help="Base name for the output executable (default: mnemosky)",
    )
    args = parser.parse_args()

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
                      ("ultralytics", "ultralytics")]:
        optional_available[name] = check_dependency(name, pkg)
        status = "OK" if optional_available[name] else "not installed (skipped)"
        print(f"[{'OK' if optional_available[name] else '--'}] {name}: {status}")

    print()

    # ── Build Nuitka command ────────────────────────────────────────────
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
        "--product-version=1.0.0",
        "--file-description=Satellite and Airplane Trail Detector",

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
