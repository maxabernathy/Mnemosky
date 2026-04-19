# 09 · Hardware & Delivery

## The phased rollout

### Phase 1 — Ratify the shell (this handoff)

Output: agreement on the four-mode shell grammar — Tune · Observe · Review · Inspect + Ledger ribbon + Observer card. Everything else in the package follows from that grammar.

No code changes in phase 1; this is review.

### Phase 2 — Port theme into the existing OpenCV shell (v0.3.x)

Landing the visual system inside `satellite_trail_detector.py` with no framework migration:

- Warm palette (night/night2/panel + ink tiers + semantic).
- Type pairing where possible (OpenCV's `putText` has limited font options — use Tinos or Liberation Serif Italic as serif substitute; JetBrains Mono or platform monospace for mono).
- Ledger ribbon rendered as a fixed-height strip at the bottom of every existing OpenCV window.
- Observer card as a single-line mono header above the canvas.
- Mercury loaders as a separate PNG-sequence overlay at thresholds (OpenCV can't render SVG; pre-render each loader at 280 × 280 px × N frames and play them back via `cv2.imshow` on a compositing window).

Partial — OpenCV windowing won't deliver the same polish, but the language lands and users see progress toward the museum build.

### Phase 3 — Gallery build

A new shell in Qt/PySide6 (recommended) or a web shell (Tauri / Electron with a Python backend).

**Recommendation: PySide6.** Reasons:

- Matches the Python core; the existing detector runs in-process.
- Native 8K scaling with `QGuiApplication.setHighDpiScaleFactorRoundingPolicy`.
- Capacitive touch supported natively via `QTouchEvent`.
- Room-sensor hardware integrates via `pyserial` or `RPi.GPIO` in the same process.
- Offline by design — no browser attack surface on a wall-mounted kiosk.
- QSS (Qt Style Sheets) can express the full design-token system.

A web shell is viable if the team is more comfortable in TypeScript; use Tauri over Electron (smaller footprint, Rust-based IPC to the Python detector). Not recommended for a first install.

## Hardware spec

### Display

- **8K panel**, 85" or larger, 16:9. Reference class: LG 88ZX, Samsung QN900.
- Wall-mounted at **1.2 m to panel centre** (seated/standing mixed audience).
- Matte finish preferred; gloss is fighting the grass-bokeh palette under gallery lighting.
- HDMI 2.1 @ 60 Hz minimum. DisplayPort 1.4 acceptable.

### Compute

- Workstation-class: **RTX 4090 (24 GB VRAM) or better**, 64 GB system RAM, NVMe SSD ≥ 2 TB.
- Ubuntu 22.04 LTS or later. Windows 11 acceptable.
- The detector is the workload; don't under-spec the GPU. The UI shell itself is trivial.

### Console

- **Capacitive touch bar at standing height** (~1.05 m), 32" 4K touchscreen mounted flush.
- USB to the main workstation.
- Surfaces the keyboard shortcut cheatsheet visible in every mode's sidebar "Keys" block. Each key is a touch target in docent mode.

### Room sensor (optional, recommended)

- Ultrasonic or ToF presence sensor (e.g. **Benewake TF-Luna** or a simple PIR + ultrasonic combo).
- Serial over USB to the workstation.
- Reports presence and approximate distance to the display at 5 Hz.
- If omitted, the attract loop degrades to a pure timer-driven loop — still works.

### Docent key

- Physical **key switch** (e.g. a `Adafruit Industries #3142` keylock) wired to a USB-HID controller.
- Two-state: attract / docent.
- Not a software toggle — the physicality is intentional; docents feel the mode change.

## Venue requirements

The following are **unresolved** and the developer should gate phase-3 work on getting answers:

1. **Venue dimensions** — wall size, room size, seating positions. Determines viewing-distance assumptions and speaker coverage (if audio is added).
2. **Ambient light spec** — measured lux during open hours. Determines whether the dark palette needs a "gallery bright" variant.
3. **Corpus of footage** — which nights, how many hours, what cameras/observers. Determines how often confirmed detections surface in Beat 1.
4. **Docent script language** — English only, or multilingual? Affects the Observer card's free-text notes field (it's the one user-facing copy surface).
5. **Final shell decision** — Qt vs. web.
6. **Hardware budget** — 8K panel, workstation, console touch, sensor, docent key, mounting. Ballpark USD 25–45 k depending on panel choice.

## Project conventions to respect

- **Single source of truth for the core:** `satellite_trail_detector.py`. The shell consumes its output; it does not re-implement any detection logic.
- **Ledger JSONL format:** whatever the current detector emits. The Ledger ribbon binds to the in-memory session counts, not to parsing the file.
- **No new dependencies on the detector side.** Shell-only libs are fine.

## Non-goals

- No cloud sync, telemetry, or analytics.
- No user accounts.
- No internationalisation in v2.
- No tutorial mode or onboarding (the attract loop *is* the onboarding).

## Definition of done (phase 3)

1. The shell boots into the splash, transitions to Observe when the detector reports ready.
2. All four modes are reachable via `1 / 2 / 3 / 4` and render per `04_modes.md`.
3. The Ledger ribbon and Observer card appear in every mode per `05_ledger_and_observer.md`.
4. The four mercury loaders appear at the thresholds listed in `06_mercury_loaders.md`.
5. The attract loop runs per `07_attract_loop.md` with presence-sensor input (or pure timer if sensor is absent).
6. Docent key toggles docent mode.
7. All type roles in `tokens.json` are implemented and pass the accessibility checks in `02_design_tokens.md`.
8. The full design matches `references/Mnemosky UI Audit.html` section 03 at gallery scale, side-by-side with a real detector run.
