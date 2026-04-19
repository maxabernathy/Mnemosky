# 08 · Splash

Reference: `references/splash_loading.html`.

## Purpose

Cold-start surface shown while the detector and shell initialise. Also the first thing seen if the gallery display is power-cycled.

## Layout

Full-viewport, grass-bokeh stage (same recipe as the title screen — see `02_design_tokens.md > grass-bokeh background`).

Centered in the viewport, stacked vertically, gap 64 px at gallery scale:

1. **Wordmark** — `display.wordmark` (240 px italic serif), `color.surface.night` on the bright bokeh
2. **Tagline** — *"centrifugal data management"* in italic serif 44 px, `color.surface.night` at 62% opacity, letter-spacing 0.18 em, lowercase
3. **Loader** — 420 × 420 px mercury blob, cycling through the four variants (see below)
4. **Status line** — italic serif 46 px, `color.surface.night`, sentence case. Changes per loader
5. **Footer** — `mono.micro` (16 px tracked uppercase) in dark ink at 62%: `mnemozine · tracksuit · v0.3.0`

## Loader cycle

Runs on `motion.meditative` (7 s per loader, 28 s total before looping).

| Beat (s) | Loader | Status copy |
|---|---|---|
| 0–7 | `trefoil` | *"warming the optics"* |
| 7–14 | `eye` | *"resolving targets"* |
| 14–21 | `gather` | *"gathering frames"* |
| 21–28 | `drip` | *"calibrating pipeline"* |

Loader crossfade on `motion.settle`. Status text crossfades in sync.

## Dismissal

The splash **dismisses on a system signal** from the detector (`READY` event) — not on a timer. If the detector boots in under 7 s, the splash still shows a minimum of 7 s so the first beat completes. This is a dignity rule, not a technical one.

On dismiss: fade the whole splash to `color.surface.night` on `motion.settle` (260 ms), then reveal the shell in Observe mode with the dashboard already populated.

## Not a loading screen for modes

This splash is **only** for the cold-start. In-shell loaders (see `06_mercury_loaders.md`) are a different surface; they appear at specific thresholds inside an already-running shell.

## Desktop port

For the OpenCV shell port, the splash can either:

- **Run as a pywebview / `QWebEngineView` overlay** for a few seconds using the same HTML file (`references/splash_loading.html`) pointed at on disk; or
- Be replicated with Qt primitives (a full-window `QWidget` with a painted bokeh gradient, `QLabel` wordmark, and a `QSvgWidget` cycling through the four loaders).

Either is fine. The content and timing are the same.
