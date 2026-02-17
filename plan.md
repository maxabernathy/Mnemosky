# Plan: Radon Debug Preview Window

## Overview

Add a new `show_radon_preview()` function — an interactive GUI for visually debugging the Radon pipeline, following the same dark-grey/fluorescent-accent theme and single-window architecture as the existing `show_preprocessing_preview()`.

Triggered by `--preview` when `--algorithm radon` is specified (instead of the default preprocessing preview).

## Layout

```
┌──────────────────────────────────┬──────────────┐
│                                  │  MNEMOSKY    │
│         ORIGINAL                 │  Radon Debug │
│      (full-height left column)   │              │
│                                  │  [6 sliders] │
│                                  │              │
│                                  │  CONTROLS    │
│                                  │  help text   │
├──────────┬──────────┬───────┬────┴──────────────┤
│ RESIDUAL │ SINOGRAM │ LSD   │ DETECTIONS        │
│ (star-   │ (SNR     │ LINES │ (final results    │
│  cleaned)│  heatmap)│       │  after PCF)       │
├──────────┴──────────┴───────┴───────────────────┤
│ ◉ Frame  ████████░░░░░░░░░░░░░  Frame 123/9000 │
└─────────────────────────────────────────────────┘
```

**Top row**: Original frame (left, ~58% width) + sidebar with 6 sliders and controls (right).

**Bottom row**: 4 panels showing pipeline stages side-by-side:

1. **RESIDUAL** — Star-cleaned, background-subtracted image displayed as a teal-tinted intensity map. Star mask overlaid as dim red markers. Shows what the Radon transform actually "sees".

2. **SINOGRAM** — The SNR sinogram (angles on X-axis, offsets on Y-axis) as an amber/orange heatmap. Detected peaks shown as bright markers. The visual "fingerprint" of the Radon transform — each horizontal linear streak in the image appears as a bright spot here.

3. **LSD LINES** — CLAHE-enhanced grayscale frame with LSD line segments overlaid in bright green-yellow. Shows Stage 1 detections before classification.

4. **DETECTIONS** — Original frame with all final detections overlaid: Radon candidates in amber (pre-PCF), PCF-confirmed streaks in bright green, rejected candidates as dim red dashes. Shows the end result of the full pipeline.

**Status bar**: Full-width frame slider + frame info (same pattern as preprocessing preview).

## 6 Tunable Parameters (Sliders)

| # | Parameter | Label | Default | Range | Internal storage |
|---|-----------|-------|---------|-------|-----------------|
| 1 | `radon_snr_threshold` | Radon SNR | 3.0 | 1.0–8.0 | ×10 int (10–80) |
| 2 | `pcf_ratio_threshold` | PCF Ratio | 2.0 | 0.5–5.0 | ×10 int (5–50) |
| 3 | `star_mask_sigma` | Star Mask σ | 5.0 | 2.0–10.0 | ×10 int (20–100) |
| 4 | `lsd_log_eps` | LSD Signif. | 1.0 | -2.0–5.0 | ×10 int, offset (-20–50) |
| 5 | `pcf_kernel_len` | PCF Kernel | 31 | 5–81 | int (odd) |
| 6 | `min_streak_length` | Min Length | 50 | 10–200 | int |

These cover the most impactful parameters across all three Radon stages (LSD, Radon peak finding, PCF filtering).

## Panel Accent Colors (BGR, matching existing theme)

- **RESIDUAL**: Teal `(200, 180, 60)` — distinct from preprocessing's cyan edges
- **SINOGRAM**: Amber/orange `(50, 160, 255)` — warm, distinct from MF's magenta
- **LSD LINES**: Green-yellow `(80, 255, 120)` — segment overlay colour
- **DETECTIONS**: Multi-colour — amber for raw Radon candidates, bright green for PCF-confirmed, dim red for rejected

## Implementation Changes

### 1. New function: `show_radon_preview(video_path, initial_params=None)`

Placed right after `show_preprocessing_preview()` (after line ~1099). Follows the identical architectural pattern:

- Same theme colours (`BG_DARK`, `BG_PANEL`, etc.) plus new panel-specific accents
- Same screen-size detection, layout scaling, sidebar width computation
- Same custom slider drawing, hit-testing, and mouse drag interaction
- Same dirty-flag optimization (only recompute when parameters change)
- Same keyboard controls (SPACE/ENTER to accept, ESC to cancel, R to reset, N/P to navigate frames)
- No trail marking (not relevant for Radon debugging)

**Core computation (on parameter or frame change):**
1. Convert frame to grayscale
2. Background subtraction (spatial median, or temporal if available)
3. Noise estimation (MAD-based)
4. Star masking with tunable sigma → produces `cleaned` and `star_mask`
5. Downsample `cleaned` for Radon (matching the real pipeline's 250k pixel cap)
6. Run `_radon_transform()` → sinogram
7. SNR-normalize sinogram, baseline removal, peak finding with tunable threshold
8. Run `_perpendicular_cross_filter()` on Radon candidates with tunable ratio/kernel
9. Run `_detect_lines_lsd()` on CLAHE-enhanced frame with tunable log_eps
10. Build 4-panel display + sidebar

**Performance:** The Radon transform on the downsampled (~500×500) image is fast enough for interactive use (~100-300ms per frame on CPU). The dirty-flag system ensures zero CPU when idle.

**Return value:** Dict with the 6 tunable parameters (converted to real-valued), or `None` if cancelled. These flow into `RadonStreakDetector.__init__()` to override its defaults.

### 2. Wire into `main()` (line ~7519)

Change the `--preview` handling:

```python
if args.preview:
    if args.algorithm == 'radon':
        radon_params = show_radon_preview(args.input)
        if radon_params is not None:
            # Store Radon-specific params to pass to RadonStreakDetector
            radon_preview_params = radon_params
    else:
        preprocessing_params = show_preprocessing_preview(args.input)
        ...
```

### 3. Thread Radon preview params through `process_video()` → `RadonStreakDetector`

Add a `radon_params` kwarg to `process_video()` that gets forwarded to the `RadonStreakDetector` constructor. The constructor applies the preview overrides to `self.radon_snr_threshold`, `self.pcf_ratio_threshold`, `self.pcf_kernel_length`, and the star mask sigma / LSD log_eps parameters.

### 4. Update CLAUDE.md

Add the new function to the file locations table and document the `--preview` + `--algorithm radon` interaction.

## Keyboard Controls

Same as preprocessing preview:
- **SPACE / ENTER** — Accept current parameters
- **ESC** — Cancel, use defaults
- **R** — Reset to defaults
- **N** — Jump forward 1 second
- **P** — Jump back 1 second

## What This Does NOT Change

- The existing `show_preprocessing_preview()` is untouched
- The Radon detection pipeline logic is untouched (we instantiate a temporary `RadonStreakDetector` to call its methods, but don't modify any algorithm code)
- No new external dependencies
- Single-file architecture preserved
