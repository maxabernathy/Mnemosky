# CLAUDE.md - AI Assistant Guide for Mnemosky

## Project Overview

**Mnemosky** is a satellite and airplane trail detector for MP4 videos. It uses classical computer vision techniques to identify and classify celestial trails in night sky footage, distinguishing between satellites (smooth, uniform trails) and airplanes (dotted patterns with bright navigation lights). Detected trails can optionally be exported as a YOLO-format ML dataset for training object detection models.

Preprocessing adjustments

<img width="1861" height="931" alt="image" src="https://github.com/user-attachments/assets/053cb18d-e020-44a2-b73f-3d9374eef083" />

Debug functionality

<img width="1366" height="1024" alt="image" src="https://github.com/user-attachments/assets/6911a2f1-840d-4e76-b08f-5d59583c456c" />

Output

<img width="1366" height="1024" alt="image" src="https://github.com/user-attachments/assets/07d3a4d7-51ed-44ce-adb6-fde3623cd305" />



## Repository Structure

```
Mnemosky/
├── satellite_trail_detector.py   # Main application (single-file implementation)
├── .gitignore                     # Standard Python gitignore
└── CLAUDE.md                      # This file
```

This is a single-file Python application with no additional modules or packages.

## Tech Stack

- **Python 3** - Primary language
- **OpenCV (cv2)** - Computer vision and video processing
- **NumPy** - Numerical operations and array handling
- **argparse** - CLI argument parsing (stdlib)
- **pathlib** - Path handling (stdlib)

### Installation

```bash
pip install opencv-python numpy
```

## Architecture

### Class Hierarchy

1. **`BaseDetectionAlgorithm`** (ABC) - Abstract base class defining the detection interface
   - `preprocess_frame()` - Frame preprocessing
   - `detect_lines()` - Line detection
   - `classify_trail()` - Trail classification — returns `(trail_type, detection_info)` where `detection_info` is a dict with `bbox`, `angle`, `center`, `length`, `avg_brightness`, `max_brightness`, `line`
   - `detect_trails()` - Main pipeline (can be overridden) — returns `[('satellite', detection_info), ...]`
   - `merge_overlapping_boxes()` - Utility for combining detections (angle-agnostic)

2. **`DefaultDetectionAlgorithm`** (extends BaseDetectionAlgorithm) - Partial implementation
   - Implements `preprocess_frame()`, `detect_lines()`, `detect_point_features()`
   - Note: `classify_trail()` is not implemented (incomplete class)

3. **`SatelliteTrailDetector`** - Main detector class with sensitivity presets
   - Provides `low`, `medium`, `high` sensitivity configurations
   - Contains full two-stage detection pipeline: primary (Canny + Hough) and supplementary (directional matched filter for very dim trails)
   - `classify_trail()` — core classification with star false-positive suppression and spatial spread checks
   - `merge_airplane_detections()` - Angle-aware merge that keeps distinct airplanes separate (crossing paths are not merged)
   - `_detect_dim_lines_matched_filter()` — supplementary dim-trail detection using oriented filter bank
   - `_compute_trail_snr()` — per-trail signal-to-noise ratio via perpendicular flank sampling
   - `_apply_signal_envelope()` — dynamically adapts thresholds from user-marked trail examples
   - Supports custom preprocessing parameters via `preprocessing_params` argument

### Key Functions

- `show_preprocessing_preview()` - Interactive GUI for tuning preprocessing parameters (CLAHE, blur, Canny). Asymmetric layout: large Original panel (left column, ~58% width) + CLAHE/Blur/Edges stacked vertically (right column), sidebar with custom-drawn sliders and trail example thumbnails, full-width frame slider in status bar. Sleek dark-grey theme with fluorescent accent highlights (single window, no external trackbar window).
- `process_video()` - Main video processing pipeline (handles I/O, frame iteration, output, optional YOLO dataset export)
- `main()` - CLI entry point with argument parsing

### Detection Data Structures

Detection results use enriched `detection_info` dicts instead of bare bounding-box tuples:

```python
# detect_trails() returns:
[
    ('satellite', {
        'bbox': (x_min, y_min, x_max, y_max),
        'angle': 45.0,          # degrees 0-180
        'center': (640.0, 360.0),
        'length': 285.0,        # pixels
        'avg_brightness': 12.5,
        'max_brightness': 28,
        'line': (x1, y1, x2, y2),
    }),
    ('airplane', { ... }),
    ('airplane', { ... }),  # Multiple airplanes per frame supported
]
```

## Detection Algorithm

### Pipeline

**Stage 1 — Primary detection (Canny + Hough):**
1. **Preprocessing**: Grayscale conversion → CLAHE enhancement (clip=6.0) → Gaussian blur (default k=5, σ=1.8)
2. **Edge Detection**: Canny edge detection with configurable thresholds (default low=4, high=100)
3. **Morphological Operations**: Dilation (3x) + Erosion (1x), then directional dilation with elongated kernels at 0/45/90/135 degrees to bridge gaps in dim linear features, followed by a cleanup erosion
4. **Line Detection**: Hough line transform (HoughLinesP) with wider gap tolerance for fragmented trails
5. **Classification**: Brightness analysis, color analysis, point feature detection (with star false-positive suppression), contrast-to-background measurement, spatial spread verification

**Stage 2 — Supplementary dim-trail detection (Matched Filter):**
1. **Background subtraction**: Large-kernel median filter removes sky gradients and light pollution
2. **Noise estimation**: Robust MAD-based σ estimation (immune to signal pixels)
3. **Oriented filter bank**: Averaging kernels at 5° steps (36 orientations), kernel length 31px → ~√31 SNR improvement
4. **SNR thresholding**: Pixels with SNR ≥ 2.5 are marked as significant
5. **Line extraction**: Hough on the thresholded map, duplicate suppression vs primary detections
6. **Per-trail SNR confirmation**: Perpendicular flank sampling validates each candidate

### Classification Criteria

| Feature | Satellite | Airplane |
|---------|-----------|----------|
| Trail pattern | Smooth, uniform | Dotted, bright points |
| Brightness | Dim, consistent | Variable, with peaks |
| Color | Monochromatic | May have colored lights |
| Length (1080p) | 100-1200px (medium) | Any length |
| Visual marker | GOLD box | ORANGE box |
| Smoothness | Adaptive threshold (relative + absolute std fallback for dim trails) | N/A |
| Contrast | Configurable per sensitivity (1.08 medium) | N/A |
| Merge strategy | Angle-agnostic box merge | Angle-aware merge (keeps distinct airplanes with >20deg angle difference separate) |

### False Positive Suppression

- **Star false-positive suppression**: When bright spots or high variance are detected along a trail, a spatial spread check verifies that bright pixels span >15% of the trail length. Single stars illuminate <15% and are suppressed, while real airplane navigation lights are distributed along the trail.
- **Minimum peak separation**: In `detect_point_features()`, consecutive brightness peaks must be at least 10% of the trail length apart. Prevents a single star from being counted as multiple airplane navigation lights.
- **Cloud/texture filtering**: High surrounding texture variance (std > 25, mean > 35) or very bright surroundings (mean > 80) reject non-trail features.

### Satellite Detection Paths

The satellite classifier uses multiple detection paths to catch dim and long trails:

1. **Primary**: All 4 criteria met (dim + monochrome + smooth + length range)
2. **Strong 3/4**: At least 3 criteria including smoothness and length
3. **Very dim**: Smooth + below brightness threshold + in length range
4. **Extended — dim+smooth+monochrome**: No max-length cap, catches long trails
5. **Extended — dim+smooth+contrast**: Uses measured trail-to-background contrast ratio
6. **Extended — very dim+smooth**: Below brightness threshold, relaxed monochrome (dim trails have negligible colour)
7. **SNR-based (supplementary only)**: Matched-filter candidates with per-trail SNR ≥ 2.5 and smooth brightness

## YOLO Dataset Export

When `--dataset` is passed, each frame with detections is saved to a YOLO-format dataset alongside the output video:

```
<output_stem>_dataset/
├── data.yaml              # YOLOv8 config (class names, paths)
├── images/
│   ├── <video>_f000123.jpg   # Original (clean) frames — no annotations drawn
│   └── ...
└── labels/
    ├── <video>_f000123.txt   # One line per detection
    └── ...
```

**Label format** (standard YOLO): `<class_id> <x_center> <y_center> <width> <height>` — all coordinates normalized to [0, 1].

| Class ID | Name |
|----------|------|
| 0 | satellite |
| 1 | airplane |

- Images are JPEG quality 95 (clean originals, no bounding boxes drawn)
- Only frames with at least one detection are saved
- `data.yaml` is compatible with Ultralytics `yolo train` out of the box
- Appends to existing dataset directory (allows processing multiple videos into one dataset)
- Post-run summary prints image count, annotation counts by class, and dataset location

## Running the Application

### Basic Usage

```bash
python satellite_trail_detector.py input.mp4 output.mp4
```

### Common Options

```bash
# Sensitivity levels: low, medium (default), high
python satellite_trail_detector.py input.mp4 output.mp4 --sensitivity high

# Freeze duration for detected trails (default: 1.0 second)
python satellite_trail_detector.py input.mp4 output.mp4 --freeze-duration 2.0

# Limit processing duration
python satellite_trail_detector.py input.mp4 output.mp4 --max-duration 30

# Filter detection type
python satellite_trail_detector.py input.mp4 output.mp4 --detect-type satellites
python satellite_trail_detector.py input.mp4 output.mp4 --detect-type airplanes

# Debug modes
python satellite_trail_detector.py input.mp4 output.mp4 --debug       # Side-by-side view
python satellite_trail_detector.py input.mp4 output.mp4 --debug-only  # Debug only

# Hide labels
python satellite_trail_detector.py input.mp4 output.mp4 --no-labels

# Interactive preprocessing preview (tune CLAHE, blur, Canny parameters)
python satellite_trail_detector.py input.mp4 output.mp4 --preview

# Export detections as YOLO ML dataset
python satellite_trail_detector.py input.mp4 output.mp4 --dataset
```

## Code Conventions

### Style Guidelines

- **Docstrings**: All classes and major functions have comprehensive docstrings
- **Comments**: Inline comments explain complex detection logic
- **Type hints**: Not used (consider adding for future development)
- **Line length**: No strict limit, but generally readable

### Parameter Organization

Detection parameters are organized in dictionaries by sensitivity level within `SatelliteTrailDetector.__init__()`:
- `low` - Stricter detection, fewer false positives
- `medium` - Balanced (default)
- `high` - More permissive, catches dimmer and longer trails

Each preset includes `satellite_contrast_min` (configurable contrast-to-background threshold) and widened `satellite_max_length` ranges.

### Key Constants

- **Resolution optimization**: Tuned for 1920x1080 video
- **CLAHE settings**: clipLimit=6.0, tileGridSize=(6, 6)
- **Preview defaults**: blur kernel=5, blur sigma=1.8, canny_high=100
- **Satellite length range (medium)**: 100-1200 pixels (at 1080p)
- **Satellite contrast minimum (medium)**: 1.08 (trail must be 8% brighter than background)
- **Color codes**: GOLD (0, 185, 255 BGR) for satellites, ORANGE (0, 140, 255 BGR) for airplanes
- **Angle merge threshold**: 20 degrees (airplanes with >20deg angle difference stay separate)
- **Star suppression threshold**: Bright pixels must span ≥15% of trail length to count as airplane dots
- **Peak separation**: Brightness peaks must be ≥10% of trail length apart (prevents single-star multi-counting)
- **YOLO image quality**: JPEG quality 95, class IDs 0=satellite 1=airplane

### Preview GUI Theme

The preprocessing preview window uses a custom-drawn dark-grey theme — everything lives in a single window with no external dialogs:
- **Background**: Dark grey (#1E1E1E) with panel cards (#2A2A2A)
- **Text**: Light grey (#D2D2D2) primary, dim grey (#787878) secondary
- **Accents**: Fluorescent green-yellow (#50FFC8 / BGR 200,255,80) for active values, slider fills, and CLAHE label
- **Edges panel**: Cyan-tinted edge overlay instead of raw white edges
- **Sliders**: Custom-drawn in the sidebar — thin accent-coloured track with circular thumb, mouse click+drag interaction via `cv2.setMouseCallback`
- **Layout**: Asymmetric — large Original panel (full left column, ~58% width, full height) + CLAHE/Blur/Edges stacked vertically in right column. Right sidebar with interactive sliders, trail example thumbnails, and controls help. Full-width frame slider in bottom status bar (56px tall) for fine-grained frame navigation.
- **Trail thumbnails**: Completed trail examples shown as photo cutout thumbnails in the sidebar (40px padding, scaled to sidebar width, capped at 120px height) with trail line overlay and stats label — Original panel stays clean.
- **Blur preview**: GaussianBlur applied at display scale on the panel-sized CLAHE image (not downscaled from full-res) so small kernels remain visible.

## Development Workflow

### Testing Changes

1. Run on sample video with `--debug` flag to visualize detection
2. Check classification accuracy in debug output
3. Use `--debug-only` for detailed edge/line visualization
4. Adjust sensitivity or parameters as needed
5. Use `--dataset` to export detections and inspect label files for correctness

### Extending the Algorithm

To create a custom detection algorithm:

```python
class CustomDetectionAlgorithm(BaseDetectionAlgorithm):
    def preprocess_frame(self, frame):
        # Custom preprocessing
        pass

    def detect_lines(self, preprocessed):
        # Custom line detection
        pass

    def classify_trail(self, line, gray_frame, color_frame):
        # Must return (trail_type, detection_info_dict) or (None, None)
        pass
```

## Important Notes for AI Assistants

1. **Single-file architecture**: All code is in `satellite_trail_detector.py`. Do not create additional modules unless specifically requested.

2. **Resolution dependency**: Parameters are optimized for 1920x1080. When modifying thresholds, consider scaling for different resolutions.

3. **Video codec handling**: The code includes fallback logic for video codecs (MPEG-4 → H.264 variants → system default). Maintain this pattern.

4. **Debug modes**: Preserve the `--debug` and `--debug-only` functionality when making changes.

5. **Classification balance**: Satellites use multiple graduated detection paths (primary + extended + SNR-based). Airplanes only need characteristic point features. The extended paths allow detection of very dim and very long satellite trails that would have been missed by the strict primary criteria.

6. **No external dependencies beyond OpenCV/NumPy**: Keep the dependency footprint minimal.

7. **Boundary checking**: ROI operations include boundary checks. Maintain these when working with image regions.

8. **Detection data format**: `detect_trails()` returns `(trail_type, detection_info)` tuples where `detection_info` is a dict with `bbox` and metadata keys. Do not assume bare bbox tuples — always access `detection_info['bbox']`.

9. **Multi-airplane support**: The `merge_airplane_detections()` method uses angle-aware merging. Two airplane detections only merge if their bounding boxes overlap AND trail angles are within 20 degrees. This prevents crossing flight paths from being collapsed into a single detection.

10. **Preview GUI theme**: The preview window uses a custom dark-grey/fluorescent-accent theme drawn entirely with OpenCV primitives in a single window. Sliders are custom-drawn (not native trackbars) with mouse callback interaction. Trail examples are shown as photo cutout thumbnails in the sidebar (not drawn on the Original panel). Blur is applied at display scale. Maintain the sleek minimal aesthetic when modifying.

11. **Two-stage detection pipeline**: The primary pipeline (Canny + Hough) is supplemented by a matched-filter stage that catches trails too dim for edge detection. The `supplementary=True` flag relaxes contrast thresholds and enables an SNR-based detection path. Do not remove either stage.

12. **Star false-positive suppression**: `classify_trail()` includes spatial spread checks that suppress `has_bright_spots` and `has_high_variance` when bright pixels span <15% of trail length. `detect_point_features()` uses minimum peak separation to prevent single stars from counting as multiple airplane lights. Preserve both mechanisms.

13. **Signal envelope**: When users mark trail examples in the preview, `_compute_signal_envelope()` measures brightness, contrast, length, and angle ranges. `_apply_signal_envelope()` dynamically widens detection thresholds to match. The envelope flows from `show_preprocessing_preview()` → `main()` → `process_video()` → `SatelliteTrailDetector.__init__()`.

14. **YOLO dataset export**: The `--dataset` flag saves clean (unannotated) original frames and normalized bounding-box labels. Class IDs are `0=satellite`, `1=airplane`. Export happens inside the detection loop in `process_video()`. `data.yaml` is written after the main loop. Do not draw annotations on exported images.

## Common Tasks

### Adding a new sensitivity preset

Add to `SatelliteTrailDetector.__init__()`:
```python
presets['custom'] = {
    'canny_low': ...,
    'canny_high': ...,
    'satellite_min_length': ...,
    'satellite_max_length': ...,
    'satellite_contrast_min': ...,
    # ... other parameters
}
```

### Modifying classification logic

Edit `SatelliteTrailDetector.classify_trail()` - this is the core decision function. Returns `(trail_type, detection_info_dict)`.

### Adding new CLI arguments

Add to the `main()` function's argument parser, then handle in `process_video()`.

### Changing visual output

- Box styling: `draw_dotted_rectangle()`
- Labels: `draw_highlight()`
- Debug panels: `create_detection_debug_panel()`
- Preview GUI: `show_preprocessing_preview()` — uses custom dark theme with `_fill_rect`, `_put_text`, `_draw_tag` helpers

## File Locations Quick Reference

| Component | Location |
|-----------|----------|
| Preprocessing preview | `satellite_trail_detector.py:show_preprocessing_preview()` (line ~40) |
| Abstract interface | `satellite_trail_detector.py:BaseDetectionAlgorithm` (line ~896) |
| Partial implementation | `satellite_trail_detector.py:DefaultDetectionAlgorithm` (line ~1059) |
| Main detector class | `satellite_trail_detector.py:SatelliteTrailDetector` (line ~1219) |
| Sensitivity presets | `satellite_trail_detector.py:SatelliteTrailDetector.__init__()` (line ~1230) |
| Signal envelope adaptation | `satellite_trail_detector.py:SatelliteTrailDetector._apply_signal_envelope()` (line ~1323) |
| Matched filter detection | `satellite_trail_detector.py:SatelliteTrailDetector._detect_dim_lines_matched_filter()` (line ~1513) |
| Trail SNR computation | `satellite_trail_detector.py:SatelliteTrailDetector._compute_trail_snr()` (line ~1649) |
| Point feature detection | `satellite_trail_detector.py:SatelliteTrailDetector.detect_point_features()` (line ~1730) |
| Classification logic | `satellite_trail_detector.py:SatelliteTrailDetector.classify_trail()` (line ~1816) |
| Angle-aware airplane merge | `satellite_trail_detector.py:SatelliteTrailDetector.merge_airplane_detections()` (line ~2222) |
| Two-stage detect_trails | `satellite_trail_detector.py:SatelliteTrailDetector.detect_trails()` (line ~2335) |
| Video processing | `satellite_trail_detector.py:process_video()` (line ~2780) |
| YOLO dataset export | `satellite_trail_detector.py:process_video()` — dataset setup ~2917, per-frame export ~2943, data.yaml ~3047 |
| Main entry point | `satellite_trail_detector.py:main()` (line ~3137) |
