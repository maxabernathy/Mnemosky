# CLAUDE.md - AI Assistant Guide for Mnemosky

## Project Overview

**Mnemosky** is a satellite and airplane trail detector for MP4 videos. It uses classical computer vision techniques to identify and classify celestial trails in night sky footage, distinguishing between satellites (smooth, uniform trails) and airplanes (dotted patterns with bright navigation lights).

Preprocessing adjustments

<img width="2163" height="1269" alt="image" src="https://github.com/user-attachments/assets/0fb875ba-94b2-476c-8b0d-14f23285afcc" />

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
   - Contains full detection pipeline including `classify_trail()` logic
   - `merge_airplane_detections()` - Angle-aware merge that keeps distinct airplanes separate (crossing paths are not merged)
   - Supports custom preprocessing parameters via `preprocessing_params` argument

### Key Functions

- `show_preprocessing_preview()` - Interactive GUI for tuning preprocessing parameters (CLAHE, blur, Canny). Sleek dark-grey theme with fluorescent accent highlights, 2x2 panel grid with sidebar containing custom-drawn sliders (single window, no external trackbar window).
- `process_video()` - Main video processing pipeline (handles I/O, frame iteration, output)
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

1. **Preprocessing**: Grayscale conversion → CLAHE enhancement (clip=6.0) → Gaussian blur
2. **Edge Detection**: Canny edge detection with configurable thresholds
3. **Morphological Operations**: Dilation (3x) + Erosion (1x), then directional dilation with elongated kernels at 0/45/90/135 degrees to bridge gaps in dim linear features, followed by a cleanup erosion
4. **Line Detection**: Hough line transform (HoughLinesP) with wider gap tolerance for fragmented trails
5. **Classification**: Brightness analysis, color analysis, point feature detection, contrast-to-background measurement

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

### Satellite Detection Paths

The satellite classifier uses multiple detection paths to catch dim and long trails:

1. **Primary**: All 4 criteria met (dim + monochrome + smooth + length range)
2. **Strong 3/4**: At least 3 criteria including smoothness and length
3. **Very dim**: Smooth + below brightness threshold + in length range
4. **Extended — dim+smooth+monochrome**: No max-length cap, catches long trails
5. **Extended — dim+smooth+contrast**: Uses measured trail-to-background contrast ratio
6. **Extended — very dim+smooth**: Below brightness threshold, relaxed monochrome (dim trails have negligible colour)

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
- **Satellite length range (medium)**: 100-1200 pixels (at 1080p)
- **Satellite contrast minimum (medium)**: 1.08 (trail must be 8% brighter than background)
- **Color codes**: GOLD (0, 185, 255 BGR) for satellites, ORANGE (0, 140, 255 BGR) for airplanes
- **Angle merge threshold**: 20 degrees (airplanes with >20deg angle difference stay separate)

### Preview GUI Theme

The preprocessing preview window uses a custom-drawn dark-grey theme — everything lives in a single window with no external dialogs:
- **Background**: Dark grey (#1E1E1E) with panel cards (#2A2A2A)
- **Text**: Light grey (#D2D2D2) primary, dim grey (#787878) secondary
- **Accents**: Fluorescent green-yellow (#50FFC8 / BGR 200,255,80) for active values, slider fills, and CLAHE label
- **Edges panel**: Cyan-tinted edge overlay instead of raw white edges
- **Sliders**: Custom-drawn in the sidebar — thin accent-coloured track with circular thumb, mouse click+drag interaction via `cv2.setMouseCallback`. Coordinate mapping handles window scaling.
- **Layout**: 2x2 panel grid + right sidebar with interactive sliders, controls help + bottom status bar with frame counter

## Development Workflow

### Testing Changes

1. Run on sample video with `--debug` flag to visualize detection
2. Check classification accuracy in debug output
3. Use `--debug-only` for detailed edge/line visualization
4. Adjust sensitivity or parameters as needed

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

3. **Video codec handling**: The code includes fallback logic for video codecs (H.264 → MPEG-4 → default). Maintain this pattern.

4. **Debug modes**: Preserve the `--debug` and `--debug-only` functionality when making changes.

5. **Classification balance**: Satellites use multiple graduated detection paths (primary + extended). Airplanes only need characteristic point features. The extended paths allow detection of very dim and very long satellite trails that would have been missed by the strict primary criteria.

6. **No external dependencies beyond OpenCV/NumPy**: Keep the dependency footprint minimal.

7. **Boundary checking**: ROI operations include boundary checks. Maintain these when working with image regions.

8. **Detection data format**: `detect_trails()` returns `(trail_type, detection_info)` tuples where `detection_info` is a dict with `bbox` and metadata keys. Do not assume bare bbox tuples — always access `detection_info['bbox']`.

9. **Multi-airplane support**: The `merge_airplane_detections()` method uses angle-aware merging. Two airplane detections only merge if their bounding boxes overlap AND trail angles are within 20 degrees. This prevents crossing flight paths from being collapsed into a single detection.

10. **Preview GUI theme**: The preview window uses a custom dark-grey/fluorescent-accent theme drawn entirely with OpenCV primitives in a single window. Sliders are custom-drawn (not native trackbars) with mouse callback interaction. Maintain the sleek minimal aesthetic when modifying.

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
| Abstract interface | `satellite_trail_detector.py:BaseDetectionAlgorithm` (line ~420) |
| Partial implementation | `satellite_trail_detector.py:DefaultDetectionAlgorithm` (line ~583) |
| Main detector class | `satellite_trail_detector.py:SatelliteTrailDetector` (line ~730) |
| Sensitivity presets | `satellite_trail_detector.py:SatelliteTrailDetector.__init__()` (line ~741) |
| Classification logic | `satellite_trail_detector.py:SatelliteTrailDetector.classify_trail()` (line ~995) |
| Angle-aware airplane merge | `satellite_trail_detector.py:SatelliteTrailDetector.merge_airplane_detections()` (line ~1352) |
| Video processing | `satellite_trail_detector.py:process_video()` (line ~1883) |
| Main entry point | `satellite_trail_detector.py:main()` (line ~2177) |
