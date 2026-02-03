# CLAUDE.md - AI Assistant Guide for Mnemosky

## Project Overview

**Mnemosky** is a satellite and airplane trail detector for MP4 videos. It uses classical computer vision techniques to identify and classify celestial trails in night sky footage, distinguishing between satellites (smooth, uniform trails) and airplanes (dotted patterns with bright navigation lights).

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
   - `classify_trail()` - Trail classification
   - `detect_trails()` - Main pipeline (can be overridden)
   - `merge_overlapping_boxes()` - Utility for combining detections

2. **`DefaultDetectionAlgorithm`** (extends BaseDetectionAlgorithm) - Partial implementation
   - Implements `preprocess_frame()`, `detect_lines()`, `detect_point_features()`
   - Note: `classify_trail()` is not implemented (incomplete class)

3. **`SatelliteTrailDetector`** - Main detector class with sensitivity presets
   - Provides `low`, `medium`, `high` sensitivity configurations
   - Contains full detection pipeline including `classify_trail()` logic
   - Supports custom preprocessing parameters via `preprocessing_params` argument

### Key Functions

- `show_preprocessing_preview()` - Interactive GUI for tuning preprocessing parameters (CLAHE, blur, Canny)
- `process_video()` - Main video processing pipeline (handles I/O, frame iteration, output)
- `main()` - CLI entry point with argument parsing

## Detection Algorithm

### Pipeline

1. **Preprocessing**: Grayscale conversion → CLAHE enhancement → Gaussian blur
2. **Edge Detection**: Canny edge detection with configurable thresholds
3. **Morphological Operations**: Dilation (3x) + Erosion (1x) to connect trail fragments
4. **Line Detection**: Hough line transform (HoughLinesP)
5. **Classification**: Brightness analysis, color analysis, point feature detection

### Classification Criteria

| Feature | Satellite | Airplane |
|---------|-----------|----------|
| Trail pattern | Smooth, uniform | Dotted, bright points |
| Brightness | Dim, consistent | Variable, with peaks |
| Color | Monochromatic | May have colored lights |
| Length (1080p) | 180-300px | Any length |
| Visual marker | GOLD box | ORANGE box |

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

Detection parameters are organized in dictionaries by sensitivity level:
- `low_sensitivity_params` - Stricter detection, fewer false positives
- `medium_sensitivity_params` - Balanced (default)
- `high_sensitivity_params` - More permissive, more detections

### Key Constants

- **Resolution optimization**: Tuned for 1920x1080 video
- **CLAHE settings**: clipLimit=4.0, tileGridSize=(6, 6)
- **Satellite length range**: 180-300 pixels (at 1080p)
- **Color codes**: GOLD (255, 215, 0) for satellites, ORANGE (255, 165, 0) for airplanes

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
        # Custom classification logic
        pass
```

## Important Notes for AI Assistants

1. **Single-file architecture**: All code is in `satellite_trail_detector.py`. Do not create additional modules unless specifically requested.

2. **Resolution dependency**: Parameters are optimized for 1920x1080. When modifying thresholds, consider scaling for different resolutions.

3. **Video codec handling**: The code includes fallback logic for video codecs (H.264 → MPEG-4 → default). Maintain this pattern.

4. **Debug modes**: Preserve the `--debug` and `--debug-only` functionality when making changes.

5. **Classification balance**: The algorithm prioritizes avoiding false positives. Satellites require ALL criteria to be met; airplanes only need characteristic point features.

6. **No external dependencies beyond OpenCV/NumPy**: Keep the dependency footprint minimal.

7. **Boundary checking**: ROI operations include boundary checks. Maintain these when working with image regions.

## Common Tasks

### Adding a new sensitivity preset

Add to `SatelliteTrailDetector.__init__()`:
```python
self.sensitivity_presets['custom'] = {
    'canny_low': ...,
    'canny_high': ...,
    # ... other parameters
}
```

### Modifying classification logic

Edit `DefaultDetectionAlgorithm.classify_trail()` - this is the core decision function.

### Adding new CLI arguments

Add to the `main()` function's argument parser, then handle in `process_video()`.

### Changing visual output

- Box styling: `draw_dotted_rectangle()`
- Labels: `draw_highlight()`
- Debug panels: `create_detection_debug_panel()`

## File Locations Quick Reference

| Component | Location |
|-----------|----------|
| Preprocessing preview | `satellite_trail_detector.py:show_preprocessing_preview()` (line ~40) |
| Abstract interface | `satellite_trail_detector.py:BaseDetectionAlgorithm` (line ~338) |
| Partial implementation | `satellite_trail_detector.py:DefaultDetectionAlgorithm` (line ~496) |
| Main detector class | `satellite_trail_detector.py:SatelliteTrailDetector` (line ~643) |
| Sensitivity presets | `satellite_trail_detector.py:SatelliteTrailDetector.__init__()` (line ~654) |
| Classification logic | `satellite_trail_detector.py:SatelliteTrailDetector.classify_trail()` (line ~878) |
| Video processing | `satellite_trail_detector.py:process_video()` (line ~1535) |
| Main entry point | `satellite_trail_detector.py:main()` (line ~1825) |
