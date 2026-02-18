# CLAUDE.md - AI Assistant Guide for Mnemosky

## Project Overview

**Mnemosky** is a satellite and airplane trail detector for MP4 videos. It uses classical computer vision techniques and optionally neural network models to identify and classify celestial trails in night sky footage, distinguishing between satellites (smooth, uniform trails) and airplanes (dotted patterns with bright navigation lights). Detected trails can optionally be exported as a YOLO-format ML dataset for training object detection models, which can then be fed back into the NN detection pipeline for real-time inference.

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
├── hitl_architecture.md           # HITL RL system design document
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
# Optional: scipy improves Radon NMS quality (falls back to cv2.dilate if absent)
pip install scipy
# Optional: neural network backends for --algorithm nn (auto-installed on first use)
pip install ultralytics       # Primary NN backend (YOLOv8/v11)
pip install onnxruntime       # Alternative ONNX backend
# cv2.dnn is included with opencv-python (no extra install needed)
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

3. **`_NNBackend`** - Unified neural network inference wrapper
   - Supports three backends: `ultralytics` (YOLOv8/v11), `cv2dnn` (OpenCV DNN), `onnxruntime` (ONNX Runtime)
   - Lazy import — backend package only loaded when selected; auto-install prompt for pip packages
   - `predict(frame)` → list of `{bbox, class_id, class_name, confidence}` dicts
   - `_parse_yolo_output()` handles both YOLOv5 (5+C) and YOLOv8 (4+C) output layouts
   - GPU/CPU device selection per backend; respects `--no-gpu`

4. **`SatelliteTrailDetector`** - Main detector class with sensitivity presets
   - Provides `low`, `medium`, `high` sensitivity configurations
   - Contains full two-stage detection pipeline: primary (Canny + Hough) and supplementary (directional matched filter for very dim trails)
   - `classify_trail()` — core classification with star false-positive suppression and spatial spread checks
   - `merge_airplane_detections()` - Angle-aware merge that keeps distinct airplanes separate (crossing paths are not merged)
   - `_detect_dim_lines_matched_filter()` — supplementary dim-trail detection using oriented filter bank
   - `_compute_trail_snr()` — per-trail signal-to-noise ratio via perpendicular flank sampling
   - `_apply_signal_envelope()` — dynamically adapts thresholds from user-marked trail examples
   - Supports custom preprocessing parameters via `preprocessing_params` argument

5. **`NeuralNetDetector`** (extends SatelliteTrailDetector) - Neural network model-based detector
   - Uses `_NNBackend` for inference (ultralytics, cv2dnn, or onnxruntime)
   - `detect_trails()` runs model → maps class IDs to trail types → computes post-hoc detection_info
   - `_bbox_to_detection_info()` synthesizes angle, brightness, line, contrast from model bboxes
   - `_merge_nn_classical()` for hybrid mode (NN + classical pipeline results merged)
   - Deferred backend loading: `_NNBackend` created on first `detect_trails()` call (pickle-safe for workers)
   - Configurable class mapping: `{'satellite': [0], 'airplane': [1]}` (multi-class supported)
   - Adds `nn_confidence` field to detection_info dicts

### HITL (Human-in-the-Loop) Classes

7. **`AnnotationDatabase`** - COCO-compatible annotation database with correction tracking
   - Stores images, annotations (with `mnemosky_ext` metadata), missed annotations, corrections, sessions
   - `add_detection()` converts internal bbox `(x_min, y_min, x_max, y_max)` to COCO `[x, y, w, h]`
   - `add_missed()` for user-drawn false negatives
   - `record_correction()` with full audit trail (accept/reject/reclassify/adjust_bbox/add_missed)
   - `get_calibration_set()` builds `(detection_meta, true_label, detector_label)` tuples for learning
   - `get_frames_needing_review()` sorts by minimum detection confidence (active learning)
   - `export_coco()` strips Mnemosky extensions for pure COCO format
   - `undo_last_correction()` reverts the last action

8. **`ParameterAdapter`** - Two-tier parameter learning from human corrections
   - **Tier 1 (EMA)**: Each correction nudges relevant parameters via exponential moving average with decaying learning rate (`base_lr=0.3`, `decay_rate=0.1`)
   - **Tier 2 (batch)**: Coordinate-wise golden section search over 13 optimizable parameters, minimizing weighted FP/FN/misclassification loss
   - `CORRECTION_RULES` maps `(action, trail_type)` to parameter adjustments with diagnostic lambdas
   - `PARAMETER_SAFETY_BOUNDS` enforces hard min/max per parameter to prevent drift
   - `compute_confidence()` — pseudo-confidence score for active learning prioritization
   - `save_profile()` / `load_profile()` persist learned parameters to `~/.mnemosky/learned_params.json`

9. **`ReviewUI`** - Interactive OpenCV review window for correcting detections
   - Single window with dark-grey/fluorescent-accent theme (matches preview GUI)
   - Main frame view (left) + 280px sidebar with detection cards, session stats, controls hint
   - 56px bottom status bar with frame slider and learn/save buttons
   - Keyboard shortcuts: A(ccept), R(eject), S(atellite), P(lane), M(ark missed), Z(undo), L(earn), Tab (cycle), N(ext unreviewed), Space (accept all), X (reject all), F(ull frame), H(elp), Q(uit)
   - Mouse: click detection boxes, click sidebar cards, drag frame slider, draw bbox in mark mode
   - Active learning: frames sorted by minimum detection confidence, low-confidence detections pulsed with amber star
   - Feeds corrections to `AnnotationDatabase` and applies Tier 1 learning via `ParameterAdapter`

10. **`ProcessingWindow`** - Live processing dashboard shown during video processing
   - Same dark-grey/fluorescent-accent theme as preview windows
   - Layout: LIVE FRAME (current frame thumbnail with detection boxes) | TRAIL MAP (2-D scatter of all detected trail centres over normalised frame space) | DETECTION TIMELINE (horizontal strip with stacked bars, waveform overlay, progress playhead) | STATUS BAR (progress ring, processing FPS, detection counts, ETA)
   - Throttled display (~20 fps redraw) to keep CPU overhead negligible; always redraws on detection events
   - Press Q/ESC to abort processing early
   - Automatically enabled when `--preview` is used (seamless transition) or via `--show-processing`

### Key Functions

- `show_preprocessing_preview()` - Interactive GUI for tuning preprocessing parameters (CLAHE, blur, Canny). Asymmetric layout: large Original panel (left column, ~58% width) + CLAHE/Blur/Edges stacked vertically (right column), sidebar with custom-drawn sliders and trail example thumbnails, full-width frame slider in status bar. Sleek dark-grey theme with fluorescent accent highlights (single window, no external trackbar window).
- `show_radon_preview()` - Interactive GUI for tuning the Radon detection pipeline (Radon SNR, PCF ratio, star mask sigma, LSD significance, PCF kernel, min length). Same dark-grey/fluorescent theme. Four diagnostic panels: Residual (star-cleaned), Sinogram (SNR heatmap with peaks), LSD Lines, and Detections (PCF-confirmed vs rejected). Activated by `--preview` with `--algorithm radon`.
- `show_nn_preview()` - Interactive GUI for tuning neural network detection parameters (confidence, NMS IoU). Same dark-grey/fluorescent theme. Main frame panel with detection overlays, sidebar with model info card, sliders, class mapping, confidence bars, inference FPS. Activated by `--preview` with `--algorithm nn`.
- `load_config()` / `save_config()` - Application-wide config persistence to `~/.mnemosky/config.json`. Stores all algorithm parameters (default, radon, nn), model paths, backend choice. Deep-merges user config with defaults.
- `_worker_init()` / `_worker_detect()` - Multiprocessing worker functions for parallel frame detection
- `process_video()` - Main video processing pipeline (handles I/O, frame iteration, output, optional YOLO dataset export, parallel dispatch)
- `main()` - CLI entry point with argument parsing

### Parallelism Architecture

Detection is stateless per-frame (`detect_trails()` has no frame-to-frame mutation), so frames are distributed across a `multiprocessing.Pool` of N worker processes:

```
Main Process (sequential)          Worker Pool (N processes, parallel)
────────────────────────           ─────────────────────────────────
Read frame                    ──>  Worker 1: detect_trails(frame_A)
Feed temporal buffer               Worker 2: detect_trails(frame_B)
Copy temporal context               Worker 3: detect_trails(frame_C)
Submit to pool                      Worker 4: detect_trails(frame_D)
                              <──
Collect results (in order)
Apply freeze overlays
Write output frame
```

- **Main process**: sequential I/O, temporal buffer feeding, freeze overlay, output writing
- **Worker pool**: parallel detection via `apply_async()` with prefetch depth `workers * 2`
- **Each worker**: has its own detector instance (created via `_worker_init()` pool initializer)
- **GPU**: optional CUDA acceleration for `filter2D` (matched filter) and `warpAffine` (Radon)
- **`_frame_results()` generator**: nested inside `process_video()`, abstracts sequential vs parallel detection so the post-processing loop is shared
- **Default workers**: `min(cpu_count - 1, 8)`, auto-detected. `--workers 0` for sequential.

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

### Alternative Pipeline: Radon + LSD + PCF (`--algorithm radon`)

**Stage 1 — LSD detection (a-contrario line segments):**
1. **Downsampling**: Frame scaled to max 1920px wide for performance
2. **CLAHE enhancement**: clipLimit=8.0 for dim feature visibility
3. **LSD**: OpenCV `createLineSegmentDetector(LSD_REFINE_ADV)` with NFA-based significance
4. **Classification**: Same `classify_trail()` as default pipeline

**Stage 2 — Radon transform streak detection:**
1. **Background subtraction**: Temporal (if buffer available) or spatial median
2. **Dual-threshold star masking**: 5-sigma stars identified and replaced with background
3. **Aggressive downsampling**: Frame capped at ~250k pixels total (e.g., 500x500 for 1080p)
4. **Radon transform**: Image projected at 90 angles (2-degree steps) via `cv2.warpAffine`
5. **SNR normalisation**: Sinogram divided by expected noise-per-projection
6. **Baseline removal**: Gaussian blur on sinogram columns to remove trends
7. **Non-maximum suppression**: Peak detection in SNR sinogram
8. **Line reconstruction**: Peaks converted to image-space line segments

**Stage 3 — Perpendicular cross filtering (PCF):**
1. For each Radon candidate, sample brightness parallel and perpendicular to the trail
2. Real streaks have asymmetric cross-sections; stars/noise are symmetric
3. Candidates with perpendicular/parallel ratio below threshold are rejected

**Ground truth calibration (optional):**
- Loads PNG patches from `--groundtruth` directory
- Measures PSF width (FWHM of perpendicular cross-section)
- Measures trail brightness, contrast, angle, and length distributions
- Adapts Radon SNR threshold, PCF kernel length, and classification parameters

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

## ML Dataset Export

When `--dataset` is passed, detections are exported as a training-ready ML dataset with automatic train/val/test splitting, temporal deduplication, and negative sample generation.

### Dataset Formats (`--dataset-format`)

| Format | Flag | Label format | Use case |
|--------|------|-------------|----------|
| **AABB** (default) | `--dataset-format aabb` | `class_id xc yc w h` | Standard YOLO detection |
| **OBB** | `--dataset-format obb` | `class_id x1 y1 x2 y2 x3 y3 x4 y4` | YOLO OBB — ideal for thin trails |
| **Segment** | `--dataset-format segment` | `class_id x1 y1 ... x4 y4` | YOLO instance segmentation |
| **COCO** | `--dataset-format coco` | COCO JSON per split | Detectron2, MMDetection |

All coordinates are normalized to [0, 1]. Class IDs: `0=satellite`, `1=airplane`.

### Directory Structure

```
<output_stem>_dataset/
├── data.yaml                  # YOLOv8 config (task, split paths, class names, metadata)
├── train/
│   ├── images/
│   │   └── <video>_f000123.jpg
│   ├── labels/
│   │   └── <video>_f000123.txt
│   └── annotations.json       # (COCO format only)
├── val/
│   ├── images/ ...
│   └── labels/ ...
└── test/
    ├── images/ ...
    └── labels/ ...
```

### Key Features

- **Train/val/test splitting** (`--dataset-split 0.7 0.2 0.1`): Temporal-episode-based splitting prevents data leakage from consecutive near-identical frames. All frames from a single continuous trail sighting go into the same split.
- **Temporal deduplication** (`--dataset-dedup 5`): Perceptual-hash-based near-duplicate filtering removes redundant frames from the same scene.
- **Frame skip** (`--dataset-skip N`): Exports every Nth frame with detections. Default: auto (fps/2, i.e., one frame per 0.5 seconds).
- **Negative samples** (`--dataset-negatives 0.2`): Frames without detections are exported as hard negatives (empty label files) to reduce false positive rates during training.
- **Oriented bounding boxes**: OBB format uses the original trail line endpoints and perpendicular width to compute tight rotated boxes. Eliminates ~50% wasted area vs axis-aligned boxes on diagonal trails.
- **HITL-verified export** (`--dataset-from-annotations`): Exports only human-confirmed detections from an annotation JSON for highest label quality.
- **Statistics report**: Prints class distribution, split sizes, bbox size stats, and health warnings (class imbalance, high annotation density).
- **Configurable images**: `--dataset-image-format jpg|png`, `--dataset-image-quality 95`

### DatasetExporter Class

Encapsulates all dataset export logic: frame filtering (skip + dedup), label writing (all 4 formats), temporal episode grouping, train/val/test splitting, file reorganization, `data.yaml` generation, COCO JSON export, and statistics.

### Utility Functions

- `_compute_obb_corners(x1, y1, x2, y2, half_width)` — 4 oriented box corners from line endpoints
- `_trail_to_polygon(x1, y1, x2, y2, half_width, frame_w, frame_h)` — Normalized polygon for YOLO segment format
- `_compute_phash(frame_gray)` — 64-bit perceptual hash (resize to 8×8 + median threshold)
- `_hamming_distance(h1, h2)` — Hamming distance between two perceptual hashes

## HITL Annotation Database

When `--review` is used, detections are stored in a COCO-compatible JSON annotation file (`<output>.json`):

```json
{
    "info": { "description": "Mnemosky HITL annotation database", "version": "1.0" },
    "categories": [{"id": 0, "name": "satellite"}, {"id": 1, "name": "airplane"}],
    "images": [{"id": 1, "frame_index": 123, "video_source": "input.mp4", ...}],
    "annotations": [{
        "id": 1, "image_id": 1, "category_id": 0,
        "bbox": [100, 200, 350, 40],
        "mnemosky_ext": {
            "source": "detector", "status": "confirmed", "confidence": 0.82,
            "detection_meta": {"angle": 135, "length": 352, "contrast_ratio": 1.12, ...},
            "parameters_snapshot": {"satellite_contrast_min": 1.03, ...}
        }
    }],
    "missed_annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [...]}],
    "corrections": [{"action": "accept", "annotation_id": 1, "timestamp": "..."}],
    "sessions": [{"parameters_before": {...}, "parameters_after": {...}}],
    "learned_parameters": {"current": {...}, "update_count": 12}
}
```

- `annotations[].mnemosky_ext.status`: `"pending"` | `"confirmed"` | `"rejected"`
- `annotations[].bbox`: COCO format `[x, y, width, height]` (differs from internal `(x_min, y_min, x_max, y_max)`)
- `corrections[].action`: `"accept"` | `"reject"` | `"reclassify"` | `"adjust_bbox"` | `"add_missed"`
- Learned parameters also persist in `~/.mnemosky/learned_params.json` (profile-based, survives across videos)

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

# Advanced Radon+LSD+PCF algorithm (catches dimmer trails)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm radon

# Radon algorithm with ground truth calibration
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm radon --groundtruth ./groundtruth

# Neural network algorithm (requires a trained model)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt

# NN with ONNX model via OpenCV DNN backend (no extra deps)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.onnx --nn-backend cv2dnn

# NN with custom confidence/NMS thresholds
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt --confidence 0.5 --nms-iou 0.3

# NN hybrid mode: merge NN + classical pipeline for maximum recall
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt --nn-hybrid

# NN with custom class mapping (e.g. multi-class model)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt --nn-class-map '{"satellite": [0, 2], "airplane": [1, 3]}'

# NN preview (interactive tuning of confidence/NMS before processing)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt --preview

# Save current parameters to config file for future runs
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm nn --model trail_detector.pt --save-config

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

# Radon pipeline debug preview (tune Radon SNR, PCF, star mask, LSD parameters)
python satellite_trail_detector.py input.mp4 output.mp4 --algorithm radon --preview

# Export ML dataset (default: AABB with 70/20/10 split, dedup, negatives)
python satellite_trail_detector.py input.mp4 output.mp4 --dataset

# Export with oriented bounding boxes (ideal for thin trails)
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-format obb

# Export as COCO JSON for Detectron2/MMDetection
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-format coco

# Export as instance segmentation polygons
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-format segment

# Custom split ratios (train/val/test)
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-split 0.8 0.1 0.1

# Disable dedup and negatives for raw export
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-dedup 0 --dataset-negatives 0

# Export lossless PNG images
python satellite_trail_detector.py input.mp4 output.mp4 --dataset --dataset-image-format png

# Export from HITL-verified annotations (highest quality labels)
python satellite_trail_detector.py input.mp4 output.mp4 --dataset-from-annotations output.json

# Parallel processing (default: auto-detected workers, capped at 8)
python satellite_trail_detector.py input.mp4 output.mp4              # auto parallel (default)
python satellite_trail_detector.py input.mp4 output.mp4 --workers 4  # 4 workers
python satellite_trail_detector.py input.mp4 output.mp4 --workers 0  # sequential (no multiprocessing)

# Disable CUDA GPU acceleration (CPU only)
python satellite_trail_detector.py input.mp4 output.mp4 --no-gpu

# Live processing dashboard (frame preview, trail map, timeline, progress ring)
python satellite_trail_detector.py input.mp4 output.mp4 --show-processing

# Preview + processing dashboard (seamless transition: --preview auto-enables dashboard)
python satellite_trail_detector.py input.mp4 output.mp4 --preview

# HITL review mode: process video, then open interactive review UI
python satellite_trail_detector.py input.mp4 output.mp4 --review

# Review existing annotations without re-processing
python satellite_trail_detector.py input.mp4 output.mp4 --review-only --annotations output.json

# Use a named learned parameter profile
python satellite_trail_detector.py input.mp4 output.mp4 --review --hitl-profile my_camera

# Disable auto-accept (review all detections manually)
python satellite_trail_detector.py input.mp4 output.mp4 --review --auto-accept 1.0

# Review mode without parameter learning (corrections saved, params unchanged)
python satellite_trail_detector.py input.mp4 output.mp4 --review --no-learn
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

6. **No external dependencies beyond OpenCV/NumPy**: Keep the dependency footprint minimal. `multiprocessing`, `os`, `collections.deque` are stdlib.

7. **Boundary checking**: ROI operations include boundary checks. Maintain these when working with image regions.

8. **Detection data format**: `detect_trails()` returns `(trail_type, detection_info)` tuples where `detection_info` is a dict with `bbox` and metadata keys. Do not assume bare bbox tuples — always access `detection_info['bbox']`.

9. **Multi-airplane support**: The `merge_airplane_detections()` method uses angle-aware merging. Two airplane detections only merge if their bounding boxes overlap AND trail angles are within 20 degrees. This prevents crossing flight paths from being collapsed into a single detection.

10. **Preview GUI theme**: The preview window uses a custom dark-grey/fluorescent-accent theme drawn entirely with OpenCV primitives in a single window. Sliders are custom-drawn (not native trackbars) with mouse callback interaction. Trail examples are shown as photo cutout thumbnails in the sidebar (not drawn on the Original panel). Blur is applied at display scale. Maintain the sleek minimal aesthetic when modifying.

11. **Two-stage detection pipeline**: The primary pipeline (Canny + Hough) is supplemented by a matched-filter stage that catches trails too dim for edge detection. The `supplementary=True` flag relaxes contrast thresholds and enables an SNR-based detection path. Do not remove either stage.

12. **Star false-positive suppression**: `classify_trail()` includes spatial spread checks that suppress `has_bright_spots` and `has_high_variance` when bright pixels span <15% of trail length. `detect_point_features()` uses minimum peak separation to prevent single stars from counting as multiple airplane lights. Preserve both mechanisms.

13. **Signal envelope**: When users mark trail examples in the preview, `_compute_signal_envelope()` measures brightness, contrast, length, and angle ranges. `_apply_signal_envelope()` dynamically widens detection thresholds to match. The envelope flows from `show_preprocessing_preview()` → `main()` → `process_video()` → `SatelliteTrailDetector.__init__()`.

14. **ML dataset export**: The `--dataset` flag exports training-ready datasets via the `DatasetExporter` class. Supports 4 formats: AABB (standard YOLO), OBB (oriented bounding boxes using trail line endpoints), segment (instance segmentation polygons), and COCO JSON. By default, applies temporal-episode-based train/val/test splitting, perceptual-hash deduplication, frame skip, and negative sample generation. Class IDs are `0=satellite`, `1=airplane`. Export is driven by `DatasetExporter` methods called from the detection loop in `process_video()`. `data.yaml` and COCO JSON are written during `DatasetExporter.finalize()` after the main loop. Do not draw annotations on exported images. `--dataset-from-annotations` exports from HITL-verified annotations for highest label quality.

15. **RadonStreakDetector (advanced algorithm)**: The `--algorithm radon` flag switches to the advanced `RadonStreakDetector` class which inherits from `SatelliteTrailDetector` and overrides `detect_trails()` with a three-stage pipeline: LSD + Radon Transform + Perpendicular Cross Filtering. It optionally calibrates detection thresholds from ground truth trail patches via `--groundtruth <dir>`. The Radon stage downsamples aggressively (250k pixel area cap) for performance. The parent's matched filter is skipped (Radon subsumes it).

16. **Ground truth calibration**: When `--groundtruth` is provided with `--algorithm radon`, `RadonStreakDetector._calibrate_from_groundtruth()` loads PNG patches, extracts PSF width, brightness, contrast, angle, and length statistics, then `_apply_gt_calibration()` adapts detection thresholds. The calibration prints summary statistics on startup.

17. **Frame-level parallelism**: `process_video()` accepts `num_workers` and `no_gpu` parameters. When `num_workers >= 1`, a `multiprocessing.Pool` distributes `detect_trails()` calls across worker processes. The temporal buffer is fed sequentially in the main process; temporal context arrays are copied for each worker. The `_frame_results()` generator inside `process_video()` abstracts the sequential/parallel paths so all post-processing code (freeze overlays, debug panels, YOLO export, output writing) is shared. Detection results are identical regardless of worker count.

18. **CUDA GPU acceleration**: `_HAS_CUDA` is detected at module load. `SatelliteTrailDetector._use_gpu` controls per-instance GPU usage. GPU paths exist in `_detect_dim_lines_matched_filter()` (filter2D) and `_radon_transform()` (warpAffine). Both upload data once, run all operations on GPU, and download results. On any CUDA exception, `_use_gpu` is set to `False` permanently and the CPU path is used. The `--no-gpu` CLI flag disables CUDA.

19. **HITL review mode**: `--review` processes the video then launches `ReviewUI` for interactive correction. `--review-only` skips processing and opens an existing annotation file. The review UI runs sequentially after the parallel worker pool is shut down. Detections are collected in `detections_by_frame` dict during `process_video()` when `review_mode=True`.

20. **HITL annotation database**: Annotations are stored in COCO-compatible JSON (`<output>.json`). The `mnemosky_ext` field on each annotation holds Mnemosky-specific metadata (detection_meta, status, confidence, parameters_snapshot). Bounding boxes are stored in COCO format `[x, y, width, height]` (converted from internal `(x_min, y_min, x_max, y_max)`). The `missed_annotations` array is separate from `annotations` to maintain clean COCO compatibility.

21. **HITL parameter learning**: `ParameterAdapter` uses two tiers. Tier 1 (EMA) applies immediately per correction via `CORRECTION_RULES` mapping. Tier 2 (golden section search) runs on-demand when user presses `L` in the review UI, requires >= 10 calibration entries. `PARAMETER_SAFETY_BOUNDS` prevents catastrophic drift. Learned parameters persist across sessions via `~/.mnemosky/learned_params.json` profiles. On subsequent runs, `main()` loads the active profile and merges learned `canny_low`/`canny_high` into `preprocessing_params`.

22. **HITL confidence scoring**: `ParameterAdapter.compute_confidence()` computes a pseudo-confidence from contrast margin, SNR margin, length score, and smoothness, squashed through a sigmoid. Used for active learning prioritization (low-confidence detections reviewed first) and auto-accept (detections above `--auto-accept` threshold are pre-accepted).

23. **Radon debug preview**: `show_radon_preview()` provides an interactive GUI for tuning the 6 most important Radon pipeline parameters before processing. Triggered by `--preview` when `--algorithm radon` is active; the default `show_preprocessing_preview()` is used otherwise. The preview displays 4 diagnostic panels (Residual, Sinogram, LSD Lines, Detections) showing intermediate pipeline stages. Accepted parameters flow through `process_video()` → `RadonStreakDetector` via the `radon_params` dict and `_apply_radon_preview_params()`. The tunable parameters are: `radon_snr_threshold`, `pcf_ratio_threshold`, `star_mask_sigma` (stored as `_star_mask_sigma`), `lsd_log_eps` (stored as `_lsd_log_eps`), `pcf_kernel_length`, and `satellite_min_length`.

24. **Neural network detection (`--algorithm nn`)**: The `NeuralNetDetector` class uses a trained object detection model (YOLOv8/v11, ONNX, etc.) for trail detection. Three backends are supported via `_NNBackend`: `ultralytics` (primary, auto-installed), `cv2dnn` (zero extra deps, ONNX/TF/Darknet), and `onnxruntime` (broad acceleration). Backend imports are lazy — only loaded when the `nn` algorithm is selected. The model class IDs are mapped to satellite/airplane via `class_map` (configurable, default `{satellite:[0], airplane:[1]}`). Detection results are converted to the standard `detection_info` format via `_bbox_to_detection_info()` which estimates trail angle, length, brightness, and contrast from the model bbox. Hybrid mode (`--nn-hybrid`) runs both NN and classical pipeline, merging results. Workers each load their own model instance (not pickled).

25. **Application config system**: `~/.mnemosky/config.json` stores all algorithm parameters, model paths, backend choice, and general settings. Loaded via `load_config()`, saved via `save_config()`. CLI arguments override config values. `--save-config` persists current CLI parameters. The config coexists with the existing `learned_params.json` (HITL profiles) — they serve different purposes.

26. **NN preview GUI**: `show_nn_preview()` provides an interactive GUI for tuning confidence and NMS IoU thresholds with live model inference on video frames. Same dark-grey/fluorescent theme. Main frame panel shows detection overlays; sidebar has model info card, parameter sliders, class mapping, per-detection confidence bars, and inference FPS stats. Triggered by `--preview` when `--algorithm nn` is active. Tuned parameters flow through `process_video()` → `NeuralNetDetector` via `nn_params`.

27. **NN backend auto-install**: When a backend is not installed and the user selects it, `_ensure_nn_backend()` attempts `pip install` automatically. Falls back to a manual install hint on failure. Backend availability is cached in `_NN_BACKENDS_CHECKED` dict.

28. **Processing window (`--show-processing`)**: The `ProcessingWindow` class provides a live dashboard during video processing.  Four panels: LIVE FRAME (thumbnailed current frame with detection boxes), TRAIL MAP (2-D scatter of all trail centre positions plotted over normalised frame space with aspect-ratio-preserving grid), DETECTION TIMELINE (stacked satellite/airplane bar chart bucketed across the video duration, with a waveform overlay showing per-second detection density and a progress playhead), and STATUS BAR (progress ring with percentage, processing FPS, algorithm label, satellite/airplane counts with colour-coded dots, ETA, and abort hint).  Throttled to ~20 fps redraw; always redraws immediately on detection events.  Automatically enabled when `--preview` is used (seamless transition from preview to processing).  Press Q/ESC to abort.

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
| Config system | `satellite_trail_detector.py:load_config()` / `save_config()` (line ~163) |
| Config defaults | `satellite_trail_detector.py:_DEFAULT_CONFIG` (line ~132) |
| NN backend abstraction | `satellite_trail_detector.py:_NNBackend` (line ~221) |
| NN backend lazy import | `satellite_trail_detector.py:_check_nn_backend()` / `_ensure_nn_backend()` (line ~66) |
| Preprocessing preview | `satellite_trail_detector.py:show_preprocessing_preview()` (line ~475) |
| Radon debug preview | `satellite_trail_detector.py:show_radon_preview()` (line ~1517) |
| NN detection preview | `satellite_trail_detector.py:show_nn_preview()` (line ~2411) |
| HITL safety bounds | `satellite_trail_detector.py:PARAMETER_SAFETY_BOUNDS` (line ~1207) |
| HITL correction rules | `satellite_trail_detector.py:CORRECTION_RULES` (line ~1224) |
| Annotation database | `satellite_trail_detector.py:AnnotationDatabase` (line ~1296) |
| Parameter adapter | `satellite_trail_detector.py:ParameterAdapter` (line ~1684) |
| Review UI | `satellite_trail_detector.py:ReviewUI` (line ~1927) |
| Abstract interface | `satellite_trail_detector.py:BaseDetectionAlgorithm` (line ~2567) |
| Partial implementation | `satellite_trail_detector.py:DefaultDetectionAlgorithm` (line ~2730) |
| Main detector class | `satellite_trail_detector.py:SatelliteTrailDetector` (line ~2890) |
| Sensitivity presets | `satellite_trail_detector.py:SatelliteTrailDetector.__init__()` (line ~2901) |
| Signal envelope adaptation | `satellite_trail_detector.py:SatelliteTrailDetector._apply_signal_envelope()` (line ~2998) |
| Matched filter detection (+ GPU) | `satellite_trail_detector.py:SatelliteTrailDetector._detect_dim_lines_matched_filter()` (line ~3204) |
| Trail SNR computation | `satellite_trail_detector.py:SatelliteTrailDetector._compute_trail_snr()` (line ~3424) |
| Point feature detection | `satellite_trail_detector.py:SatelliteTrailDetector.detect_point_features()` (line ~3816) |
| Classification logic | `satellite_trail_detector.py:SatelliteTrailDetector.classify_trail()` (line ~3916) |
| Angle-aware airplane merge | `satellite_trail_detector.py:SatelliteTrailDetector.merge_airplane_detections()` (line ~4343) |
| Two-stage detect_trails | `satellite_trail_detector.py:SatelliteTrailDetector.detect_trails()` (line ~4456) |
| RadonStreakDetector class | `satellite_trail_detector.py:RadonStreakDetector` (line ~4953) |
| Radon GT calibration | `satellite_trail_detector.py:RadonStreakDetector._calibrate_from_groundtruth()` (line ~5012) |
| LSD detection | `satellite_trail_detector.py:RadonStreakDetector._detect_lines_lsd()` (line ~5252) |
| Radon transform (+ GPU) | `satellite_trail_detector.py:RadonStreakDetector._radon_transform()` (line ~5300) |
| Perpendicular cross filter | `satellite_trail_detector.py:RadonStreakDetector._perpendicular_cross_filter()` (line ~5475) |
| Radon detect_trails | `satellite_trail_detector.py:RadonStreakDetector.detect_trails()` (line ~5610) |
| NeuralNetDetector class | `satellite_trail_detector.py:NeuralNetDetector` (line ~7570) |
| NN detect_trails | `satellite_trail_detector.py:NeuralNetDetector.detect_trails()` (line ~7570) |
| NN param helper | `satellite_trail_detector.py:_apply_nn_params()` (line ~7570) |
| Processing window | `satellite_trail_detector.py:ProcessingWindow` (line ~10035) |
| Worker functions | `satellite_trail_detector.py:_worker_init()` / `_worker_detect()` (line ~7930) |
| Dataset utility functions | `satellite_trail_detector.py:_compute_obb_corners()` etc. (line ~6026) |
| DatasetExporter class | `satellite_trail_detector.py:DatasetExporter` (line ~6091) |
| HITL-verified dataset export | `satellite_trail_detector.py:export_dataset_from_annotations()` (line ~6635) |
| Video processing | `satellite_trail_detector.py:process_video()` (line ~6796) |
| ML dataset integration | `satellite_trail_detector.py:process_video()` dataset exporter section (line ~6791) |
| Main entry point | `satellite_trail_detector.py:main()` (line ~7365) |
