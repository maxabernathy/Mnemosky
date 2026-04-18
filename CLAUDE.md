# CLAUDE.md - AI Assistant Guide for Mnemosky

## Project Overview

**Mnemosky** (v0.2.0-sts) is a satellite and airplane trail detector for MP4 videos and RAW image folders. It uses classical computer vision, an optional Radon transform pipeline, and optionally neural network models to identify and classify celestial trails in night sky footage. Detected trails can be exported as YOLO/COCO ML datasets and fed back into the NN pipeline. The project draws on Science & Technology Studies (STS) concepts for transparency and situated observation.

**Preprocessing adjustments** `--preview`

<img width="1861" height="931" alt="Preprocessing preview" src="https://github.com/user-attachments/assets/053cb18d-e020-44a2-b73f-3d9374eef083" />

**Debug visualization** `--debug`

<img width="1366" height="1024" alt="Debug view" src="https://github.com/user-attachments/assets/6911a2f1-840d-4e76-b08f-5d59583c456c" />

**Detection output**

<img width="1366" height="1024" alt="Output" src="https://github.com/user-attachments/assets/07d3a4d7-51ed-44ce-adb6-fde3623cd305" />

## Repository Structure

```
Mnemosky/
├── satellite_trail_detector.py   # Main application (~13,580 lines, single-file)
├── hitl_architecture.md          # HITL RL system design document
├── .gitignore
└── CLAUDE.md                     # This file
```

## Tech Stack

**Core**: Python 3, OpenCV (cv2), NumPy, argparse/pathlib/multiprocessing (stdlib)

**Optional**:
```bash
pip install opencv-python numpy
pip install scipy                 # Radon NMS quality (fallback: cv2.dilate)
pip install rawpy                 # RAW image decoding (ARW, CR2, NEF, DNG)
pip install exifread              # EXIF metadata extraction
pip install ultralytics           # NN backend: YOLOv8/v11 (auto-installed on first use)
pip install onnxruntime           # NN backend: ONNX Runtime
# cv2.dnn included with opencv-python (no extra install)
```

## Architecture

### Class Hierarchy

```
SatelliteTrailDetector              (line 5857)   Core detector, two-stage pipeline
  ├── RadonStreakDetector            (line 8470)   Radon+LSD+PCF, overrides detect_trails()
  └── NeuralNetDetector             (line 9795)   NN inference, overrides detect_trails()

_NNBackend                          (line 384)    Unified NN inference wrapper
TemporalFrameBuffer                 (line 3665)   Rolling 7-frame median background
DetectionTracker                    (line 3787)   Temporal consistency + tracklet builder
TranslationLedger                   (line 3998)   Rejection audit trail (STS-inspired)
AnnotationDatabase                  (line 4240)   COCO-compatible HITL storage
ParameterAdapter                    (line 4678)   Two-tier parameter learning
ReviewUI                            (line 4956)   Interactive HITL correction window
DatasetExporter                     (line 10302)  ML dataset export (4 formats)
ProcessingWindow                    (line 11587)  Live processing dashboard
```

### Detection Classes

**`SatelliteTrailDetector`** (line 5857) — Main detector with `low`/`medium`/`high` sensitivity presets.

```python
# Key methods:
def preprocess_frame(self, frame) -> (gray, blurred)
def detect_lines(self, preprocessed) -> [line_tuples]           # Stage 1: Canny + Hough
def classify_trail(self, line, gray, color, supplementary=False) -> (trail_type, detection_info) | (None, None)
def detect_trails(self, frame, ...) -> [('satellite', {detection_info}), ...]
def _detect_dim_lines_matched_filter(self, gray, frame, ...) -> [detections]  # Stage 2: MF bank
def _compute_trail_snr(self, line, gray, noise_map) -> float
def _estimate_psf_from_stars(self, gray) -> float               # Adaptive PSF sigma
def _detect_lines_multiscale(self, preprocessed) -> [lines]     # Full-res + 0.5x Hough
def _merge_satellites_oriented(self, detections) -> [merged]    # Angle-aware satellite merge
def merge_airplane_detections(self, detections) -> [merged]     # Angle-aware airplane merge
```

**`RadonStreakDetector`** (line 8470) — Inherits SatelliteTrailDetector. Three-stage pipeline: LSD + Radon + PCF.

```python
# Overrides detect_trails() with:
def _detect_lines_lsd(self, preprocessed) -> [line_tuples]      # A-contrario line segments
def _radon_transform(self, image) -> (sinogram, peaks)          # GPU-accelerated warpAffine
def _perpendicular_cross_filter(self, residual, lines, star_mask) -> [confirmed_lines]
def _calibrate_from_groundtruth(self, gt_dir) -> calibration_dict

# Dedup/merge utilities:
def _merge_collinear_segments(self, lines) -> [merged]
def _merge_lines_oriented(self, lines) -> [merged]
def _is_duplicate_line(self, line_a, line_b) -> bool
```

**`NeuralNetDetector`** (line 9795) — Inherits SatelliteTrailDetector. Uses `_NNBackend` for inference.

```python
def detect_trails(self, frame, ...) -> [('satellite'|'airplane', {detection_info})]
def _bbox_to_detection_info(self, bbox, gray, color, nn_confidence) -> detection_info
def _merge_nn_classical(self, nn_trails, classical_trails) -> [merged]  # --nn-hybrid
def _ensure_backend(self)  # Lazy _NNBackend initialization (pickle-safe for workers)
```

**`_NNBackend`** (line 384) — Wraps `ultralytics`, `cv2dnn`, `onnxruntime` behind `predict(frame)`.

```python
def predict(self, frame) -> [{'bbox': (x1,y1,x2,y2), 'class_id': int, 'class_name': str, 'confidence': float}]
```

### Temporal & Tracking Classes

**`TemporalFrameBuffer`** (line 3665) — Rolling buffer of grayscale frames (default capacity 7). Computes per-pixel temporal median for background subtraction and MAD-based noise estimation.

```python
def add(self, gray_frame)
def is_ready(self) -> bool                   # True when buffer has >= 5 frames
def get_temporal_context(self, current_gray) -> {'diff_image', 'noise_map', 'reference', 'buffer_depth'}
```

**`DetectionTracker`** (line 3787) — Temporal consistency filter. Confirms detections seen in >= `min_hits` of last `window` frames. Tags 3+ frame detections with `tracklet_id`/`tracklet_length`.

```python
def update(self, frame_idx, detections) -> [confirmed_detections_with_temporal_hits]
```

### HITL Classes

**`AnnotationDatabase`** (line 4240) — COCO-compatible JSON with `mnemosky_ext` metadata, correction history, session tracking. Atomic writes with corruption recovery.

```python
def add_detection(self, image_id, category_id, bbox_xyxy, detection_info, params_snapshot, confidence)
def record_correction(self, annotation_id, action, ...)  # accept/reject/reclassify/adjust_bbox/add_missed
def get_calibration_set(self) -> [(detection_meta, true_label, detector_label)]
def export_coco(self, path)  # Strips mnemosky_ext for pure COCO format
```

**`ParameterAdapter`** (line 4678) — **Tier 1**: Immediate EMA per-correction (lr=0.3, decay=0.1). **Tier 2**: Batch golden-section search over 13 parameters. Configurable loss profiles: `discovery`, `precision`, `balanced`, `catalog`.

```python
def apply_correction(self, action, trail_type, detection_meta)  # Tier 1
def optimize_batch(self, calibration_set)                       # Tier 2
def compute_confidence(detection_info, params) -> float         # Sigmoid pseudo-confidence
def save_profile(self, name='default')  # Persist to ~/.mnemosky/learned_params.json
```

**`ReviewUI`** (line 4956) — Interactive OpenCV window. Dark-grey/fluorescent-accent theme. Main frame + 280px sidebar + 56px status bar. Keyboard: A(ccept) R(eject) S(atellite) P(lane) M(ark missed) Z(undo) L(earn) Tab N(ext) Space X F(ull) H(elp) Q(uit).

### STS-Inspired Classes

**`TranslationLedger`** (line 3998) — Tracks rejection statistics per pipeline stage (Callon-inspired). 14 rejection types + 3 classification counters. Makes filtering assumptions transparent and auditable via `--ledger`.

```python
def record_rejection(self, reason, line=None)
def record_classification(self, trail_type)
def summary_lines(self) -> [str]  # Human-readable rejection breakdown
def to_dict(self) -> dict         # Machine-readable for serialization
```

### Export & Visualization

**`DatasetExporter`** (line 10302) — ML dataset export: 4 formats (aabb/obb/segment/coco), temporal-episode splitting, perceptual-hash dedup, frame skip, negative mining.

**`ProcessingWindow`** (line 11587) — Live dashboard: LIVE FRAME | TRAIL MAP | DETECTION TIMELINE | STATUS BAR. Throttled ~20fps, immediate redraw on detections. Q/ESC to abort.

### Detection Data Structure

All `detect_trails()` implementations return a list of `(trail_type, detection_info)` tuples:

```python
[
    ('satellite', {
        'bbox': (x_min, y_min, x_max, y_max),  # Internal format (NOT COCO)
        'angle': 45.0,            # degrees 0-180
        'center': (640.0, 360.0),
        'length': 285.0,          # pixels
        'avg_brightness': 12.5,
        'max_brightness': 28,
        'line': (x1, y1, x2, y2),
        'contrast_ratio': 1.15,   # trail / background
        'trail_snr': 3.2,         # matched filter SNR
        'is_smooth': True,
        # Optional fields:
        'nn_confidence': 0.92,    # NN models only
        'tracklet_id': 5,         # temporal tracker
        'tracklet_length': 7,     # frames in tracklet
    }),
    ('airplane', { ... }),
]
```

### Parallelism Architecture

Detection is stateless per-frame, so frames are distributed across a `multiprocessing.Pool`:

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

- **Workers**: `min(cpu_count - 1, 8)` auto-detected. `--workers 0` for sequential.
- **GPU**: Optional CUDA for `filter2D` (matched filter) and `warpAffine` (Radon). Per-operation flags with auto CPU fallback.
- **`_frame_results()` generator**: Abstracts sequential vs parallel so post-processing is shared.

### Key Module-Level Functions

| Function | Line | Purpose |
|----------|------|---------|
| `load_config()` / `save_config()` | 184 / 211 | App config persistence (`~/.mnemosky/config.json`) |
| `_detect_hardware()` | 254 | CPU/RAM/GPU detection (cached) |
| `_optimal_raw_params()` | 332 | Auto-select RAW conversion parameters |
| `show_preprocessing_preview()` | 1011 | Interactive CLAHE/blur/Canny tuning GUI |
| `show_radon_preview()` | 2176 | Interactive Radon pipeline tuning GUI |
| `show_nn_preview()` | 3183 | Interactive NN confidence/NMS tuning GUI |
| `_worker_init()` / `_worker_detect()` | 10170 / 10221 | Multiprocessing worker functions |
| `convert_raw_folder_to_video()` | 10982 | RAW folder → MP4 with 16-bit enhancement |
| `export_dataset_from_annotations()` | 11419 | HITL-verified dataset export |
| `process_video()` | 12119 | Main video processing pipeline |
| `main()` | 12871 | CLI entry point with argument parsing |

## Detection Algorithms

### Default Pipeline (`--algorithm default`)

**Stage 1 — Canny + Hough:**
1. Grayscale → CLAHE (clip=6.0) → Gaussian blur (k=5, σ=1.8)
2. Adaptive Canny (P70/P95 gradient percentiles)
3. Morphological ops (5x5 kernels, 2 dilations, directional bridging)
4. Multi-scale HoughLinesP (full-res + 0.5x, deduped)
5. `classify_trail()` — 7 detection paths for satellites, star FP suppression

**Stage 2 — Matched Filter (supplementary dim trails):**
1. Background subtraction (median filter, k=51)
2. MAD-based noise estimation
3. Oriented filter bank (24 angles x 2 kernels = 48 convolutions at 1/2 scale)
4. SNR thresholding (>= 2.5) → Hough → per-trail SNR confirmation

### Radon Pipeline (`--algorithm radon`)

**Stage 1 — LSD:** CLAHE (clip=8.0) → `createLineSegmentDetector` at 960px → cap 50 results → classify.

**Stage 2 — Radon Transform:** Background subtraction → star masking → downsample to 500k pixels → warpAffine at 90 angles → SNR normalization → NMS peak detection → line reconstruction.

**Stage 3 — PCF:** Sample parallel/perpendicular brightness per candidate. Symmetric cross-sections (stars) rejected. Ratio threshold 2.0.

**Multi-frame accumulation**: Stacks cleaned residuals (depth=4), divides noise by sqrt(N) for SNR boost.

**Ground truth calibration** (`--groundtruth <dir>`): Loads PNG patches, measures PSF/brightness/contrast/angle distributions, adapts thresholds.

### NN Pipeline (`--algorithm nn`)

Model inference via `_NNBackend` (ultralytics/cv2dnn/onnxruntime) → class ID mapping → `_bbox_to_detection_info()`. Optional hybrid mode (`--nn-hybrid`) merges NN + classical results.

### Classification Criteria

| Feature | Satellite | Airplane |
|---------|-----------|----------|
| Pattern | Smooth, uniform | Dotted, bright points |
| Brightness | Dim, consistent | Variable, with peaks |
| Color | Monochromatic | May have colored lights |
| Visual marker | GOLD box | ORANGE box |
| Merge strategy | Angle-aware oriented merge (15deg, 20px) | Angle-aware merge (20deg threshold) |
| FP suppression | Star eccentricity rejection, PSF estimation | Bright-spot cluster check (single flare != airplane) |

### Satellite Detection Paths

1. **Primary**: dim + monochrome + smooth + length range (all 4 criteria)
2. **Strong 3/4**: 3+ criteria including smoothness and length
3. **Very dim**: smooth + below brightness threshold + in length range
4. **Extended dim+smooth+mono**: no max-length cap
5. **Extended dim+smooth+contrast**: measured contrast ratio
6. **Extended very dim+smooth**: relaxed monochrome
7. **SNR-based**: matched-filter SNR >= 2.5 + smooth (supplementary only)

## Running the Application

### Basic Usage

```bash
python satellite_trail_detector.py input.mp4 output.mp4
python satellite_trail_detector.py /path/to/raw/folder/ output.mp4   # RAW images
```

### Common Options

```bash
# Sensitivity & algorithm
--sensitivity high                     # low / medium (default) / high
--algorithm radon                      # default / radon / nn
--algorithm nn --model trail.pt        # NN detection

# Output control
--freeze-duration 2.0                  # Freeze frame on detection (seconds)
--max-duration 30                      # Limit processing duration
--detect-type satellites               # both / all / satellites / airplanes / anomalous
--no-labels                            # Hide detection labels

# Interactive preview (algorithm-specific GUI)
--preview                              # Tune parameters before processing

# Debug
--debug                                # Side-by-side debug view
--debug-only                           # Debug visualization only

# Parallelism & GPU
--workers 4                            # Parallel workers (0=sequential, default=auto)
--no-gpu                               # Disable CUDA acceleration

# Live dashboard
--show-processing                      # Processing progress window

# ML dataset export
--dataset                              # Export YOLO dataset
--dataset-format obb                   # aabb (default) / obb / segment / coco
--dataset-split 0.8 0.1 0.1           # Train/val/test ratios
--dataset-from-annotations output.json # Export from HITL-verified annotations

# HITL review
--review                               # Process + open review UI
--review-only --annotations out.json   # Review existing annotations
--hitl-profile my_camera               # Named learned parameter profile
--loss-profile precision               # discovery / precision / balanced / catalog

# RAW image options
--half-size-raw                        # Fast demosaic
--keep-video                           # Keep intermediate MP4
--no-raw-enhance                       # Skip 16-bit enhancement
--target-height 2160                   # Downscale height

# NN options
--nn-backend cv2dnn                    # ultralytics / cv2dnn / onnxruntime
--confidence 0.5 --nms-iou 0.3        # NN thresholds
--nn-hybrid                            # Merge NN + classical results
--nn-class-map '{"satellite":[0,2]}'   # Custom class mapping

# STS-inspired features
--ledger                               # Enable TranslationLedger (rejection audit)
--observer-lat 40.7 --observer-lon -74 # Observer location (Haraway)
--observer-bortle 5                    # Dark sky scale (1-9)
--observer-notes "Clear night"         # Free-text notes

# Config
--save-config                          # Persist params to ~/.mnemosky/config.json
--config /path/to/config.json          # Custom config path
```

## ML Dataset Export

Supports 4 formats via `--dataset`:

| Format | Label format | Use case |
|--------|-------------|----------|
| **aabb** | `class_id xc yc w h` | Standard YOLO |
| **obb** | `class_id x1 y1 x2 y2 x3 y3 x4 y4` | YOLO OBB (ideal for thin trails) |
| **segment** | `class_id x1 y1 ... x4 y4` | YOLO instance segmentation |
| **coco** | COCO JSON per split | Detectron2, MMDetection |

Class IDs: `0=satellite`, `1=airplane`. Features: temporal-episode splitting, perceptual-hash dedup, frame skip, negative sample mining, HITL-verified export.

## HITL Annotation Database

Annotations stored in COCO-compatible JSON (`<output>.json`):

```json
{
    "annotations": [{
        "bbox": [100, 200, 350, 40],
        "mnemosky_ext": {
            "source": "detector", "status": "confirmed", "confidence": 0.82,
            "detection_meta": {"angle": 135, "length": 352, "contrast_ratio": 1.12}
        }
    }],
    "corrections": [{"action": "accept", "annotation_id": 1}],
    "learned_parameters": {"current": {...}, "update_count": 12}
}
```

- `bbox` is COCO format `[x, y, w, h]` (internal format is `(x_min, y_min, x_max, y_max)`)
- Status: `"pending"` | `"confirmed"` | `"rejected"`
- Actions: `"accept"` | `"reject"` | `"reclassify"` | `"adjust_bbox"` | `"add_missed"`
- Learned params persist in `~/.mnemosky/learned_params.json`

## GUI Theme

All preview/review/processing windows use a consistent dark-grey/fluorescent-accent theme:
- **Background**: #1E1E1E, panels: #2A2A2A
- **Text**: #D2D2D2 primary, #787878 secondary
- **Accent**: #50FFC8 (BGR 200,255,80) — sliders, active values, highlights
- Custom-drawn sliders (not native trackbars), single-window layout, `cv2.setMouseCallback` interaction
- Built-in documentation overlays with konami easter egg

## Module-Level Constants

| Constant | Line | Purpose |
|----------|------|---------|
| `__version__` | 48 | `'0.2.0-sts'` |
| `_HAS_CUDA` | 57 | CUDA GPU available |
| `_HAS_SCIPY` | 52 | scipy available (Radon NMS) |
| `_HAS_RAWPY` / `_HAS_EXIFREAD` | 68 / 76 | RAW/EXIF support |
| `_NN_BACKENDS_CHECKED` | 83 | Cached backend availability |
| `_DEFAULT_CONFIG` | 153 | Default app configuration |
| `_HARDWARE_PROFILE` | 251 | Cached CPU/RAM/GPU detection |
| `LOSS_PROFILES` | 4127 | 4 named loss profiles (discovery/precision/balanced/catalog) |
| `PARAMETER_SAFETY_BOUNDS` | 4151 | Hard min/max for 13 learnable parameters |
| `CORRECTION_RULES` | 4168 | (action, trail_type) → parameter adjustment rules |

## File Locations Quick Reference

| Component | Location |
|-----------|----------|
| Config system | `load_config()` / `save_config()` (line 184) |
| Hardware detection | `_detect_hardware()` (line 254) |
| NN backend | `_NNBackend` (line 384) |
| Preprocessing preview | `show_preprocessing_preview()` (line 1011) |
| Radon preview | `show_radon_preview()` (line 2176) |
| NN preview | `show_nn_preview()` (line 3183) |
| Temporal buffer | `TemporalFrameBuffer` (line 3665) |
| Detection tracker | `DetectionTracker` (line 3787) |
| Translation ledger | `TranslationLedger` (line 3998) |
| HITL safety bounds | `PARAMETER_SAFETY_BOUNDS` (line 4151) |
| Annotation database | `AnnotationDatabase` (line 4240) |
| Parameter adapter | `ParameterAdapter` (line 4678) |
| Review UI | `ReviewUI` (line 4956) |
| Main detector | `SatelliteTrailDetector` (line 5857) |
| Sensitivity presets | `SatelliteTrailDetector.__init__()` (line 5868) |
| Signal envelope | `_apply_signal_envelope()` (line 6010) |
| PSF estimation | `_estimate_psf_from_stars()` (line 6089) |
| Multi-scale Hough | `_detect_lines_multiscale()` (line 6150) |
| Classification logic | `classify_trail()` (line ~6300) |
| Radon detector | `RadonStreakDetector` (line 8470) |
| GT calibration | `_calibrate_from_groundtruth()` (line 8545) |
| NN detector | `NeuralNetDetector` (line 9795) |
| Worker functions | `_worker_init()` / `_worker_detect()` (line 10170) |
| Dataset utilities | `_compute_obb_corners()` etc. (line 10237) |
| Dataset exporter | `DatasetExporter` (line 10302) |
| RAW EXIF extraction | `_extract_exif_from_raw()` (line 10851) |
| RAW conversion | `convert_raw_folder_to_video()` (line 10982) |
| HITL dataset export | `export_dataset_from_annotations()` (line 11419) |
| Processing window | `ProcessingWindow` (line 11587) |
| Video processing | `process_video()` (line 12119) |
| CLI entry point | `main()` (line 12871) |

## Important Notes for AI Assistants

1. **Single-file architecture**: All code is in `satellite_trail_detector.py` (~13,580 lines). Do not create additional modules unless specifically requested.

2. **Detection data format**: `detect_trails()` returns `(trail_type, detection_info)` tuples where `detection_info` is a dict. Never assume bare bbox tuples.

3. **Class hierarchy**: `SatelliteTrailDetector` is the base class (not abstract). `RadonStreakDetector` and `NeuralNetDetector` inherit from it and override `detect_trails()`. There is no ABC.

4. **Two-stage pipeline**: Primary (Canny + Hough) + supplementary (matched filter). `supplementary=True` relaxes contrast thresholds and enables SNR-based detection. Do not remove either stage.

5. **Star FP suppression**: Spatial spread checks (>15% trail length), minimum peak separation (>10% trail length), star eccentricity rejection (moments-based). Preserve all three mechanisms.

6. **Signal envelope**: User-marked trail examples flow: `show_preprocessing_preview()` → `main()` → `process_video()` → `SatelliteTrailDetector.__init__()` → `_apply_signal_envelope()`.

7. **Parallelism**: Workers have independent detector instances. `_NNBackend` is lazily initialized per-worker (not pickled). `freeze_support()` required for Windows.

8. **GPU acceleration**: Per-operation flags (`_use_gpu_filter`, `_use_gpu_warp`, `_use_gpu_median`). On CUDA failure, only the failed operation falls back — others continue on GPU.

9. **HITL bbox format**: Internal `(x_min, y_min, x_max, y_max)` vs COCO `[x, y, width, height]`. `AnnotationDatabase.add_detection()` handles conversion.

10. **Video codec fallback**: MPEG-4 → H.264 variants → system default. Maintain this pattern.

11. **Preview theme**: Custom dark-grey/fluorescent-accent theme drawn with OpenCV primitives. Single window, no native trackbars. Maintain the aesthetic when modifying.

12. **RAW enhancement**: 16-bit CLAHE + percentile stretch in LAB space (L-channel only). Preserves colour for airplane classification.

13. **Config persistence**: `~/.mnemosky/config.json` (app config) coexists with `~/.mnemosky/learned_params.json` (HITL profiles) — different purposes.

14. **Translation Ledger**: `--ledger` enables rejection auditing via `TranslationLedger`. Counts 14 rejection types per pipeline stage. Summary printed after processing.

15. **Loss profiles**: `--loss-profile` selects named profiles (`discovery`/`precision`/`balanced`/`catalog`) that weight FP/FN/misclassification differently in Tier 2 learning.

## Common Tasks

### Adding a new sensitivity preset

Add to `SatelliteTrailDetector.__init__()` (line 5868):
```python
presets['custom'] = {
    'canny_low': ..., 'canny_high': ...,
    'satellite_min_length': ..., 'satellite_max_length': ...,
    'satellite_contrast_min': ...,
}
```

### Modifying classification logic

Edit `SatelliteTrailDetector.classify_trail()` — returns `(trail_type, detection_info_dict)` or `(None, None)`.

### Adding new CLI arguments

Add to `main()` (line 12871) argument parser, then handle in `process_video()` (line 12119).

### Extending the detector

```python
class CustomDetector(SatelliteTrailDetector):
    def detect_trails(self, frame, debug_info=None, temporal_context=None,
                      exposure_time=13.0, fov_degrees=None):
        # Custom pipeline, must return [(trail_type, detection_info_dict), ...]
        pass
```
