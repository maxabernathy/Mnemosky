# HITL Reinforcement Learning Architecture for Mnemosky

## Table of Contents
1. [SOTA Method Selection](#1-sota-method-selection)
2. [Database Schema](#2-database-schema)
3. [Review UI Design](#3-review-ui-design)
4. [Learning Loop Design](#4-learning-loop-design)
5. [Integration Plan](#5-integration-plan)

---

## 1. SOTA Method Selection

### Problem Characterization

Mnemosky's detection pipeline has **~15 continuous parameters** (per sensitivity preset) that control thresholds for edge detection, line detection, brightness classification, contrast ratios, length ranges, and morphological operations. Human feedback comes as **sparse corrections**: accept/reject/reclassify individual detections, or mark missed trails. The goal is to adapt these parameters over time so the detector learns the user's specific sky conditions, camera setup, and tolerance for false positives vs. missed detections.

Key constraints:
- **Small sample size**: Users correct maybe 5-50 detections per video session
- **No gradient**: The detection pipeline is non-differentiable (Canny, Hough, morphology)
- **15-dimensional continuous space**: Too many for grid search, too few for deep RL
- **Safety**: Parameters must stay within physically meaningful bounds
- **Latency**: Learning should complete in <1 second after a review session
- **No external dependencies beyond numpy**: Must not require PyTorch, scikit-learn, etc.

### Methods Evaluated

| Method | Strengths | Weaknesses for Mnemosky |
|--------|-----------|------------------------|
| **Bayesian Optimization (GP-UCB)** | Sample-efficient, handles noise, principled exploration | Requires scipy or GPy; GP scales O(n^3) with corrections; overkill for streaming corrections |
| **CMA-ES** | Excellent for 5-50D continuous, derivative-free, self-adapting covariance | Needs population of ~20+ evaluations per generation; not well-suited to streaming single corrections |
| **Thompson Sampling / Contextual Bandits** | Online, handles sparse feedback, principled exploration | Designed for discrete arms, not 15D continuous parameter spaces; contextual version needs feature engineering |
| **Nelder-Mead** | Simple, no derivatives, low overhead | Gets stuck in local optima; no uncertainty quantification; poor in >10D |
| **Calibration Set + Weighted Loss** | Dead simple, interpretable, fast, numpy-only | No principled exploration; can overfit to small correction sets |

### Selected Approach: Hybrid Calibration-Set + Exponential-Smoothing Parameter Adaptation

**Rationale**: Given the constraints (numpy-only, sparse feedback, 15 parameters, must be fast), the most practical and robust approach is a **two-tier system**:

**Tier 1 -- Direct threshold adaptation** (immediate, per-correction):
When a user corrects a detection, we can directly infer which parameter(s) caused the error and nudge them. For example:
- User rejects a satellite detection as false positive --> the trail's measured contrast was 1.05 and `satellite_contrast_min` is 1.03 --> raise `satellite_contrast_min` toward 1.05
- User marks a missed satellite trail --> the trail's measured length was 55px and `satellite_min_length` is 60 --> lower `satellite_min_length` toward 55

This is **exponential moving average (EMA) adaptation**: each correction pulls the parameter toward the value that would have produced the correct result, with a learning rate that decays as more corrections accumulate (preventing oscillation).

**Tier 2 -- Periodic global re-optimization** (batch, after each session):
After a review session, use the accumulated **calibration set** (all confirmed TPs, confirmed FPs, and marked FNs) to evaluate candidate parameter vectors. Use **coordinate-wise golden-section search** (1D optimization per parameter, cycling through all 15) to minimize a weighted loss:

```
L = w_fp * count(false_positives) + w_fn * count(false_negatives) + w_cls * count(misclassifications)
```

This is essentially a structured grid search that exploits the observation that most parameters are **approximately separable** (changing `satellite_contrast_min` barely affects `canny_low`'s optimal value).

**Why not Bayesian Optimization?** While BO is the theoretically optimal choice for sample-efficient black-box optimization, it requires either scipy's `minimize` or a custom GP implementation. The GP kernel computation is O(n^3) and the acquisition function optimization adds complexity. For 15 parameters with <50 corrections per session, the overhead is not justified. The calibration-set approach achieves similar practical performance with pure numpy.

**Why not CMA-ES?** CMA-ES needs a population of ~2*N = 30 candidate evaluations per generation, each requiring a full pass over the calibration set. With only 5-50 corrections, we would need many generations to converge, making it slow for interactive use. It excels when you have thousands of cheap evaluations, not dozens of expensive ones.

### Adaptation Rules (Tier 1 Detail)

Each correction maps to specific parameters via a **correction-to-parameter mapping table**:

| Correction Type | Diagnostic | Parameter(s) Affected | Direction |
|----------------|------------|----------------------|-----------|
| FP rejected (was satellite) | contrast < threshold+margin | `satellite_contrast_min` | raise |
| FP rejected (was satellite) | brightness > airplane_min | `airplane_brightness_min` | lower |
| FP rejected (was satellite) | length outside range | `satellite_min_length` or `satellite_max_length` | tighten |
| FP rejected (was satellite) | near cloud (high surround std) | (no parameter; flag for filtering) | -- |
| FP rejected (was airplane) | no dotted pattern | `airplane_brightness_min`, `airplane_saturation_min` | raise |
| FN missed satellite | contrast was X | `satellite_contrast_min` | lower toward X |
| FN missed satellite | length was X | `satellite_min_length` / `satellite_max_length` | widen toward X |
| FN missed satellite | MF SNR was X | `mf_snr_threshold` | lower toward X |
| Reclassified sat-->airplane | had bright spots | (no parameter change; classification was wrong) | -- |
| Reclassified airplane-->sat | low brightness variation | `airplane_brightness_min` | raise |

The EMA update rule for each parameter p:

```
p_new = p_old + alpha * (p_target - p_old)

alpha = base_lr / (1 + n_corrections_for_p * decay_rate)
```

Where:
- `base_lr = 0.3` (aggressive initial learning)
- `decay_rate = 0.1` (slow decay so early corrections have strong effect)
- `p_target` = the parameter value that would have produced the correct result
- `n_corrections_for_p` = how many times this specific parameter has been updated

---

## 2. Database Schema

### Design Principles

1. **COCO-compatible core**: Standard `images`, `annotations`, `categories` fields so the database can be exported directly for ML training
2. **Correction history**: Every human action is recorded with timestamps for audit and replay
3. **Detection metadata**: Full detection_info dict preserved for learning loop analysis
4. **Session tracking**: Group corrections by video/session for batch re-optimization
5. **Single JSON file**: No external database; the file lives alongside the video output

### Schema Definition

```json
{
    "info": {
        "description": "Mnemosky HITL annotation database",
        "version": "1.0",
        "date_created": "2026-02-15T12:00:00Z",
        "mnemosky_version": "1.0.0"
    },

    "categories": [
        {"id": 0, "name": "satellite", "supercategory": "trail"},
        {"id": 1, "name": "airplane", "supercategory": "trail"}
    ],

    "images": [
        {
            "id": 1,
            "file_name": "video_name_f000123.jpg",
            "width": 1920,
            "height": 1080,
            "frame_index": 123,
            "video_source": "input_video.mp4",
            "session_id": "sess_20260215_143022"
        }
    ],

    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [100, 200, 350, 40],
            "area": 14000,
            "iscrowd": 0,

            "mnemosky_ext": {
                "source": "detector",
                "status": "confirmed",
                "confidence": 0.82,
                "review_action": "accepted",
                "reviewed_at": "2026-02-15T14:35:12Z",

                "detection_meta": {
                    "angle": 135.0,
                    "center": [275.0, 220.0],
                    "length": 352.0,
                    "avg_brightness": 22.5,
                    "max_brightness": 41,
                    "line": [100, 240, 450, 200],
                    "contrast_ratio": 1.12,
                    "brightness_std": 3.2,
                    "trail_snr": 4.1,
                    "photometry_class": "steady",
                    "velocity_px_per_sec": 27.1,
                    "orbit_class": "LEO",
                    "supplementary": false
                },

                "original_category_id": 0,
                "original_bbox": [100, 200, 350, 40],

                "parameters_snapshot": {
                    "sensitivity": "high",
                    "satellite_contrast_min": 1.03,
                    "satellite_min_length": 60,
                    "satellite_max_length": 1400,
                    "canny_low": 2,
                    "canny_high": 35
                }
            }
        }
    ],

    "missed_annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 0,
            "bbox": [500, 300, 200, 30],
            "area": 6000,
            "marked_at": "2026-02-15T14:36:45Z",
            "session_id": "sess_20260215_143022",

            "estimated_meta": {
                "angle": 98.0,
                "length": 203.0,
                "avg_brightness": 15.0,
                "contrast_ratio": 1.04
            }
        }
    ],

    "corrections": [
        {
            "id": 1,
            "annotation_id": 1,
            "session_id": "sess_20260215_143022",
            "timestamp": "2026-02-15T14:35:12Z",
            "action": "accept",
            "previous_status": "pending",
            "new_status": "confirmed",
            "previous_category_id": null,
            "new_category_id": null,
            "previous_bbox": null,
            "new_bbox": null,
            "notes": null
        },
        {
            "id": 2,
            "annotation_id": 3,
            "session_id": "sess_20260215_143022",
            "timestamp": "2026-02-15T14:35:18Z",
            "action": "reject",
            "previous_status": "pending",
            "new_status": "rejected",
            "previous_category_id": 0,
            "new_category_id": null,
            "previous_bbox": null,
            "new_bbox": null,
            "notes": "Cloud edge, not a trail"
        },
        {
            "id": 3,
            "annotation_id": 5,
            "session_id": "sess_20260215_143022",
            "timestamp": "2026-02-15T14:36:02Z",
            "action": "reclassify",
            "previous_status": "pending",
            "new_status": "confirmed",
            "previous_category_id": 0,
            "new_category_id": 1,
            "previous_bbox": null,
            "new_bbox": null,
            "notes": null
        },
        {
            "id": 4,
            "annotation_id": null,
            "session_id": "sess_20260215_143022",
            "timestamp": "2026-02-15T14:36:45Z",
            "action": "add_missed",
            "previous_status": null,
            "new_status": "confirmed",
            "previous_category_id": null,
            "new_category_id": 0,
            "previous_bbox": null,
            "new_bbox": [500, 300, 200, 30],
            "missed_annotation_id": 1,
            "notes": null
        }
    ],

    "sessions": [
        {
            "id": "sess_20260215_143022",
            "video_source": "input_video.mp4",
            "started_at": "2026-02-15T14:30:22Z",
            "completed_at": "2026-02-15T14:42:10Z",
            "sensitivity": "high",
            "algorithm": "default",
            "frames_reviewed": 45,
            "total_frames": 200,
            "corrections_count": 12,
            "accepted_count": 30,
            "rejected_count": 5,
            "reclassified_count": 2,
            "missed_count": 3,

            "parameters_before": {
                "satellite_contrast_min": 1.03,
                "satellite_min_length": 60,
                "satellite_max_length": 1400,
                "canny_low": 2,
                "canny_high": 35,
                "hough_threshold": 20,
                "min_line_length": 35,
                "max_line_gap": 60,
                "brightness_threshold": 12,
                "airplane_brightness_min": 45,
                "airplane_saturation_min": 2,
                "min_aspect_ratio": 3,
                "mf_snr_threshold": 2.3,
                "mf_sigma_perp": 0.7
            },
            "parameters_after": {
                "satellite_contrast_min": 1.04,
                "satellite_min_length": 55,
                "satellite_max_length": 1400,
                "canny_low": 2,
                "canny_high": 35,
                "hough_threshold": 20,
                "min_line_length": 35,
                "max_line_gap": 60,
                "brightness_threshold": 12,
                "airplane_brightness_min": 48,
                "airplane_saturation_min": 2,
                "min_aspect_ratio": 3,
                "mf_snr_threshold": 2.1,
                "mf_sigma_perp": 0.7
            }
        }
    ],

    "learned_parameters": {
        "current": {
            "satellite_contrast_min": 1.04,
            "satellite_min_length": 55,
            "satellite_max_length": 1400,
            "canny_low": 2,
            "canny_high": 35,
            "hough_threshold": 20,
            "min_line_length": 35,
            "max_line_gap": 60,
            "brightness_threshold": 12,
            "airplane_brightness_min": 48,
            "airplane_saturation_min": 2,
            "min_aspect_ratio": 3,
            "mf_snr_threshold": 2.1,
            "mf_sigma_perp": 0.7
        },
        "update_count": 12,
        "last_updated": "2026-02-15T14:42:10Z",
        "history": [
            {
                "timestamp": "2026-02-15T14:42:10Z",
                "session_id": "sess_20260215_143022",
                "parameters": { "...": "..." },
                "loss_before": 0.35,
                "loss_after": 0.22
            }
        ]
    }
}
```

### Field Specifications

#### `annotations[].mnemosky_ext.status`
- `"pending"` -- Detection not yet reviewed by user
- `"confirmed"` -- User accepted detection (TP)
- `"rejected"` -- User rejected detection (FP)

#### `annotations[].mnemosky_ext.source`
- `"detector"` -- Produced by the detection pipeline
- `"human"` -- Manually drawn by the user (for missed detections)

#### `annotations[].mnemosky_ext.confidence`
Pseudo-confidence score computed from detection metadata:
```python
confidence = sigmoid(
    0.3 * normalized_contrast +
    0.2 * normalized_snr +
    0.2 * normalized_smoothness +
    0.15 * length_in_range_score +
    0.15 * brightness_in_range_score
)
```
Used for active learning prioritization (review low-confidence detections first).

#### `annotations[].bbox` (COCO format)
`[x, y, width, height]` -- top-left corner + dimensions. Differs from Mnemosky's internal `(x_min, y_min, x_max, y_max)` format. Conversion happens at import/export boundaries.

#### `corrections[].action`
One of: `"accept"`, `"reject"`, `"reclassify"`, `"adjust_bbox"`, `"add_missed"`, `"delete"`

#### `missed_annotations`
Separate from `annotations` to maintain clean COCO compatibility. Each missed annotation represents a false negative -- a trail the user marked that the detector did not find. These are the most valuable for learning: they indicate exactly where the detector's thresholds are too strict.

### File Naming Convention
```
<output_video_stem>_annotations.json
```
Example: `output_annotations.json` alongside `output.mp4`.

### Export to Pure COCO
The database can be exported to standard COCO format by:
1. Filtering annotations where `status == "confirmed"`
2. Including missed_annotations as regular annotations
3. Stripping `mnemosky_ext` fields
4. Converting category IDs to match COCO convention

---

## 3. Review UI Design

### Layout Overview

The review UI operates as a **single OpenCV window** (consistent with the existing preview GUI pattern). It follows the same dark-grey/fluorescent-accent theme used in `show_preprocessing_preview()`.

```
+------------------------------------------------------------------------+
|  MNEMOSKY REVIEW                                   [frame 23/200]  F5  |
+------------------------------------------------------------------------+
|                                                    |                   |
|                                                    |  DETECTIONS (3)   |
|                                                    |  +-----------+    |
|                                                    |  | #1 SAT    |    |
|               MAIN FRAME VIEW                      |  | conf 0.82 |    |
|            (current frame with                     |  | [ACCEPT]  |    |
|             detection overlays)                    |  +-----------+    |
|                                                    |  | #2 SAT  * |    |
|           Confirmed: green box                     |  | conf 0.41 |    |
|           Pending: pulsing accent box              |  | [REVIEW]  |    |
|           Rejected: dim red box (fades)            |  +-----------+    |
|           Selected: thick bright accent box        |  | #3 AIR    |    |
|                                                    |  | conf 0.93 |    |
|                                                    |  | [ACCEPT]  |    |
|                                                    |  +-----------+    |
|                                                    |                   |
|                                                    |  MISSED (draw)    |
|                                                    |  [M] to mark      |
|                                                    |                   |
|                                                    |  PARAMETERS       |
|                                                    |  contrast: 1.04   |
|                                                    |  SNR thr:  2.1    |
|                                                    |                   |
|                                                    |  SESSION STATS    |
|                                                    |  Reviewed: 23     |
|                                                    |  Accepted: 18     |
|                                                    |  Rejected:  3     |
|                                                    |  Reclassed: 1     |
|                                                    |  Missed:    1     |
+------------------------------------------------------------------------+
|  [<] prev frame  |  frame slider  |  next frame [>]  |  [LEARN] [SAVE] |
+------------------------------------------------------------------------+
```

### Window Dimensions
- **Total**: Same as input video width + 280px sidebar, height + 56px status bar
- **Main frame area**: Input video resolution (e.g. 1920x1080), occupying ~87% of width
- **Right sidebar**: 280px wide, full height minus status bar
- **Bottom status bar**: 56px tall, full width (same as preview GUI)

### Color Scheme (matching existing theme)

| Element | Color (BGR) | Hex |
|---------|------------|-----|
| Background | (30, 30, 30) | #1E1E1E |
| Panel cards | (42, 42, 42) | #2A2A2A |
| Primary text | (210, 210, 210) | #D2D2D2 |
| Secondary text | (120, 120, 120) | #787878 |
| Accent (fluorescent) | (200, 255, 80) | #50FFC8 |
| Confirmed detection | (80, 200, 80) | #50C850 (green) |
| Pending detection | (200, 255, 80) | #50FFC8 (accent, pulsing) |
| Rejected detection | (80, 80, 180) | #B45050 (dim red) |
| Selected detection | (200, 255, 80) | #50FFC8 (thick, bright) |
| Low-confidence star | (0, 200, 255) | #FFC800 (amber warning) |
| Missed (user-drawn) | (255, 100, 200) | #C864FF (magenta) |

### Detection Overlay on Main Frame

Each detection is drawn as a bounding box on the main frame:
- **Confirmed (accepted)**: Solid green box, 1px, with small green checkmark icon
- **Pending (not yet reviewed)**: Pulsing accent-colored dotted box (brightness oscillates 0.5-1.0 over 1 second), 2px
- **Rejected**: Thin dim red box, 1px, 50% opacity
- **Currently selected**: Thick bright accent box, 3px, with detection number overlay
- **Low confidence (< 0.5)**: Amber star (*) overlay in top-right corner of box

Active-learning priority: When navigating frames, the UI **auto-advances to the next frame containing low-confidence detections** (confidence < 0.5), skipping frames where all detections are high-confidence. The user can override this with manual frame navigation.

### Sidebar Detection Cards

Each detection gets a card in the sidebar showing:
- Detection number (#1, #2, ...)
- Category label (SAT / AIR) with category color
- Confidence score (0.00-1.00)
- Key metadata: length, angle, contrast, SNR
- Status badge: [ACCEPT] green, [REVIEW] amber, [REJECT] red
- Low-confidence indicator: amber star (*) next to detection number

The currently selected detection card has a bright accent border. Clicking a card selects it; the main view centers/zooms on that detection.

### Keyboard Shortcuts

| Key | Action | Context |
|-----|--------|---------|
| `A` | **Accept** selected detection | Detection selected |
| `R` | **Reject** selected detection | Detection selected |
| `S` | **Reclassify** as satellite | Detection selected, currently airplane |
| `P` | **Reclassify** as airplane (plane) | Detection selected, currently satellite |
| `M` | Enter **mark missed** mode (draw bbox) | Any time |
| `Escape` | Cancel current operation / exit mark mode | During mark mode |
| `Tab` | Cycle to **next detection** on current frame | Any time |
| `Shift+Tab` | Cycle to **previous detection** | Any time |
| `Right` / `D` | **Next frame** | Any time |
| `Left` / `W` | **Previous frame** | Any time |
| `N` | **Next unreviewed frame** (active learning skip) | Any time |
| `B` | **Previous unreviewed frame** | Any time |
| `Space` | **Accept all** detections on current frame | Any time |
| `X` | **Reject all** detections on current frame | Any time |
| `1`-`9` | **Select detection** by number | Any time |
| `Z` | **Undo** last correction | Any time |
| `Ctrl+S` | **Save** annotations to disk | Any time |
| `L` | **Run learning** (apply accumulated corrections to params) | Any time |
| `Q` | **Quit** review mode (prompts save) | Any time |
| `F` | Toggle **full frame** view (hide sidebar) | Any time |
| `+` / `-` | **Zoom** in/out on selected detection | Detection selected |
| `H` | Show **help** overlay | Any time |

### Mouse Interactions

| Action | Behavior |
|--------|----------|
| **Click on detection box** | Select that detection |
| **Click on sidebar card** | Select that detection and scroll main view to it |
| **Click + drag frame slider** | Navigate to frame |
| **Right-click on detection** | Context menu: Accept / Reject / Reclassify / Delete |
| **Middle-click on frame** | Quick-mark missed: starts bbox drawing from click point |
| **Click + drag (in mark mode)** | Draw bounding box for missed detection |
| **Scroll wheel** | Navigate frames (up=forward, down=back) |
| **Scroll wheel + Ctrl** | Zoom in/out on main view |
| **Click + drag box edge (selected)** | Adjust bounding box |
| **Click + drag box corner (selected)** | Resize bounding box |

### Mark Missed Mode

When the user presses `M` or middle-clicks:
1. Cursor changes to crosshair
2. Status bar shows "MARK MISSED: Click and drag to draw bounding box"
3. User draws a bounding box on the frame
4. After releasing mouse, a popup card appears asking for category (satellite/airplane)
5. The missed annotation is created and added to `missed_annotations`
6. The detection is immediately visible as a magenta box with "MISSED" label

### Active Learning Prioritization

The review UI uses detection confidence to prioritize human attention:

1. **Frame ordering**: Frames are sorted by **minimum detection confidence** (lowest-confidence frame first). The `N` key advances to the next frame with unreviewed low-confidence detections.

2. **Within-frame ordering**: Detections on a frame are sorted by confidence (lowest first). Tab-cycling follows this order.

3. **Visual emphasis**: Low-confidence detections (< 0.5) get an amber star indicator and a pulsing box animation. High-confidence detections (> 0.8) have a subtle, thin box.

4. **Smart skip**: Frames where ALL detections have confidence > 0.9 are auto-accepted (with a configurable threshold) and skipped in the review queue. The user can still navigate to them manually.

### Confidence Computation

Detection confidence is computed from the detection metadata:

```python
def compute_confidence(detection_info, params):
    """Compute pseudo-confidence score for active learning prioritization."""
    contrast = detection_info.get('contrast_ratio', 1.0)
    snr = detection_info.get('trail_snr', 0.0)
    length = detection_info.get('length', 0.0)
    brightness = detection_info.get('avg_brightness', 0.0)
    brightness_std = detection_info.get('brightness_std', 0.0)

    # How far above the threshold is each feature?
    contrast_margin = (contrast - params['satellite_contrast_min']) / params['satellite_contrast_min']
    snr_margin = (snr - 2.5) / 2.5 if snr > 0 else 0

    # Length: score 1.0 if centered in range, 0.0 at edges
    len_mid = (params['satellite_min_length'] + params['satellite_max_length']) / 2
    len_range = params['satellite_max_length'] - params['satellite_min_length']
    length_score = 1.0 - min(1.0, 2 * abs(length - len_mid) / len_range) if len_range > 0 else 0.5

    # Smoothness score (lower variation = higher confidence for satellites)
    smoothness = 1.0 - min(1.0, brightness_std / (brightness + 1e-5) / 0.4)

    # Weighted combination
    raw = (0.30 * max(0, min(1, contrast_margin * 5))
         + 0.25 * max(0, min(1, snr_margin * 2))
         + 0.20 * max(0, length_score)
         + 0.25 * max(0, smoothness))

    # Sigmoid squash to [0, 1]
    return 1.0 / (1.0 + np.exp(-6 * (raw - 0.5)))
```

---

## 4. Learning Loop Design

### Overview

The learning loop converts human corrections into parameter updates through two mechanisms operating at different timescales:

1. **Immediate adaptation** (Tier 1): Each individual correction nudges the relevant parameter(s) via EMA
2. **Session-level re-optimization** (Tier 2): After a review session, batch optimization over the full calibration set

### Tier 1: Immediate EMA Adaptation

#### Correction-to-Parameter Mapping

```python
# Maps (correction_action, trail_type) to a list of parameter adjustments
CORRECTION_RULES = {
    # False positive rejection -- tighten thresholds
    ('reject', 'satellite'): [
        {
            'param': 'satellite_contrast_min',
            'diagnostic': lambda meta: meta.get('contrast_ratio'),
            'direction': 'raise',  # raise threshold to exclude this detection
            'target_fn': lambda diag_val, current: max(current, diag_val + 0.01),
        },
        {
            'param': 'satellite_min_length',
            'diagnostic': lambda meta: meta.get('length'),
            'direction': 'raise',
            'condition': lambda meta, params: meta.get('length', 999) < params['satellite_min_length'] * 1.5,
            'target_fn': lambda diag_val, current: max(current, diag_val + 5),
        },
    ],
    ('reject', 'airplane'): [
        {
            'param': 'airplane_brightness_min',
            'diagnostic': lambda meta: meta.get('avg_brightness'),
            'direction': 'raise',
            'target_fn': lambda diag_val, current: max(current, diag_val + 5),
        },
    ],

    # False negative (missed trail) -- relax thresholds
    ('add_missed', 'satellite'): [
        {
            'param': 'satellite_contrast_min',
            'diagnostic': lambda meta: meta.get('contrast_ratio'),
            'direction': 'lower',
            'target_fn': lambda diag_val, current: min(current, max(1.01, diag_val - 0.005)),
        },
        {
            'param': 'satellite_min_length',
            'diagnostic': lambda meta: meta.get('length'),
            'direction': 'lower',
            'target_fn': lambda diag_val, current: min(current, max(20, diag_val - 10)),
        },
        {
            'param': 'mf_snr_threshold',
            'diagnostic': lambda meta: meta.get('trail_snr'),
            'direction': 'lower',
            'condition': lambda meta, params: meta.get('trail_snr') is not None,
            'target_fn': lambda diag_val, current: min(current, max(1.5, diag_val - 0.2)),
        },
    ],

    # Reclassification -- adjust class boundary
    ('reclassify_to_airplane', 'satellite'): [
        {
            'param': 'airplane_brightness_min',
            'diagnostic': lambda meta: meta.get('avg_brightness'),
            'direction': 'lower',
            'target_fn': lambda diag_val, current: min(current, diag_val - 5),
        },
    ],
    ('reclassify_to_satellite', 'airplane'): [
        {
            'param': 'airplane_brightness_min',
            'diagnostic': lambda meta: meta.get('avg_brightness'),
            'direction': 'raise',
            'target_fn': lambda diag_val, current: max(current, diag_val + 10),
        },
    ],
}
```

#### EMA Update Implementation

```python
class ParameterAdapter:
    """Adapts detection parameters from human corrections using EMA."""

    def __init__(self, initial_params, safety_bounds):
        self.params = dict(initial_params)
        self.safety_bounds = safety_bounds  # {param: (min_val, max_val)}
        self.update_counts = {p: 0 for p in initial_params}  # per-parameter
        self.base_lr = 0.3
        self.decay_rate = 0.1

    def apply_correction(self, correction_action, trail_type, detection_meta):
        """Apply a single human correction to parameters."""
        key = (correction_action, trail_type)
        rules = CORRECTION_RULES.get(key, [])

        updates = {}
        for rule in rules:
            param = rule['param']
            if param not in self.params:
                continue

            # Check condition (if any)
            if 'condition' in rule and not rule['condition'](detection_meta, self.params):
                continue

            # Get diagnostic value
            diag_val = rule['diagnostic'](detection_meta)
            if diag_val is None:
                continue

            # Compute target
            target = rule['target_fn'](diag_val, self.params[param])

            # EMA update with decaying learning rate
            n = self.update_counts[param]
            alpha = self.base_lr / (1.0 + n * self.decay_rate)

            new_val = self.params[param] + alpha * (target - self.params[param])

            # Enforce safety bounds
            lo, hi = self.safety_bounds[param]
            new_val = max(lo, min(hi, new_val))

            self.params[param] = new_val
            self.update_counts[param] = n + 1
            updates[param] = new_val

        return updates
```

### Tier 2: Session-Level Batch Re-Optimization

After a review session completes (user presses `L` or quits), the system runs coordinate-wise optimization over the calibration set.

#### Calibration Set Construction

The calibration set consists of:
- **True positives (TP)**: Accepted detections -- detection_meta + correct category
- **False positives (FP)**: Rejected detections -- detection_meta + "should not have been detected"
- **False negatives (FN)**: Missed annotations -- the user-drawn boxes with estimated metadata
- **Misclassifications (MC)**: Reclassified detections -- detection_meta + correct category

#### Loss Function

```python
def compute_loss(params, calibration_set, weights=None):
    """Evaluate a parameter vector against the calibration set.

    Does NOT re-run the full detector. Instead, replays each calibration
    entry through the classification logic using the stored detection metadata
    and the candidate parameter vector.

    Args:
        params: Dict of parameter values to evaluate
        calibration_set: List of (detection_meta, true_label, detector_label) tuples
        weights: Dict with 'fp', 'fn', 'mc' weights (default: 1.0 each)

    Returns:
        Weighted loss (lower is better)
    """
    if weights is None:
        weights = {'fp': 1.0, 'fn': 2.0, 'mc': 0.5}  # FN penalized more

    fp_count = 0
    fn_count = 0
    mc_count = 0

    for meta, true_label, detector_label in calibration_set:
        # Simulate classification decision with candidate params
        would_detect, predicted_label = simulate_classify(meta, params)

        if true_label is None:
            # Should NOT have been detected (FP in calibration)
            if would_detect:
                fp_count += 1
        elif detector_label is None:
            # Was missed (FN in calibration)
            if not would_detect:
                fn_count += 1
        else:
            # Was detected -- check classification
            if not would_detect:
                fn_count += 1  # Now we would miss it
            elif predicted_label != true_label:
                mc_count += 1

    return (weights['fp'] * fp_count +
            weights['fn'] * fn_count +
            weights['mc'] * mc_count)


def simulate_classify(meta, params):
    """Simulate whether a detection would pass classification with given params.

    Uses stored detection metadata (contrast, brightness, length, etc.)
    to replay the decision tree from classify_trail() without running
    the full detection pipeline.
    """
    contrast = meta.get('contrast_ratio', 1.0)
    length = meta.get('length', 0)
    brightness = meta.get('avg_brightness', 0)
    brightness_std = meta.get('brightness_std', 0)
    saturation = meta.get('avg_saturation', 0)
    has_dotted = meta.get('has_dotted_pattern', False)

    # Would the contrast threshold reject it?
    if contrast < params.get('satellite_contrast_min', 1.08):
        return False, None

    # Would the length range reject it?
    if length < params.get('satellite_min_length', 60):
        return False, None
    if length > params.get('satellite_max_length', 1400):
        return False, None

    # Airplane vs satellite classification
    is_bright = brightness > params.get('airplane_brightness_min', 45)
    is_colorful = saturation > params.get('airplane_saturation_min', 2)

    if has_dotted and is_bright:
        return True, 'airplane'

    is_dim = brightness < params.get('airplane_brightness_min', 45)
    brightness_var = brightness_std / (brightness + 1e-5)
    is_smooth = brightness_var < 0.40

    if is_dim and is_smooth:
        return True, 'satellite'

    return False, None
```

#### Coordinate-Wise Golden Section Search

```python
def optimize_parameters(current_params, calibration_set, safety_bounds,
                        max_iterations=3, tol=0.01):
    """Optimize parameters by cycling through each one with golden section search.

    Args:
        current_params: Starting parameter dict
        calibration_set: From the review session
        safety_bounds: {param: (min, max)} hard limits
        max_iterations: Number of full cycles through all parameters
        tol: Convergence tolerance (relative improvement)

    Returns:
        Optimized parameter dict
    """
    params = dict(current_params)
    best_loss = compute_loss(params, calibration_set)

    # Golden ratio
    phi = (1 + 5 ** 0.5) / 2
    resphi = 2 - phi  # 0.382

    optimizable_params = [
        'satellite_contrast_min', 'satellite_min_length', 'satellite_max_length',
        'canny_low', 'canny_high', 'hough_threshold', 'min_line_length',
        'max_line_gap', 'brightness_threshold', 'airplane_brightness_min',
        'airplane_saturation_min', 'min_aspect_ratio', 'mf_snr_threshold',
    ]

    for iteration in range(max_iterations):
        improved = False

        for param_name in optimizable_params:
            if param_name not in params or param_name not in safety_bounds:
                continue

            lo, hi = safety_bounds[param_name]

            # Golden section search on this parameter
            a, b = lo, hi
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)

            test_params = dict(params)

            test_params[param_name] = x1
            f1 = compute_loss(test_params, calibration_set)

            test_params[param_name] = x2
            f2 = compute_loss(test_params, calibration_set)

            for _ in range(15):  # ~15 iterations gives <0.1% precision
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + resphi * (b - a)
                    test_params[param_name] = x1
                    f1 = compute_loss(test_params, calibration_set)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = b - resphi * (b - a)
                    test_params[param_name] = x2
                    f2 = compute_loss(test_params, calibration_set)

            optimal = (a + b) / 2
            test_params[param_name] = optimal
            new_loss = compute_loss(test_params, calibration_set)

            if new_loss < best_loss:
                params[param_name] = optimal
                best_loss = new_loss
                improved = True

        if not improved:
            break

    return params
```

### Safety Bounds

Parameters have hard min/max bounds to prevent catastrophic drift:

```python
PARAMETER_SAFETY_BOUNDS = {
    'satellite_contrast_min':   (1.01,  1.30),
    'satellite_min_length':     (20,    200),
    'satellite_max_length':     (300,   2000),
    'canny_low':                (1,     30),
    'canny_high':               (20,    150),
    'hough_threshold':          (10,    80),
    'min_line_length':          (15,    120),
    'max_line_gap':             (10,    100),
    'brightness_threshold':     (5,     50),
    'airplane_brightness_min':  (20,    150),
    'airplane_saturation_min':  (1,     25),
    'min_aspect_ratio':         (2,     8),
    'mf_snr_threshold':         (1.5,   5.0),
    'mf_sigma_perp':            (0.3,   2.0),
}
```

These bounds are derived from:
- Physical constraints (contrast ratios > 1.0, lengths > 0)
- Empirical limits (no satellite trail is shorter than 20px at 1080p)
- Safety margins (prevent settings where everything is detected or nothing is)

### Persistence Format

Learned parameters are stored in two places:

1. **In the annotation JSON** (`learned_parameters` field) -- tied to a specific video/session
2. **In a standalone profile file** -- persists across videos

Profile file: `~/.mnemosky/learned_params.json`

```json
{
    "version": 1,
    "profiles": {
        "default": {
            "base_sensitivity": "high",
            "parameters": {
                "satellite_contrast_min": 1.04,
                "...": "..."
            },
            "calibration_stats": {
                "total_corrections": 47,
                "sessions_count": 3,
                "last_loss": 0.15,
                "videos_processed": ["video1.mp4", "video2.mp4", "video3.mp4"]
            },
            "last_updated": "2026-02-15T14:42:10Z"
        }
    },
    "active_profile": "default"
}
```

Multiple profiles allow different camera/sky configurations. The `--hitl-profile` CLI argument selects which profile to load.

### Anti-Drift Mechanisms

1. **Safety bounds**: Hard-coded min/max per parameter (see above)
2. **Decay learning rate**: EMA alpha decreases as corrections accumulate, preventing late corrections from drastically changing well-tuned parameters
3. **Reversion safety**: If Tier 2 optimization produces a loss that is >20% worse than the current parameters on the calibration set, the optimization result is discarded
4. **Profile snapshots**: Every session stores a snapshot of parameters_before and parameters_after, enabling manual reversion
5. **Minimum calibration size**: Tier 2 optimization only runs when the calibration set has at least 10 entries (prevents overfitting to 2-3 corrections)

---

## 5. Integration Plan

### New Classes and Their Locations

All new code lives in `satellite_trail_detector.py` (single-file architecture is maintained).

#### Class 1: `AnnotationDatabase` (~200 lines)

**Location**: Insert after `TemporalFrameBuffer` class (around line 1200), before `BaseDetectionAlgorithm`.

```python
class AnnotationDatabase:
    """COCO-compatible annotation database with correction tracking.

    Manages detection annotations, human corrections, and session metadata.
    Supports loading/saving to JSON and export to pure COCO format.
    """

    def __init__(self, path=None):
        """Load existing database or create empty one."""

    def add_image(self, frame_index, video_source, width, height) -> int:
        """Register a frame image, return image_id."""

    def add_detection(self, image_id, category, bbox_xyxy, detection_info,
                      params_snapshot, confidence) -> int:
        """Add a detector-produced annotation, return annotation_id.
        Converts from internal (x_min, y_min, x_max, y_max) to COCO (x, y, w, h)."""

    def add_missed(self, image_id, category, bbox_xyxy, estimated_meta=None) -> int:
        """Add a user-marked missed detection, return missed_annotation_id."""

    def record_correction(self, annotation_id, action,
                          new_category=None, new_bbox=None, notes=None):
        """Record a human correction action."""

    def get_calibration_set(self) -> list:
        """Build calibration set for the learning loop.
        Returns list of (detection_meta, true_label, detector_label) tuples."""

    def get_pending_annotations(self, image_id=None) -> list:
        """Get annotations awaiting review, sorted by confidence (low first)."""

    def get_frames_needing_review(self) -> list:
        """Get image_ids sorted by minimum detection confidence."""

    def start_session(self, video_source, sensitivity, algorithm, params):
        """Begin a new review session."""

    def end_session(self, params_after):
        """Finalize current session with post-learning parameters."""

    def save(self, path=None):
        """Write database to JSON file."""

    def export_coco(self, path):
        """Export pure COCO format (confirmed annotations only)."""

    def undo_last_correction(self) -> bool:
        """Undo the most recent correction. Returns True if successful."""
```

#### Class 2: `ParameterAdapter` (~150 lines)

**Location**: Insert immediately after `AnnotationDatabase`.

```python
class ParameterAdapter:
    """Adapts detection parameters from human corrections.

    Tier 1: Immediate EMA adaptation per correction.
    Tier 2: Batch coordinate-wise optimization over calibration set.
    """

    def __init__(self, initial_params, safety_bounds=None):
        """Initialize with starting parameters and safety bounds."""

    def apply_correction(self, correction_action, trail_type, detection_meta) -> dict:
        """Tier 1: Apply single correction via EMA. Returns {param: new_value} updates."""

    def optimize_batch(self, calibration_set) -> dict:
        """Tier 2: Run coordinate-wise golden section search.
        Returns optimized parameters dict."""

    def get_params(self) -> dict:
        """Get current parameter values."""

    def save_profile(self, profile_name='default'):
        """Save learned parameters to ~/.mnemosky/learned_params.json."""

    def load_profile(self, profile_name='default') -> bool:
        """Load learned parameters from profile. Returns True if found."""

    @staticmethod
    def compute_confidence(detection_info, params) -> float:
        """Compute pseudo-confidence score for a detection."""
```

#### Class 3: `ReviewUI` (~500 lines)

**Location**: Insert after `ParameterAdapter`, before `SatelliteTrailDetector`.

```python
class ReviewUI:
    """Interactive review interface for HITL correction workflow.

    Single OpenCV window with dark theme matching the existing preview GUI.
    Displays detections for review, handles keyboard/mouse interaction,
    and feeds corrections to AnnotationDatabase and ParameterAdapter.
    """

    def __init__(self, video_path, detections_by_frame, detector,
                 annotation_db, param_adapter, auto_accept_threshold=0.9):
        """
        Args:
            video_path: Path to input video
            detections_by_frame: Dict[frame_idx -> list of (trail_type, detection_info)]
            detector: SatelliteTrailDetector instance (for re-running detection after param update)
            annotation_db: AnnotationDatabase instance
            param_adapter: ParameterAdapter instance
            auto_accept_threshold: Confidence above which detections are auto-accepted
        """

    def run(self):
        """Main event loop. Returns when user quits."""

    def _render_frame(self):
        """Render current frame with overlays, sidebar, and status bar."""

    def _render_sidebar(self):
        """Draw detection cards, session stats, and parameter display."""

    def _render_status_bar(self):
        """Draw bottom bar with frame slider, buttons."""

    def _handle_key(self, key):
        """Process keyboard input."""

    def _handle_mouse(self, event, x, y, flags, param):
        """Process mouse input (selection, bbox drawing, slider)."""

    def _accept_detection(self, annotation_id):
        """Accept the selected detection."""

    def _reject_detection(self, annotation_id):
        """Reject the selected detection."""

    def _reclassify_detection(self, annotation_id, new_category):
        """Reclassify detection to different category."""

    def _mark_missed(self, bbox, category):
        """Add a user-drawn missed detection."""

    def _navigate_to_frame(self, frame_idx):
        """Seek video to frame and load detections."""

    def _next_unreviewed_frame(self):
        """Jump to next frame with low-confidence unreviewed detections."""

    def _run_learning(self):
        """Trigger Tier 2 batch optimization and update detector."""

    def _undo(self):
        """Undo last correction."""
```

### New CLI Arguments

Add to `main()` argument parser:

```python
# --- HITL Review Mode ---
parser.add_argument(
    '--review',
    action='store_true',
    help='Run in HITL review mode: process video, then open interactive '
         'review UI for correcting detections and learning parameters'
)

parser.add_argument(
    '--review-only',
    action='store_true',
    help='Open review UI on an existing annotation file without re-processing '
         'the video. Requires --annotations.'
)

parser.add_argument(
    '--annotations',
    type=str,
    default=None,
    help='Path to annotation JSON file. Used with --review-only to load '
         'existing annotations, or with --review to specify output path.'
)

parser.add_argument(
    '--hitl-profile',
    type=str,
    default='default',
    help='Name of the learned parameter profile to load/save '
         '(stored in ~/.mnemosky/learned_params.json). Default: "default"'
)

parser.add_argument(
    '--auto-accept',
    type=float,
    default=0.9,
    help='Confidence threshold for auto-accepting detections in review mode. '
         'Detections with confidence above this value are pre-accepted. '
         'Set to 1.0 to disable auto-accept. Default: 0.9'
)

parser.add_argument(
    '--no-learn',
    action='store_true',
    help='Disable parameter learning in review mode (corrections are still '
         'saved to the annotation database, but parameters are not updated)'
)
```

### Integration with `process_video()`

The review mode extends the existing processing pipeline:

```python
def process_video(..., review_mode=False, review_only=False,
                  annotations_path=None, hitl_profile='default',
                  auto_accept=0.9, no_learn=False):

    # ... existing video processing code ...

    # === HITL Review Mode ===
    if review_mode or review_only:
        # 1. Load or create annotation database
        if annotations_path:
            ann_path = Path(annotations_path)
        else:
            ann_path = Path(output_path).with_suffix('.json')

        if review_only and ann_path.exists():
            ann_db = AnnotationDatabase(ann_path)
        else:
            ann_db = AnnotationDatabase()

        # 2. Initialize parameter adapter
        param_adapter = ParameterAdapter(
            detector.params,
            PARAMETER_SAFETY_BOUNDS
        )

        # Try loading learned profile
        if hitl_profile:
            param_adapter.load_profile(hitl_profile)

        # 3. If not review_only, populate annotations from detection results
        if not review_only:
            # detections_by_frame was collected during processing
            ann_db.start_session(str(input_path), sensitivity,
                                algorithm, detector.params)

            for frame_idx, detections in detections_by_frame.items():
                img_id = ann_db.add_image(frame_idx, str(input_path),
                                          width, height)
                for trail_type, det_info in detections:
                    cat_id = 0 if trail_type == 'satellite' else 1
                    confidence = ParameterAdapter.compute_confidence(
                        det_info, detector.params)
                    ann_db.add_detection(img_id, cat_id, det_info['bbox'],
                                        det_info, detector.params, confidence)

        # 4. Launch review UI (MUST be sequential -- no parallel workers)
        review_ui = ReviewUI(
            str(input_path), detections_by_frame, detector,
            ann_db, param_adapter if not no_learn else None,
            auto_accept_threshold=auto_accept
        )
        review_ui.run()

        # 5. After review: save annotations and learned parameters
        ann_db.end_session(param_adapter.get_params() if not no_learn else None)
        ann_db.save(ann_path)

        if not no_learn and hitl_profile:
            param_adapter.save_profile(hitl_profile)

        print(f"\nAnnotations saved to: {ann_path}")
        if not no_learn:
            print(f"Learned parameters saved to profile: {hitl_profile}")
```

### Interaction with Parallel Workers

**Critical constraint**: The review UI **must run sequentially** (single-threaded). The existing parallel worker pool (multiprocessing.Pool) is used only during the detection phase and is shut down before the review UI launches.

The flow is:
1. **Detection phase** (parallel OK): Process video frames, collect detections
2. **Pool shutdown**: `pool.close(); pool.join()`
3. **Review phase** (sequential): OpenCV window with mouse/keyboard callbacks
4. **Learning phase** (sequential): Parameter optimization runs after review

If the user triggers re-detection after parameter updates (pressing `L` in review mode), it runs sequentially on the current frame only -- not the entire video. A full re-process requires re-running the CLI.

### Collecting Detections for Review

During `process_video()`, when `review_mode=True`, the detection loop must store all detections:

```python
# At the start of process_video, when review_mode is True:
detections_by_frame = {} if review_mode else None

# Inside the frame processing loop:
for frame, fc, detected_trails, debug_info in _frame_results():
    # ... existing processing ...

    if detections_by_frame is not None:
        detections_by_frame[fc] = detected_trails
```

The frames themselves are re-read from the video by `ReviewUI` (using `cv2.VideoCapture.set(cv2.CAP_PROP_POS_FRAMES, idx)`) rather than storing all frames in memory.

### How Learned Parameters Feed Back into the Detector

When the user presses `L` (learn) or quits the review UI:

1. `ParameterAdapter.optimize_batch()` runs coordinate-wise optimization
2. The optimized parameters are written back to `detector.params`
3. For the next video processed with the same `--hitl-profile`, `main()` loads the profile and passes it as `preprocessing_params` overrides:

```python
# In main(), before calling process_video():
if args.hitl_profile:
    adapter = ParameterAdapter({})
    if adapter.load_profile(args.hitl_profile):
        # Merge learned params into preprocessing_params
        if preprocessing_params is None:
            preprocessing_params = {}
        learned = adapter.get_params()
        for key in ['canny_low', 'canny_high']:
            if key in learned:
                preprocessing_params[key] = learned[key]
        # Other params applied via signal_envelope or direct detector.params override
```

### Summary of New File Additions

| Component | Location in satellite_trail_detector.py | Approx Lines |
|-----------|---------------------------------------|--------------|
| `PARAMETER_SAFETY_BOUNDS` dict | After TemporalFrameBuffer, ~line 1200 | 20 |
| `CORRECTION_RULES` dict | After PARAMETER_SAFETY_BOUNDS | 80 |
| `AnnotationDatabase` class | After CORRECTION_RULES | 200 |
| `ParameterAdapter` class | After AnnotationDatabase | 150 |
| `ReviewUI` class | After ParameterAdapter | 500 |
| CLI arguments | In `main()`, after existing args | 40 |
| `process_video()` review integration | At end of process_video | 60 |
| `main()` profile loading | Before process_video call | 20 |
| **Total new code** | | **~1070 lines** |

### Implementation Order

1. **AnnotationDatabase** -- No dependencies, testable independently
2. **ParameterAdapter** -- Depends on CORRECTION_RULES and PARAMETER_SAFETY_BOUNDS
3. **ReviewUI** -- Depends on AnnotationDatabase and ParameterAdapter
4. **CLI integration** -- Depends on all three classes
5. **Testing** -- End-to-end test with sample video

### Data Flow Diagram

```
                    +------------------+
                    |   Input Video    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  process_video() |
                    |  (parallel OK)   |
                    +--------+---------+
                             |
                    detections_by_frame
                             |
                    +--------v---------+
                    | AnnotationDatabase|
                    | (populate from   |
                    |  detections)     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |    ReviewUI      |
                    | (sequential,     |
                    |  interactive)    |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                                 |
   +--------v---------+            +----------v--------+
   | ParameterAdapter |            | AnnotationDatabase|
   | Tier 1: EMA per  |            | (save corrections)|
   |   correction     |            +-------------------+
   | Tier 2: batch    |
   |   optimization   |
   +--------+---------+
            |
   +--------v---------+
   | learned_params    |
   | .json profile     |
   +-------------------+
            |
            v
   (loaded on next run via --hitl-profile)
```
