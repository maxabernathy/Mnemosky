"""
Satellite and Airplane Trail Detector for MP4 Videos
====================================================
Detects and classifies trails in video frames as either satellites or airplanes.

Detection Criteria (optimized for 1920x1080 resolution):
- Satellites: SMOOTH trails, 180-300 pixels, dim, monochromatic, uniform brightness (GOLD dotted boxes)
- Airplanes: DOTTED trails with bright point features, sometimes colorful (red/green/white navigation lights)
             Can be any length including 180-300px (ORANGE dotted boxes)

Key Distinction: Airplanes show distinct bright point-like lights along the trail, while satellites have
smooth, consistent brightness.

For each detected trail, freezes only the highlighted region inside the dotted bounding box for 1 second,
while the rest of the frame continues playing normally. Output video preserves the same quality and resolution as the input.

Usage:
    python satellite_trail_detector.py input.mp4 output.mp4
    python satellite_trail_detector.py input.mp4 output.mp4 --sensitivity high
    python satellite_trail_detector.py input.mp4 output.mp4 --freeze-duration 2.0
    python satellite_trail_detector.py input.mp4 output.mp4 --max-duration 30
    python satellite_trail_detector.py input.mp4 output.mp4 --detect-type satellites
    python satellite_trail_detector.py input.mp4 output.mp4 --detect-type airplanes
    python satellite_trail_detector.py input.mp4 output.mp4 --no-labels
    python satellite_trail_detector.py input.mp4 output.mp4 --debug
    python satellite_trail_detector.py input.mp4 output.mp4 --debug-only

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import argparse
import sys
import time
import os
import json
import math
import random
import tempfile
import multiprocessing
from pathlib import Path

from collections import deque
from datetime import datetime, timezone

__version__ = '0.2.0-sts'

try:
    from scipy.ndimage import maximum_filter as _scipy_maximum_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    _HAS_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

# CUDA median filter availability (requires OpenCV contrib with CUDA support)
try:
    _HAS_CUDA_MEDIAN = _HAS_CUDA and hasattr(cv2.cuda, 'createMedianFilter')
except Exception:
    _HAS_CUDA_MEDIAN = False

# ── Neural network backend availability (lazy — import deferred to first use) ─
_NN_BACKENDS_CHECKED = {}  # name -> bool (cached after first probe)


def _check_nn_backend(name):
    """Check if a neural network backend is importable (cached)."""
    if name in _NN_BACKENDS_CHECKED:
        return _NN_BACKENDS_CHECKED[name]
    available = False
    if name == 'ultralytics':
        try:
            from ultralytics import YOLO  # noqa: F401
            available = True
        except ImportError:
            pass
    elif name == 'cv2dnn':
        available = hasattr(cv2, 'dnn')
    elif name == 'onnxruntime':
        try:
            import onnxruntime  # noqa: F401
            available = True
        except ImportError:
            pass
    _NN_BACKENDS_CHECKED[name] = available
    return available


def _ensure_nn_backend(name):
    """Ensure backend is available; attempt auto-install for pip packages.

    Returns True if the backend is ready to use, False otherwise.
    Prints user-friendly install instructions on failure.
    """
    if _check_nn_backend(name):
        return True
    # Attempt auto-install for pip-installable backends
    pkg_map = {
        'ultralytics': 'ultralytics',
        'onnxruntime': 'onnxruntime',
    }
    pkg = pkg_map.get(name)
    if pkg is None:
        print(f"Error: backend '{name}' is not available and cannot be auto-installed.")
        return False
    print(f"Backend '{name}' not found. Attempting: pip install {pkg} ...")
    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', pkg, '-q'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Re-check
        _NN_BACKENDS_CHECKED.pop(name, None)
        if _check_nn_backend(name):
            print(f"  Installed '{pkg}' successfully.")
            return True
    except Exception as e:
        print(f"  Auto-install failed: {e}")
    print(f"Please install manually:  pip install {pkg}")
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Application-wide config  (~/.mnemosky/config.json)
# ═══════════════════════════════════════════════════════════════════════════════

_MNEMOSKY_DIR = Path.home() / '.mnemosky'

_DEFAULT_CONFIG = {
    'version': 1,
    'algorithms': {
        'default': {
            'sensitivity': 'medium',
        },
        'radon': {
            'sensitivity': 'medium',
            'radon_snr_threshold': 3.0,
            'pcf_ratio_threshold': 2.0,
            'star_mask_sigma': 5.0,
            'lsd_log_eps': 1.0,
            'pcf_kernel_length': 31,
            'satellite_min_length': 50,
        },
        'nn': {
            'model_path': None,
            'backend': 'ultralytics',
            'confidence': 0.25,
            'nms_iou': 0.45,
            'device': 'auto',
            'class_map': {'satellite': [0], 'airplane': [1]},
            'input_size': 640,
            'half_precision': False,
        },
    },
}

_CONFIG_PATH = _MNEMOSKY_DIR / 'config.json'


def load_config(path=None):
    """Load application config from JSON, merging with defaults.

    Missing keys are filled from ``_DEFAULT_CONFIG``.  Unknown keys
    in the file are preserved (forward-compatible).

    Args:
        path: Optional override for config file path.
              Default: ``~/.mnemosky/config.json``.

    Returns:
        Config dict (deep copy — safe to mutate).
    """
    import copy
    config = copy.deepcopy(_DEFAULT_CONFIG)
    fpath = Path(path) if path else _CONFIG_PATH
    if fpath.exists():
        try:
            with open(fpath, 'r') as f:
                user = json.load(f)
            # Deep merge: user values override defaults
            _deep_merge(config, user)
        except Exception as e:
            print(f"Warning: could not load config {fpath}: {e}")
    return config


def save_config(config, path=None):
    """Save config dict to JSON.

    Creates ``~/.mnemosky/`` directory if needed.

    Args:
        config: Config dict to save.
        path: Optional override for config file path.
    """
    fpath = Path(path) if path else _CONFIG_PATH
    fpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(fpath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: could not save config {fpath}: {e}")


def _deep_merge(base, override):
    """Recursively merge *override* into *base* (mutates *base*)."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# Neural network inference backend  (_NNBackend)
# ═══════════════════════════════════════════════════════════════════════════════

class _NNBackend:
    """Unified interface for neural network inference backends.

    Wraps ultralytics, cv2.dnn, and onnxruntime behind a single
    ``predict(frame)`` interface.  The backend is selected at construction
    time; the actual heavy import happens once (lazy), and the model is
    loaded from disk immediately.

    Attributes:
        model_path: Resolved absolute path to the model file.
        backend: Backend name ('ultralytics', 'cv2dnn', 'onnxruntime').
        confidence: Detection confidence threshold (0-1).
        nms_iou: NMS IoU threshold (0-1).
        input_size: Square input resolution for the model.
        class_names: Dict of {class_id: class_name} from the model metadata.
    """

    def __init__(self, model_path, backend='ultralytics', device='auto',
                 confidence=0.25, nms_iou=0.45, input_size=640,
                 half_precision=False, no_gpu=False):
        if model_path is None:
            raise ValueError("model_path is required for neural network backend")
        self.model_path = str(Path(model_path).resolve())
        self.backend = backend
        self.device = 'cpu' if no_gpu else device
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.input_size = input_size
        self.half_precision = half_precision and self.device != 'cpu'
        self.class_names = {}
        self._model = None

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not _ensure_nn_backend(backend):
            raise RuntimeError(f"Backend '{backend}' is not available.")

        self._load_model()

    # ── Model loading ────────────────────────────────────────────────

    def _load_model(self):
        """Load model using the configured backend."""
        if self.backend == 'ultralytics':
            self._load_ultralytics()
        elif self.backend == 'cv2dnn':
            self._load_cv2dnn()
        elif self.backend == 'onnxruntime':
            self._load_onnxruntime()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_ultralytics(self):
        from ultralytics import YOLO
        self._model = YOLO(self.model_path)
        # Extract class names from model metadata
        if hasattr(self._model, 'names') and self._model.names:
            self.class_names = dict(self._model.names)

    def _load_cv2dnn(self):
        ext = Path(self.model_path).suffix.lower()
        if ext == '.onnx':
            self._model = cv2.dnn.readNetFromONNX(self.model_path)
        elif ext in ('.pb', '.pbtxt'):
            self._model = cv2.dnn.readNetFromTensorflow(self.model_path)
        elif ext in ('.cfg', '.weights'):
            self._model = cv2.dnn.readNetFromDarknet(self.model_path)
        elif ext == '.caffemodel':
            self._model = cv2.dnn.readNetFromCaffe(self.model_path)
        else:
            # Default: try ONNX
            self._model = cv2.dnn.readNetFromONNX(self.model_path)
        # GPU backend
        if self.device != 'cpu' and _HAS_CUDA:
            self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.class_names = {}  # cv2.dnn has no metadata API; set via class_map

    def _load_onnxruntime(self):
        import onnxruntime as ort
        providers = ['CPUExecutionProvider']
        if self.device != 'cpu':
            cuda_available = 'CUDAExecutionProvider' in ort.get_available_providers()
            if cuda_available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self._model = ort.InferenceSession(self.model_path, providers=providers)
        # Extract class names from ONNX metadata if available
        meta = self._model.get_modelmeta()
        if meta and meta.custom_metadata_map:
            names_str = meta.custom_metadata_map.get('names', '')
            if names_str:
                try:
                    self.class_names = json.loads(
                        names_str.replace("'", '"'))
                except Exception:
                    pass

    # ── Inference ────────────────────────────────────────────────────

    def predict(self, frame):
        """Run inference on a BGR frame.

        Returns:
            List of dicts, each with:
                'bbox': (x_min, y_min, x_max, y_max) in pixel coordinates
                'class_id': int (model's raw class ID)
                'class_name': str (model's class label, or '')
                'confidence': float (0-1)
        """
        if self.backend == 'ultralytics':
            return self._predict_ultralytics(frame)
        elif self.backend == 'cv2dnn':
            return self._predict_cv2dnn(frame)
        elif self.backend == 'onnxruntime':
            return self._predict_onnxruntime(frame)
        return []

    def _predict_ultralytics(self, frame):
        results = self._model(
            frame, conf=self.confidence, iou=self.nms_iou,
            imgsz=self.input_size, device=self.device,
            half=self.half_precision, verbose=False)
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    detections.append({
                        'bbox': (float(x1), float(y1), float(x2), float(y2)),
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, ''),
                        'confidence': conf,
                    })
        return detections

    def _predict_cv2dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False)
        self._model.setInput(blob)
        out_names = self._model.getUnconnectedOutLayersNames()
        outputs = self._model.forward(out_names)
        return self._parse_yolo_output(outputs, w, h)

    def _predict_onnxruntime(self, frame):
        h, w = frame.shape[:2]
        # Preprocess: resize, normalize, HWC→CHW, add batch
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        if self.half_precision:
            img = img.astype(np.float16)
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: img})
        return self._parse_yolo_output(outputs, w, h)

    def _parse_yolo_output(self, outputs, orig_w, orig_h):
        """Parse raw YOLO output tensor(s) into detection dicts.

        Handles the two common YOLO output layouts:
          - YOLOv8 transposed: (1, 4+num_classes, num_detections)
          - YOLOv5/legacy:     (1, num_detections, 5+num_classes)
        """
        detections = []
        output = outputs[0]
        if output.ndim == 3:
            output = output[0]  # drop batch dim
        rows, cols = output.shape

        # Detect layout: if cols > rows → transposed YOLOv8 format
        if cols > rows:
            output = output.T
            rows, cols = output.shape

        # Determine if layout has objectness score (YOLOv5: 5+C) or not (YOLOv8: 4+C)
        # YOLOv8: cols = 4 + num_classes (no objectness)
        # YOLOv5: cols = 5 + num_classes (col 4 is objectness)
        has_obj = cols >= 6 and (cols - 5) >= 1
        if has_obj:
            # YOLOv5 layout: [cx, cy, w, h, obj_conf, cls0, cls1, ...]
            num_classes = cols - 5
        else:
            # YOLOv8 layout: [cx, cy, w, h, cls0, cls1, ...]
            num_classes = cols - 4

        if num_classes <= 0:
            return detections

        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        boxes, confidences, class_ids = [], [], []

        for i in range(rows):
            if has_obj:
                obj_conf = output[i, 4]
                cls_scores = output[i, 5:5 + num_classes]
                scores = cls_scores * obj_conf
            else:
                scores = output[i, 4:4 + num_classes]

            max_score = float(np.max(scores))
            if max_score < self.confidence:
                continue

            cls_id = int(np.argmax(scores))
            cx, cy, bw, bh = output[i, :4]
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y

            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            confidences.append(max_score)
            class_ids.append(cls_id)

        # NMS
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
                                       self.nms_iou)
            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    bx, by, bw, bh = boxes[idx]
                    detections.append({
                        'bbox': (float(bx), float(by),
                                 float(bx + bw), float(by + bh)),
                        'class_id': class_ids[idx],
                        'class_name': self.class_names.get(class_ids[idx], ''),
                        'confidence': confidences[idx],
                    })

        return detections

    def get_model_info(self):
        """Return dict with model metadata for display."""
        return {
            'model_path': self.model_path,
            'backend': self.backend,
            'device': self.device,
            'input_size': self.input_size,
            'class_names': self.class_names,
            'half_precision': self.half_precision,
        }


# ── Documentation overlay shared infrastructure ──────────────────────────
# Used by all three preview windows (preprocessing, radon, NN) to provide
# interactive multi-page documentation accessed via easter eggs.

_DOC_SECRET_WORD = "blackbox"

# Preprocessing preview doc pages
_PREPROC_DOC_PAGES = [
    {
        'title': 'The Pipeline',
        'content': [
            ('Frame -> Grayscale -> CLAHE -> Blur -> Canny -> Hough -> Classify', 'diagram'),
            ('', 'blank'),
            ('Each step transforms the image to extract satellite and airplane trails:', 'primary'),
            ('', 'blank'),
            ('Grayscale      Convert colour frame to single-channel intensity', 'primary'),
            ('CLAHE          Boost dim features without blowing out bright stars', 'primary'),
            ('Blur           Smooth noise while preserving linear edges', 'primary'),
            ('Canny          Detect edges (brightness transitions)', 'primary'),
            ('Hough          Find straight lines in the edge map', 'primary'),
            ('Classify       Brightness, colour, smoothness, contrast analysis', 'primary'),
            ('', 'blank'),
            ('A supplementary Matched Filter stage catches trails too dim for edges.', 'dim'),
            ('Each panel below shows one of these intermediate stages.', 'dim'),
        ],
        'konami_only': False,
    },
    {
        'title': 'The Panels',
        'content': [
            ('ORIGINAL', 'accent'),
            ('  Your raw input frame. Click start+end to mark trail examples.', 'primary'),
            ('  Marked trails build a signal envelope that adapts thresholds.', 'dim'),
            ('', 'blank'),
            ('CLAHE', 'accent'),
            ('  Contrast-Limited Adaptive Histogram Equalization.', 'primary'),
            ('  Boosts dim features in dark sky without saturating bright stars.', 'primary'),
            ('  The clip limit controls enhancement aggressiveness.', 'dim'),
            ('', 'blank'),
            ('MF RESPONSE', 'accent'),
            ('  Directional matched filter SNR heatmap. Bright = linear features.', 'primary'),
            ('  Magenta lines = detections above the SNR threshold.', 'primary'),
            ('  Uses temporal reference (multi-frame) when available.', 'dim'),
            ('', 'blank'),
            ('EDGES', 'accent'),
            ('  Canny edge detection. Cyan overlay = edges fed into Hough lines.', 'primary'),
        ],
        'konami_only': False,
    },
    {
        'title': 'The Sliders',
        'content': [
            ('CLAHE Clip', 'accent'),
            ('  How aggressively to boost contrast.', 'primary'),
            ('  Lower = subtle  |  Higher = punchy but may amplify noise', 'dim'),
            ('', 'blank'),
            ('Blur Kernel / Sigma', 'accent'),
            ('  Gaussian smoothing reduces noise and small-scale texture.', 'primary'),
            ('  Higher = smoother = fewer false edges, but may blur thin trails', 'dim'),
            ('', 'blank'),
            ('Canny Low / High', 'accent'),
            ('  Edge detection sensitivity. Two thresholds with hysteresis.', 'primary'),
            ('  Lower = more edges = more candidates (and more noise)', 'dim'),
            ('', 'blank'),
            ('MF SNR', 'accent'),
            ('  Matched filter signal-to-noise threshold.', 'primary'),
            ('  Lower = catches dimmer trails but more false positives', 'dim'),
            ('  Higher = only bright, obvious trails pass', 'dim'),
        ],
        'konami_only': False,
    },
]

# Radon preview doc pages
_RADON_DOC_PAGES = [
    {
        'title': 'The Radon Pipeline',
        'content': [
            ('Frame -> BG Sub -> Star Mask -> [LSD + Radon] -> PCF -> Merge', 'diagram'),
            ('', 'blank'),
            ('Two independent detection paths complement each other:', 'primary'),
            ('', 'blank'),
            ('LSD Path (Line Segment Detector)', 'accent'),
            ('  A-contrario algorithm finding statistically significant segments.', 'primary'),
            ('  Fast, good for brighter trails with clear edges.', 'dim'),
            ('', 'blank'),
            ('Radon Path (Integral Transform)', 'accent'),
            ('  Projects the image at N angles. A straight line concentrates', 'primary'),
            ('  into a single bright peak in the sinogram. Catches very dim', 'primary'),
            ('  trails by integrating signal along full trail length.', 'primary'),
            ('', 'blank'),
            ('Both paths feed into the Perpendicular Cross Filter (PCF) which', 'dim'),
            ('rejects false positives via trail cross-section asymmetry.', 'dim'),
        ],
        'konami_only': False,
    },
    {
        'title': 'The Panels',
        'content': [
            ('ORIGINAL', 'accent'),
            ('  Input frame with detection overlays.', 'primary'),
            ('', 'blank'),
            ('RESIDUAL', 'accent'),
            ('  Star-cleaned residual. Red = masked star pixels.', 'primary'),
            ('  Only trail signal remains after BG subtraction + star removal.', 'dim'),
            ('', 'blank'),
            ('SINOGRAM', 'accent'),
            ('  Radon transform output. Each column = one projection angle.', 'primary'),
            ('  Bright spots = linear features. Circles = detected peaks.', 'primary'),
            ('', 'blank'),
            ('LSD LINES', 'accent'),
            ('  Line Segment Detector output. Green = significant segments.', 'primary'),
            ('  Uses the Helmholtz principle (unlikely from noise alone).', 'dim'),
            ('', 'blank'),
            ('DETECTIONS', 'accent'),
            ('  Final result: Green=PCF ok | Amber=raw Radon | Red=rejected', 'primary'),
        ],
        'konami_only': False,
    },
    {
        'title': 'The Sliders',
        'content': [
            ('Radon SNR', 'accent'),
            ('  Min signal-to-noise ratio in sinogram. Lower = more sensitive.', 'primary'),
            ('', 'blank'),
            ('PCF Ratio', 'accent'),
            ('  Cross-filter strictness. Real trails are brighter along their', 'primary'),
            ('  length than across. Higher = stricter FP rejection.', 'dim'),
            ('', 'blank'),
            ('Star Mask sigma', 'accent'),
            ('  Star removal aggressiveness (in noise sigma units).', 'primary'),
            ('  Lower = masks more stars but may eat trail pixels near stars.', 'dim'),
            ('', 'blank'),
            ('LSD Significance', 'accent'),
            ('  Detection threshold (log10 NFA). Lower = dimmer segments.', 'primary'),
            ('', 'blank'),
            ('PCF Kernel', 'accent'),
            ('  Cross-section sampling width. Match to trail PSF width.', 'primary'),
            ('', 'blank'),
            ('Min Length', 'accent'),
            ('  Minimum trail length (px). Shorter = more trails + more noise.', 'primary'),
        ],
        'konami_only': False,
    },
    {
        'title': 'Developer Internals',
        'content': [
            ('THE RADON TRANSFORM', 'heading'),
            ('  Projects the image along N angles. A line at angle theta', 'primary'),
            ('  maps to a peak at (theta, offset) in the sinogram.', 'primary'),
            ('  sinogram[s,theta] = integral of f(x,y) along line (s,theta)', 'diagram'),
            ('', 'blank'),
            ('SNR CALCULATION', 'heading'),
            ('  SNR = sinogram / (noise_sigma * sqrt(N_pixels))', 'diagram'),
            ('  sqrt(N) from CLT: longer projections have more noise,', 'primary'),
            ('  so normalization accounts for projection length.', 'primary'),
            ('', 'blank'),
            ('PCF GEOMETRY', 'heading'),
            ('  Samples brightness parallel vs perpendicular to trail.', 'primary'),
            ('  Real trails: ratio >> 1. Stars/noise: ratio ~ 1.', 'primary'),
            ('', 'blank'),
            ('MULTI-FRAME ACCUMULATION', 'heading'),
            ('  Stacks 2-4 cleaned residuals before Radon transform.', 'primary'),
            ('  SNR boost = sqrt(N_frames). 4 frames -> 2x improvement.', 'dim'),
            ('', 'blank'),
            ('TEMPORAL TRACKER', 'heading'),
            ('  Detections must appear in >=2 of last 4 frames.', 'primary'),
            ('  Tracklets (3+ frames) get high-confidence tags.', 'dim'),
        ],
        'konami_only': True,
    },
]

# NN preview doc pages
_NN_DOC_PAGES = [
    {
        'title': 'Neural Network Detection',
        'content': [
            ('Frame -> Model Inference -> NMS -> Class Mapping -> Detections', 'diagram'),
            ('', 'blank'),
            ('Runs a trained object detection model (YOLOv8/v11, ONNX, etc.)', 'primary'),
            ('to find satellite and airplane trails in each frame.', 'primary'),
            ('', 'blank'),
            ('Three backends are supported:', 'primary'),
            ('  ultralytics   Primary. Auto-installed. GPU support.', 'accent'),
            ('  cv2dnn        OpenCV DNN. Zero extra deps. ONNX/TF.', 'accent'),
            ('  onnxruntime   ONNX Runtime. Broad HW acceleration.', 'accent'),
            ('', 'blank'),
            ('Class IDs mapped to satellite/airplane via --nn-class-map.', 'dim'),
            ('Hybrid mode (--nn-hybrid) merges NN + classical results.', 'dim'),
        ],
        'konami_only': False,
    },
    {
        'title': 'The Sliders',
        'content': [
            ('Confidence', 'accent'),
            ('  Minimum model confidence to keep a detection.', 'primary'),
            ('  Lower = more detections (recall) but more false positives.', 'dim'),
            ('  Higher = only high-confidence detections (precision).', 'dim'),
            ('', 'blank'),
            ('NMS IoU', 'accent'),
            ('  Non-Maximum Suppression overlap threshold.', 'primary'),
            ('  Controls how aggressively overlapping boxes are merged.', 'primary'),
            ('  Lower = more aggressive merging (fewer duplicates).', 'dim'),
            ('  Higher = less merging (may keep overlapping boxes).', 'dim'),
        ],
        'konami_only': False,
    },
]


def _draw_doc_overlay(canvas, page_index, pages, konami_unlocked=False):
    """Draw interactive documentation overlay on the canvas.

    Renders a semi-transparent dark overlay with a centered content card
    showing documentation text. Supports multi-page navigation.

    Args:
        canvas: OpenCV BGR canvas (modified in-place).
        page_index: Current page (0-based).
        pages: List of page dicts (title, content, konami_only).
        konami_unlocked: Whether secret bonus pages are visible.

    Returns:
        Dict with click regions for mouse interaction:
        {
            'card': (x, y, w, h),
            'prev_arrow': (x, y, w, h) or None,
            'next_arrow': (x, y, w, h) or None,
        }
    """
    h, w = canvas.shape[:2]

    # Theme (same as preview windows)
    BG_PANEL = (42, 42, 42)
    BORDER = (58, 58, 58)
    TEXT_PRIMARY = (210, 210, 210)
    TEXT_DIM = (120, 120, 120)
    ACCENT = (200, 255, 80)
    ACCENT_DIM = (100, 170, 50)

    # Semi-transparent overlay (85% black)
    overlay = np.zeros_like(canvas)
    cv2.addWeighted(canvas, 0.15, overlay, 0.85, 0, canvas)

    # Filter visible pages
    visible = [p for p in pages
               if not p.get('konami_only') or konami_unlocked]
    n_pages = len(visible)
    if n_pages == 0:
        return {'card': (0, 0, 0, 0), 'prev_arrow': None, 'next_arrow': None}
    page_index = max(0, min(page_index, n_pages - 1))
    page = visible[page_index]

    # Content card dimensions
    card_w = int(w * 0.70)
    card_h = int(h * 0.82)
    card_x = (w - card_w) // 2
    card_y = (h - card_h) // 2

    # Card background + border
    cv2.rectangle(canvas, (card_x, card_y),
                  (card_x + card_w, card_y + card_h), BG_PANEL, -1)
    cv2.rectangle(canvas, (card_x, card_y),
                  (card_x + card_w, card_y + card_h), BORDER, 1)

    # Title bar
    ty = card_y + 30
    cv2.putText(canvas, "HOW IT WORKS", (card_x + 20, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, ACCENT, 1, cv2.LINE_AA)

    # Page indicator + close hint
    pg_txt = f"{page_index + 1}/{n_pages}"
    cv2.putText(canvas, pg_txt, (card_x + card_w - 60, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ESC to close  |  A/D navigate",
                (card_x + card_w - 290, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, TEXT_DIM, 1, cv2.LINE_AA)

    # Separator
    cv2.line(canvas, (card_x + 20, ty + 10),
             (card_x + card_w - 20, ty + 10), BORDER, 1)

    # Page title
    pty = ty + 40
    cv2.putText(canvas, page['title'], (card_x + 20, pty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ACCENT, 1, cv2.LINE_AA)

    # Content lines
    cy = pty + 28
    line_h = 19
    max_cy = card_y + card_h - 60  # leave room for nav dots
    for text, style in page['content']:
        if cy > max_cy:
            break
        if style == 'blank' or text == '':
            cy += line_h // 2
            continue
        color_map = {
            'accent': ACCENT,
            'heading': ACCENT,
            'diagram': ACCENT_DIM,
            'dim': TEXT_DIM,
            'primary': TEXT_PRIMARY,
        }
        color = color_map.get(style, TEXT_PRIMARY)
        scale = 0.42 if style == 'heading' else 0.37
        thick = 1
        cv2.putText(canvas, text, (card_x + 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                    cv2.LINE_AA)
        cy += line_h

    # Navigation arrows
    arrow_y = card_y + card_h - 28
    prev_region = None
    next_region = None
    if page_index > 0:
        cv2.putText(canvas, "< PREV (A)", (card_x + 20, arrow_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT_DIM, 1,
                    cv2.LINE_AA)
        prev_region = (card_x + 10, arrow_y - 18, 120, 28)
    if page_index < n_pages - 1:
        cv2.putText(canvas, "NEXT (D) >",
                    (card_x + card_w - 120, arrow_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT_DIM, 1,
                    cv2.LINE_AA)
        next_region = (card_x + card_w - 130, arrow_y - 18, 130, 28)

    # Page dots
    dot_y = arrow_y - 22
    dot_spacing = 14
    total_w = n_pages * dot_spacing
    dot_x0 = card_x + (card_w - total_w) // 2
    for i in range(n_pages):
        dx = dot_x0 + i * dot_spacing + 6
        if i == page_index:
            cv2.circle(canvas, (dx, dot_y), 4, ACCENT, -1)
        else:
            cv2.circle(canvas, (dx, dot_y), 3, TEXT_DIM, 1)

    return {
        'card': (card_x, card_y, card_w, card_h),
        'prev_arrow': prev_region,
        'next_arrow': next_region,
    }


def _doc_compute_title_bbox(text, x, y, scale=0.52):
    """Compute bounding box of a text string for hit testing."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return (x, y - th, tw, th + baseline)


def _doc_point_in_rect(px, py, rect):
    """Check if point (px, py) is inside rect (x, y, w, h)."""
    rx, ry, rw, rh = rect
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def _doc_tag_bbox(text, x, y, scale=0.38):
    """Compute the bounding box that _draw_tag would produce."""
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x, pad_y = 6, 4
    return (x, y - th - pad_y, tw + pad_x * 2, th + pad_y * 2)


def show_preprocessing_preview(video_path, initial_params=None):
    """
    Show an interactive preview window for tuning preprocessing parameters.

    Uses a sleek dark-grey GUI with minimal fluorescent accent highlights.
    All controls — including custom-drawn sliders — live inside the single
    main window.  Two-row layout: top row has Original at native video
    resolution (or scaled to fit) with a parameter sidebar on the right;
    bottom row has CLAHE, MF Response, and Edges panels side-by-side.
    Status bar with frame slider at the bottom.

    The MF Response panel shows the directional matched-filter SNR heatmap
    — a warm-toned intensity map revealing dim linear features that Canny
    edge detection misses.  Detected line segments are overlaid in bright
    magenta.  The MF SNR slider controls the detection threshold.

    Controls:
    - Use Frame slider to select exact frame containing trail signal
    - Click and drag sliders in the sidebar to adjust parameters
    - Click on the ORIGINAL panel to mark satellite trail start/end points
      (up to 3 examples). Marked trails build a signal envelope used to
      dynamically adapt detection thresholds.
    - Press SPACE or ENTER to accept current settings
    - Press ESC to cancel and use default settings
    - Press 'R' to reset to default values
    - Press 'U' to undo last marked trail (or cancel pending start point)
    - Press 'N' to jump forward 1 second
    - Press 'P' to jump back 1 second

    Args:
        video_path: Path to the input video file
        initial_params: Optional dict with initial parameter values

    Returns:
        Dict with selected preprocessing parameters, or None if cancelled.
        If user marked trail examples, the dict includes a 'signal_envelope'
        key containing measured brightness, contrast, length, and angle
        ranges from the marked trails.
    """

    # ── Theme colours (BGR) ──────────────────────────────────────────
    BG_DARK = (30, 30, 30)           # Main background
    BG_PANEL = (42, 42, 42)          # Panel / card background
    BG_SIDEBAR = (36, 36, 36)        # Sidebar background
    BORDER = (58, 58, 58)            # Subtle panel borders
    TEXT_PRIMARY = (210, 210, 210)    # Primary text (light grey)
    TEXT_DIM = (120, 120, 120)        # Secondary / dim text
    TEXT_HEADING = (180, 180, 180)    # Section headings
    ACCENT = (200, 255, 80)          # Fluorescent green-yellow accent (BGR)
    ACCENT_DIM = (100, 170, 50)      # Dimmed accent for less emphasis
    ACCENT_CYAN = (220, 220, 60)     # Cyan-ish accent for edges panel (BGR)
    ACCENT_MF = (120, 60, 255)       # Warm magenta for matched-filter panel (BGR)
    ACCENT_MF_DIM = (70, 40, 140)   # Dimmed MF accent (below threshold)
    ACCENT_MF_LINE = (140, 100, 255) # Bright MF accent for detected lines
    SLIDER_TRACK = (50, 50, 50)      # Slider track background
    SLIDER_FILL = (200, 255, 80)     # Slider filled portion (accent)
    SLIDER_THUMB = (240, 255, 160)   # Slider thumb highlight

    # Default parameters
    defaults = {
        'clahe_clip_limit': 60,      # Stored as int, divide by 10 for actual value (6.0)
        'clahe_tile_size': 6,
        'blur_kernel_size': 5,       # Must be odd
        'blur_sigma': 18,            # Stored as int, divide by 10 for actual value (1.8)
        'canny_low': 4,
        'canny_high': 100,
        'mf_snr_threshold': 25,     # Stored as int, divide by 10 for actual value (2.5)
    }

    # Use initial params if provided
    if initial_params:
        for key in defaults:
            if key in initial_params:
                defaults[key] = initial_params[key]

    # Open video and get a sample frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video for preview: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Start at 10% into the video to skip any intro
    current_frame_idx = max(0, int(total_frames * 0.1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        current_frame_idx = 0
        if not ret:
            print("Error: Could not read any frame from video")
            cap.release()
            return None

    original_frame = frame.copy()
    src_h, src_w = frame.shape[:2]

    # ── Slider definitions ───────────────────────────────────────────
    # Each slider: (param_key, label, display_fmt, min_val, max_val)
    # Frame slider is handled separately in the status bar for finer control.
    frame_v_min = 0
    frame_v_max = max(1, total_frames - 1)
    slider_defs = [
        ('clahe_clip_limit', 'CLAHE Clip',  lambda v: f"{v/10:.1f}", 0, 100),
        ('clahe_tile_size',  'CLAHE Tile',  lambda v: f"{v}",        2, 16),
        ('blur_kernel_size', 'Blur Kernel', lambda v: f"{v if v%2==1 else v+1}", 1, 15),
        ('blur_sigma',       'Blur Sigma',  lambda v: f"{v/10:.1f}", 0, 50),
        ('canny_low',        'Canny Low',   lambda v: f"{v}",        0, 100),
        ('canny_high',       'Canny High',  lambda v: f"{v}",        0, 200),
        ('mf_snr_threshold', 'MF SNR',      lambda v: f"{v/10:.1f}", 10, 60),
    ]

    params = defaults.copy()
    params['frame_idx'] = current_frame_idx

    # ── Detect screen size ───────────────────────────────────────────
    _screen_w, _screen_h = 1920, 1080
    try:
        import subprocess as _sp
        _xdpy = _sp.run(['xdpyinfo'], capture_output=True, text=True, timeout=2)
        for _line in _xdpy.stdout.split('\n'):
            if 'dimensions:' in _line:
                _sw, _sh = _line.split()[1].split('x')
                _screen_w, _screen_h = int(_sw), int(_sh)
                break
    except Exception:
        try:
            import subprocess as _sp
            _xr = _sp.run(['xrandr', '--current'], capture_output=True, text=True, timeout=2)
            for _line in _xr.stdout.split('\n'):
                if ' connected ' in _line and 'x' in _line:
                    import re as _re
                    _m = _re.search(r'(\d{3,5})x(\d{3,5})', _line)
                    if _m:
                        _screen_w, _screen_h = int(_m.group(1)), int(_m.group(2))
                        break
        except Exception:
            pass

    # ── Layout constants (two-row design) ────────────────────────────
    sidebar_w = max(280, min(380, int(_screen_w * 0.18)))
    status_bar_h = 56               # Taller to hold the wide frame slider
    gap = 2
    # Top row: Original at native video resolution + sidebar
    orig_w = src_w
    orig_h = src_h
    # Scale Original down if it would exceed available screen space
    _max_orig_w = _screen_w - sidebar_w - 40  # horizontal margin
    # Vertical budget: screen height minus bottom row (30% of orig_h),
    # status bar, gap, and 60px for title bar + taskbar
    _max_orig_h = int((_screen_h - status_bar_h - gap - 60) / 1.3)
    _fit_scale = min(1.0, _max_orig_w / orig_w, _max_orig_h / orig_h)
    if _fit_scale < 1.0:
        orig_w = int(orig_w * _fit_scale)
        orig_h = int(orig_h * _fit_scale)
    # Bottom row: 3 panels side-by-side (CLAHE, MF Response, Edges)
    bottom_h = int(orig_h * 0.3)    # 30% of Original panel height
    small_w = (orig_w - 2 * gap) // 3
    small_h = bottom_h
    canvas_w = orig_w + sidebar_w
    canvas_h = orig_h + gap + bottom_h + status_bar_h
    slider_row_h = 52           # Height per slider row
    slider_pad_x = 20           # Horizontal padding inside sidebar
    slider_track_h = 5          # Track bar height
    slider_thumb_r = 8          # Thumb radius
    slider_section_top = 72     # Y offset where sliders begin (below title)

    # Trail marking colours (BGR)
    TRAIL_MARK = (80, 255, 200)      # Bright green for confirmed trails
    TRAIL_PENDING = (80, 180, 255)   # Warm orange for pending start point
    TRAIL_RUBBER = (60, 200, 180)    # Dimmer green for rubber-band preview

    # Mutable state for mouse interaction
    # These will be set once we know the canvas geometry in the first frame.
    dragging = {'idx': -1}      # Index of slider being dragged (-1 = none)
    # Slider hit-test regions (populated by create_display, read by mouse_cb)
    slider_regions = []         # List of (x_start, x_end, y_center, min_val, max_val, param_key)

    # Trail marking state — user can click start+end on Original panel
    MAX_TRAILS = 3
    marked_trails = []          # List of analysis dicts (from _analyze_trail)
    pending_click = [None]      # [None] or [(src_x, src_y)] awaiting end point
    mouse_pos = [0, 0]          # Live mouse position for rubber-band line

    # Matched-filter cache — the SNR map only depends on the frame, not on
    # the threshold slider, so we recompute it only when the frame changes.
    mf_cache = {'frame_idx': -1, 'snr_map': None}
    mf_lines_cache = {'frame_idx': -1, 'snr_thresh': -1, 'lines': []}

    # Temporal reference cache — built from surrounding frames when the user
    # navigates to a new frame.  Much slower to compute than the MF cache
    # (requires reading N frames from disk), so it's done lazily and only
    # when the frame changes.
    _TEMPORAL_N = 3   # Frames to each side for temporal median (total = 2N+1 = 7)
    temporal_ref_cache = {'frame_idx': -1, 'diff_image': None, 'noise_map': None}

    # ── Documentation overlay state ───────────────────────────────────
    doc_state = {
        'visible': False,
        'page': 0,
        'konami_unlocked': False,
        'key_buffer': [],
        'flash_timer': 0,
        'title_hovered': False,
        'regions': None,       # click regions from _draw_doc_overlay
        'tag_regions': [],     # [(bbox, page_index), ...] for right-click
    }
    doc_pages = _PREPROC_DOC_PAGES

    # ── Pre-computed MF kernel bank ──────────────────────────────────
    # 36 angles (5-deg steps) × 2 kernel lengths = 72 kernels.
    # Sufficient angular resolution for the preview heatmap; the actual
    # detector uses its own 72-angle × 3-scale bank at processing time.
    import math as _math_kb
    _mf_kernel_bank = []
    for _klen in [15, 31]:
        for _i in range(36):
            _angle_deg = _i * 180.0 / 36
            _ksize = _klen if _klen % 2 == 1 else _klen + 1
            _center = _ksize // 2
            _rad = _math_kb.radians(_angle_deg)
            _cos_a, _sin_a = _math_kb.cos(_rad), _math_kb.sin(_rad)

            _yc, _xc = np.mgrid[:_ksize, :_ksize]
            _dx = (_xc - _center).astype(np.float32)
            _dy = (_yc - _center).astype(np.float32)
            _perp = np.abs(-_sin_a * _dx + _cos_a * _dy)
            _along = np.abs(_cos_a * _dx + _sin_a * _dy)

            _kern = np.exp(-0.5 * _perp ** 2).astype(np.float32)
            _kern[_along > _center] = 0
            _ksum = np.sum(_kern)
            if _ksum > 0:
                _kern /= _ksum

            _noise_factor = float(np.sqrt(np.sum(_kern ** 2)))
            _mf_kernel_bank.append((_kern, _noise_factor))
    _mf_kernel_bank = tuple(_mf_kernel_bank)  # freeze

    # ── Window setup ─────────────────────────────────────────────────
    window_name = "Mnemosky  -  Preprocessing Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ── Helper drawing functions ─────────────────────────────────────

    def _fill_rect(img, x, y, w, h, color):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    def _draw_border(img, x, y, w, h, color, thickness=1):
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, thickness)

    def _put_text(img, text, x, y, color, scale=0.42, thickness=1):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_tag(img, text, x, y, bg_color, text_color, scale=0.38):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        pad_x, pad_y = 6, 4
        _fill_rect(img, x, y - th - pad_y, tw + pad_x * 2, th + pad_y * 2, bg_color)
        _put_text(img, text, x + pad_x, y - 1, text_color, scale)

    # ── Preprocessing logic ──────────────────────────────────────────

    def apply_preprocessing(frm, p):
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        clip_limit = max(0.1, p['clahe_clip_limit'] / 10.0)
        tile_size = max(2, p['clahe_tile_size'])
        blur_kernel = p['blur_kernel_size']
        blur_sigma = p['blur_sigma'] / 10.0
        if blur_kernel < 1:
            blur_kernel = 1
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)
        if blur_kernel >= 1 and blur_sigma > 0:
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), blur_sigma)
        else:
            blurred = enhanced
        canny_low = max(1, p['canny_low'])
        canny_high = max(canny_low + 1, p['canny_high'])
        edges = cv2.Canny(blurred, canny_low, canny_high)
        return gray, enhanced, blurred, edges

    # ── Temporal reference computation ──────────────────────────────────

    def build_temporal_reference(target_idx, gray_target):
        """Load surrounding frames and build temporal median reference.

        Reads up to 2×_TEMPORAL_N frames around target_idx from the video,
        computes per-pixel temporal median, and returns the difference image
        (target − reference) plus a per-pixel noise map.  Stars and fixed
        pattern noise are removed from the difference image.

        The result is cached so repeated calls for the same frame are free.
        """
        if temporal_ref_cache['frame_idx'] == target_idx:
            return temporal_ref_cache['diff_image'], temporal_ref_cache['noise_map']

        buf = TemporalFrameBuffer(capacity=2 * _TEMPORAL_N + 1)

        # Determine range of frames to read
        lo = max(0, target_idx - _TEMPORAL_N)
        hi = min(total_frames - 1, target_idx + _TEMPORAL_N)

        saved_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)

        for idx in range(lo, hi + 1):
            ok, frm = cap.read()
            if not ok:
                break
            buf.add(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY))

        # Restore video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

        if buf.is_ready():
            ctx = buf.get_temporal_context(gray_target)
            temporal_ref_cache['diff_image'] = ctx['diff_image']
            temporal_ref_cache['noise_map'] = ctx['noise_map']
        else:
            temporal_ref_cache['diff_image'] = None
            temporal_ref_cache['noise_map'] = None

        temporal_ref_cache['frame_idx'] = target_idx
        return temporal_ref_cache['diff_image'], temporal_ref_cache['noise_map']

    # ── Matched-filter response computation ────────────────────────────

    def compute_mf_response(gray_frm):
        """Compute the multi-scale directional matched-filter SNR map.

        When temporal reference is available (built from surrounding frames),
        uses the temporal difference image instead of spatial median — stars,
        sky gradients, and vignetting are removed perfectly, and the per-pixel
        noise map gives spatially adaptive thresholding.

        Falls back to spatial median background subtraction when temporal
        reference is not yet available (e.g. first frame or too few frames).

        Returns:
            snr_map at full resolution (float32 array, same size as gray_frm)
        """
        h, w = gray_frm.shape

        # Downsample at 1/2 resolution for faster filter2D calls
        scale = 0.5
        sm_w, sm_h = int(w * scale), int(h * scale)

        # Try temporal reference first (strictly superior to spatial median)
        diff_img, noise_map = build_temporal_reference(current_frame_idx, gray_frm)

        if diff_img is not None and noise_map is not None:
            signal = cv2.resize(diff_img, (sm_w, sm_h),
                                interpolation=cv2.INTER_AREA)
            noise_map_small = cv2.resize(noise_map, (sm_w, sm_h),
                                         interpolation=cv2.INTER_LINEAR)
            noise_std = float(np.median(noise_map_small))
            if noise_std < 0.5:
                noise_std = 0.5
            # Suppress faint noise — 3σ floor (standard detection threshold)
            min_signal = 3.0 * noise_map_small
            signal[signal < min_signal] = 0
            use_noise_map = True
        else:
            # Fallback: spatial median background subtraction
            small = cv2.resize(gray_frm, (sm_w, sm_h),
                               interpolation=cv2.INTER_AREA)
            bg = cv2.medianBlur(small, 31)
            signal = small.astype(np.float32) - bg.astype(np.float32)
            signal = np.clip(signal, 0, None)

            flat = signal.ravel()
            median_val = np.median(flat)
            mad = np.median(np.abs(flat - median_val))
            noise_std = max(0.5, mad * 1.4826)
            # Suppress faint noise — 3σ floor (standard detection threshold)
            min_signal = 3.0 * noise_std
            signal[signal < min_signal] = 0
            noise_map_small = None
            use_noise_map = False

        # Multi-scale directional filter bank — uses pre-computed kernel bank
        # (72 kernels: 36 angles × 2 scales, built once at startup).
        best_snr = np.zeros_like(signal)

        t0 = time.perf_counter()
        for kern, noise_factor in _mf_kernel_bank:
            response = cv2.filter2D(signal, cv2.CV_32F, kern)

            if use_noise_map:
                snr = response / (noise_map_small * noise_factor + 1e-10)
            else:
                snr = response / (noise_std * noise_factor + 1e-10)

            better = snr > best_snr
            best_snr[better] = snr[better]
        elapsed = time.perf_counter() - t0
        print(f"  MF response: {elapsed:.3f}s ({len(_mf_kernel_bank)} kernels)")

        # Scale back to full resolution
        return cv2.resize(best_snr, (w, h), interpolation=cv2.INTER_LINEAR)

    def get_mf_snr_map(gray_frm, frame_idx):
        """Return the cached MF SNR map, recomputing only when frame changes."""
        if mf_cache['frame_idx'] != frame_idx:
            mf_cache['snr_map'] = compute_mf_response(gray_frm)
            mf_cache['frame_idx'] = frame_idx
        return mf_cache['snr_map']

    def extract_mf_lines(snr_map, snr_thresh):
        """Threshold the SNR map and extract candidate line segments.

        Results are cached by (frame_idx, snr_thresh) — the Hough transform
        and morphology only re-run when either value changes.

        Post-filtering: duplicate suppression (distance < 60px AND angle < 15deg)
        and minimum full-resolution length (60px) to cut false positives.
        """
        if (mf_lines_cache['frame_idx'] == current_frame_idx and
                mf_lines_cache['snr_thresh'] == snr_thresh):
            return mf_lines_cache['lines']

        h, w = snr_map.shape
        # Work at half resolution for Hough (matches detector behaviour)
        scale = 0.5
        small_snr = cv2.resize(snr_map, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        significant = (small_snr > snr_thresh).astype(np.uint8) * 255
        cleanup = np.ones((3, 3), np.uint8)
        significant = cv2.dilate(significant, cleanup, iterations=1)
        significant = cv2.erode(significant, cleanup, iterations=1)

        lines = cv2.HoughLinesP(significant, 1, np.pi / 180, 20,
                                minLineLength=25, maxLineGap=15)
        inv = 1.0 / scale
        min_full_length = 60  # minimum length at full resolution
        candidates = []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                fx1, fy1 = int(x1 * inv), int(y1 * inv)
                fx2, fy2 = int(x2 * inv), int(y2 * inv)
                length = np.sqrt((fx2 - fx1) ** 2 + (fy2 - fy1) ** 2)
                if length < min_full_length:
                    continue
                angle = np.degrees(np.arctan2(abs(fy2 - fy1), abs(fx2 - fx1)))
                cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
                candidates.append((fx1, fy1, fx2, fy2, length, angle, cx, cy))

        # Duplicate suppression — merge lines with centers < 60px apart
        # and angles within 15 degrees, keeping only the longest.
        candidates.sort(key=lambda c: c[4], reverse=True)  # longest first
        keep = []
        for cand in candidates:
            _, _, _, _, c_len, c_ang, c_cx, c_cy = cand
            is_dup = False
            for kept in keep:
                _, _, _, _, _, k_ang, k_cx, k_cy = kept
                dist = np.sqrt((c_cx - k_cx) ** 2 + (c_cy - k_cy) ** 2)
                ang_diff = abs(c_ang - k_ang)
                if ang_diff > 90:
                    ang_diff = 180 - ang_diff
                if dist < 60 and ang_diff < 15:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(cand)

        scaled = [(c[0], c[1], c[2], c[3]) for c in keep]

        mf_lines_cache['frame_idx'] = current_frame_idx
        mf_lines_cache['snr_thresh'] = snr_thresh
        mf_lines_cache['lines'] = scaled
        return scaled

    # ── Trail analysis helper ─────────────────────────────────────────

    def _analyze_trail(start, end, gray_frm, color_frm):
        """Sample brightness along a user-marked trail and return analysis."""
        x1, y1 = start
        x2, y2 = end
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 5:
            return None
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

        num_samples = max(20, int(length / 3))
        num_samples = min(num_samples, 200)

        brightness_samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= py < gray_frm.shape[0] and 0 <= px < gray_frm.shape[1]:
                y_lo = max(0, py - 1)
                y_hi = min(gray_frm.shape[0], py + 2)
                x_lo = max(0, px - 1)
                x_hi = min(gray_frm.shape[1], px + 2)
                brightness_samples.append(float(np.max(gray_frm[y_lo:y_hi, x_lo:x_hi])))

        if not brightness_samples:
            return None

        arr = np.array(brightness_samples)
        avg_brightness = float(np.mean(arr))
        max_brightness = int(np.max(arr))
        brightness_std = float(np.std(arr))

        # Contrast against local background
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bg_pad = 30
        bg_region = gray_frm[
            max(0, cy - bg_pad):min(gray_frm.shape[0], cy + bg_pad),
            max(0, cx - bg_pad):min(gray_frm.shape[1], cx + bg_pad)
        ]
        bg_brightness = float(np.median(bg_region)) if bg_region.size > 0 else 0.0
        contrast_ratio = avg_brightness / (bg_brightness + 1e-5)

        return {
            'start': start,
            'end': end,
            'length': length,
            'angle': angle,
            'avg_brightness': avg_brightness,
            'max_brightness': max_brightness,
            'brightness_std': brightness_std,
            'contrast_ratio': contrast_ratio,
            'smoothness': brightness_std / (avg_brightness + 1e-5),
        }

    def _compute_signal_envelope(trail_analyses):
        """Derive a signal envelope from user-marked trail examples."""
        if not trail_analyses:
            return None
        margin = 0.30  # 30 % margin on ranges

        lengths = [t['length'] for t in trail_analyses]
        avg_brs = [t['avg_brightness'] for t in trail_analyses]
        max_brs = [t['max_brightness'] for t in trail_analyses]
        contrasts = [t['contrast_ratio'] for t in trail_analyses]
        smoothnesses = [t['smoothness'] for t in trail_analyses]
        angles = [t['angle'] for t in trail_analyses]

        return {
            'num_examples': len(trail_analyses),
            'trails': trail_analyses,
            'length_range': (max(30, min(lengths) * (1 - margin)),
                             max(lengths) * (1 + margin)),
            'brightness_range': (min(avg_brs) * (1 - margin),
                                 max(avg_brs) * (1 + margin)),
            'max_brightness_range': (min(max_brs), max(max_brs)),
            'contrast_range': (min(contrasts) * (1 - margin * 0.5),
                               max(contrasts)),
            'smoothness_max': max(smoothnesses) * (1 + margin),
            'angles': angles,
        }

    # ── Composite display builder ────────────────────────────────────

    def create_display(frm, gray, enhanced, blurred, edges, p):
        nonlocal slider_regions

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = BG_DARK

        # ── Panels ───────────────────────────────────────────────────
        # Top row: Original at native (or fitted) resolution
        # Bottom row: CLAHE, MF Response, Edges side-by-side
        orig_panel = cv2.resize(frm, (orig_w, orig_h))

        enh_resized = cv2.resize(enhanced, (small_w, small_h))
        enh_bgr = cv2.cvtColor(enh_resized, cv2.COLOR_GRAY2BGR)

        # ── Matched-filter response heatmap ─────────────────────────
        snr_thresh = p['mf_snr_threshold'] / 10.0
        snr_map = get_mf_snr_map(gray, current_frame_idx)
        snr_resized = cv2.resize(snr_map, (small_w, small_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Two-tone heatmap: dim below threshold, bright above
        mf_bgr = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        mf_bgr[:] = BG_PANEL

        # Dim sub-threshold signal (subtle visibility so user sees the landscape)
        has_signal = snr_resized > 2.0
        below_thresh = has_signal & (snr_resized < snr_thresh)
        above_thresh = snr_resized >= snr_thresh

        # Vectorized heatmap rendering (no per-channel Python loops)
        _bg = np.array(BG_PANEL, dtype=np.float32)
        _dim = np.array(ACCENT_MF_DIM, dtype=np.float32)
        _bright = np.array(ACCENT_MF, dtype=np.float32)

        if np.any(below_thresh):
            intensity_below = np.clip(snr_resized[below_thresh] / snr_thresh, 0, 1)
            blend = _bg + (_dim - _bg) * intensity_below[:, np.newaxis]
            mf_bgr[below_thresh] = np.clip(blend, 0, 255).astype(np.uint8)

        if np.any(above_thresh):
            intensity_above = np.clip(snr_resized[above_thresh] / (snr_thresh * 3), 0.25, 1)
            blend = _dim + (_bright - _dim) * intensity_above[:, np.newaxis]
            mf_bgr[above_thresh] = np.clip(blend, 0, 255).astype(np.uint8)

        # Overlay detected line segments from the matched filter
        mf_lines = extract_mf_lines(snr_map, snr_thresh)
        mf_scale_x = small_w / src_w
        mf_scale_y = small_h / src_h
        for lx1, ly1, lx2, ly2 in mf_lines:
            pt1 = (int(lx1 * mf_scale_x), int(ly1 * mf_scale_y))
            pt2 = (int(lx2 * mf_scale_x), int(ly2 * mf_scale_y))
            cv2.line(mf_bgr, pt1, pt2, ACCENT_MF_LINE, 2, cv2.LINE_AA)

        # ── Edges panel ────────────────────────────────────────────────
        edge_gray_r = cv2.resize(edges, (small_w, small_h))
        edge_bgr = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        edge_bgr[:] = BG_PANEL
        edge_bgr[edge_gray_r > 0] = ACCENT_CYAN

        # Place panels — top row: Original; bottom row: 3 panels side-by-side
        bot_y = orig_h + gap
        for px, py, pw, ph, panel in [
            (0,  0,                              orig_w,  orig_h,  orig_panel),
            (0,  bot_y,                          small_w, small_h, enh_bgr),
            (small_w + gap, bot_y,               small_w, small_h, mf_bgr),
            (2 * (small_w + gap), bot_y,         small_w, small_h, edge_bgr),
        ]:
            canvas[py:py + ph, px:px + pw] = panel
            _draw_border(canvas, px, py, pw, ph, BORDER)

        # ── Pending trail interaction on Original panel ─────────────
        scale_x = orig_w / src_w
        scale_y = orig_h / src_h

        # Completed trails are shown as thumbnails in the sidebar instead
        # of being drawn on the Original panel (keeps it clean).

        # Pending start point (first click placed, waiting for end)
        if pending_click[0] is not None:
            t_sx, t_sy = pending_click[0]
            p1 = (int(t_sx * scale_x), int(t_sy * scale_y))
            cv2.circle(canvas, p1, 6, TRAIL_PENDING, 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, TRAIL_PENDING, -1, cv2.LINE_AA)
            # Rubber-band line to current mouse position (if mouse is on panel)
            mx, my = mouse_pos
            if 0 <= mx < orig_w and 0 <= my < orig_h:
                cv2.line(canvas, p1, (mx, my), TRAIL_RUBBER, 1, cv2.LINE_AA)

        # Panel tags (with bbox tracking for right-click doc access)
        tag_y, tag_x = 18, 8
        _tag_regions = []   # [(bbox, doc_page_index), ...]
        trail_count_str = f"  [{len(marked_trails)}/{MAX_TRAILS}]" if marked_trails or pending_click[0] else ""
        orig_tag_text = "ORIGINAL" + trail_count_str
        _draw_tag(canvas, orig_tag_text, tag_x, tag_y, BG_DARK, TEXT_DIM if not trail_count_str else TRAIL_MARK)
        _tag_regions.append((_doc_tag_bbox(orig_tag_text, tag_x, tag_y), 1))
        clip_val = p['clahe_clip_limit'] / 10.0
        clahe_tag_text = f"CLAHE  clip {clip_val:.1f}  tile {p['clahe_tile_size']}"
        _draw_tag(canvas, clahe_tag_text,
                  tag_x, bot_y + tag_y, BG_DARK, ACCENT)
        _tag_regions.append((_doc_tag_bbox(clahe_tag_text, tag_x, bot_y + tag_y), 1))
        mf_snr_val = p['mf_snr_threshold'] / 10.0
        mf_line_count = len(mf_lines)
        is_temporal = temporal_ref_cache['diff_image'] is not None
        mf_mode = "TEMPORAL" if is_temporal else "SPATIAL"
        mf_tag = f"MF {mf_mode}  SNR>={mf_snr_val:.1f}"
        if mf_line_count > 0:
            mf_tag += f"  [{mf_line_count}]"
        _draw_tag(canvas, mf_tag,
                  small_w + gap + tag_x, bot_y + tag_y, BG_DARK, ACCENT_MF)
        _tag_regions.append((_doc_tag_bbox(mf_tag, small_w + gap + tag_x, bot_y + tag_y), 1))
        edges_tag_text = f"EDGES  {p['canny_low']}-{p['canny_high']}"
        _draw_tag(canvas, edges_tag_text,
                  2 * (small_w + gap) + tag_x, bot_y + tag_y, BG_DARK, ACCENT_CYAN)
        _tag_regions.append((_doc_tag_bbox(edges_tag_text, 2 * (small_w + gap) + tag_x, bot_y + tag_y), 1))
        doc_state['tag_regions'] = _tag_regions

        # ── Sidebar ──────────────────────────────────────────────────
        sb_x = orig_w
        _fill_rect(canvas, sb_x, 0, sidebar_w, canvas_h - status_bar_h, BG_SIDEBAR)
        _draw_border(canvas, sb_x, 0, sidebar_w, canvas_h - status_bar_h, BORDER)

        # Title (with hover hint for doc overlay)
        _title_x, _title_y = sb_x + 14, 24
        _title_bbox = _doc_compute_title_bbox("MNEMOSKY", _title_x, _title_y, 0.52)
        doc_state['title_bbox'] = _title_bbox
        if doc_state['title_hovered']:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, (230, 255, 140), 0.52, 1)
            _put_text(canvas, "(?)", _title_x + _title_bbox[2] + 4, _title_y, ACCENT_DIM, 0.36)
        else:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, ACCENT, 0.52, 1)
        _put_text(canvas, "Preprocessing", sb_x + 14, 46, TEXT_HEADING, 0.40)
        cv2.line(canvas, (sb_x + 14, 56), (sb_x + sidebar_w - 14, 56), BORDER, 1)

        # ── Draw sliders ─────────────────────────────────────────────
        new_regions = []
        track_x_start = sb_x + slider_pad_x
        track_x_end = sb_x + sidebar_w - slider_pad_x
        track_width = track_x_end - track_x_start

        for i, (key, label, fmt_fn, v_min, v_max) in enumerate(slider_defs):
            row_y = slider_section_top + i * slider_row_h
            val = p[key]

            # Label (left) and value (right)
            _put_text(canvas, label, track_x_start, row_y + 14, TEXT_DIM, 0.34)
            _put_text(canvas, fmt_fn(val), track_x_end - 36, row_y + 14, ACCENT, 0.36, 1)

            # Slider track
            track_y = row_y + 28
            _fill_rect(canvas, track_x_start, track_y - slider_track_h // 2,
                       track_width, slider_track_h, SLIDER_TRACK)

            # Fill
            ratio = (val - v_min) / max(1, v_max - v_min)
            fill_w = int(track_width * ratio)
            if fill_w > 0:
                _fill_rect(canvas, track_x_start, track_y - slider_track_h // 2,
                           fill_w, slider_track_h, SLIDER_FILL)

            # Thumb
            thumb_x = track_x_start + fill_w
            cv2.circle(canvas, (thumb_x, track_y), slider_thumb_r, SLIDER_THUMB, -1)
            cv2.circle(canvas, (thumb_x, track_y), slider_thumb_r, ACCENT, 1, cv2.LINE_AA)

            # Register region for hit testing
            new_regions.append((track_x_start, track_x_end, track_y, v_min, v_max, key))

        # slider_regions is assigned later, after appending the frame slider

        # ── Trail examples section (below sliders) ──────────────────
        trail_y = slider_section_top + len(slider_defs) * slider_row_h + 12
        cv2.line(canvas, (sb_x + 14, trail_y - 6), (sb_x + sidebar_w - 14, trail_y - 6), BORDER, 1)
        _put_text(canvas, f"TRAIL EXAMPLES  {len(marked_trails)}/{MAX_TRAILS}",
                  sb_x + 14, trail_y + 10, TEXT_HEADING, 0.36)
        trail_y += 22
        if not marked_trails and pending_click[0] is None:
            _put_text(canvas, "Click start+end on ORIGINAL", sb_x + 14, trail_y, TEXT_DIM, 0.28)
            trail_y += 14
            _put_text(canvas, "panel to mark satellite trails", sb_x + 14, trail_y, TEXT_DIM, 0.28)
            trail_y += 14
        elif pending_click[0] is not None:
            _put_text(canvas, "Click trail END point ...", sb_x + 14, trail_y, TRAIL_PENDING, 0.30)
            trail_y += 14

        thumb_pad_x = 10  # horizontal padding inside sidebar for thumbnails
        thumb_w = sidebar_w - 2 * thumb_pad_x
        crop_pad = 40  # padding around trail bbox in source pixels
        for ti, tr in enumerate(marked_trails):
            # Compute padded crop region in source coordinates
            t_x0 = min(tr['start'][0], tr['end'][0]) - crop_pad
            t_y0 = min(tr['start'][1], tr['end'][1]) - crop_pad
            t_x1 = max(tr['start'][0], tr['end'][0]) + crop_pad
            t_y1 = max(tr['start'][1], tr['end'][1]) + crop_pad
            t_x0 = max(0, t_x0)
            t_y0 = max(0, t_y0)
            t_x1 = min(src_w, t_x1)
            t_y1 = min(src_h, t_y1)
            crop_w = t_x1 - t_x0
            crop_h = t_y1 - t_y0
            if crop_w > 0 and crop_h > 0:
                crop = frm[t_y0:t_y1, t_x0:t_x1]
                # Scale to sidebar width, preserving aspect ratio
                thumb_h = max(1, int(thumb_w * crop_h / crop_w))
                thumb_h = min(thumb_h, 120)  # cap height
                thumb = cv2.resize(crop, (thumb_w, thumb_h))

                # Draw trail line on thumbnail
                ts_x = int((tr['start'][0] - t_x0) / crop_w * thumb_w)
                ts_y = int((tr['start'][1] - t_y0) / crop_h * thumb_h)
                te_x = int((tr['end'][0] - t_x0) / crop_w * thumb_w)
                te_y = int((tr['end'][1] - t_y0) / crop_h * thumb_h)
                cv2.line(thumb, (ts_x, ts_y), (te_x, te_y), TRAIL_MARK, 1, cv2.LINE_AA)

                # Place thumbnail on canvas
                tx = sb_x + thumb_pad_x
                if trail_y + thumb_h + 18 < canvas_h - status_bar_h:
                    # Label above thumbnail
                    info_str = f"#{ti+1}  L={tr['length']:.0f}  br={tr['avg_brightness']:.1f}  c={tr['contrast_ratio']:.2f}"
                    _put_text(canvas, info_str, tx, trail_y + 10, TRAIL_MARK, 0.26)
                    trail_y += 14
                    canvas[trail_y:trail_y + thumb_h, tx:tx + thumb_w] = thumb
                    _draw_border(canvas, tx, trail_y, thumb_w, thumb_h, TRAIL_MARK)
                    trail_y += thumb_h + 6
        trail_y += 6

        # ── Controls help ─────────────────────────────────────────
        help_y = trail_y
        cv2.line(canvas, (sb_x + 14, help_y - 6), (sb_x + sidebar_w - 14, help_y - 6), BORDER, 1)
        _put_text(canvas, "CONTROLS", sb_x + 14, help_y + 10, TEXT_HEADING, 0.36)
        help_y += 26
        for key_str, desc in [
            ("SPACE / ENTER", "Accept"),
            ("ESC", "Cancel"),
            ("R", "Reset"),
            ("U", "Undo last trail"),
            ("N / P", "Next / Prev frame"),
        ]:
            _put_text(canvas, key_str, sb_x + 14, help_y, ACCENT_DIM, 0.30)
            _put_text(canvas, desc, sb_x + 126, help_y, TEXT_DIM, 0.30)
            help_y += 16

        # ── Status bar with wide frame slider ─────────────────────────
        sb_y = canvas_h - status_bar_h
        _fill_rect(canvas, 0, sb_y, canvas_w, status_bar_h, BG_PANEL)
        cv2.line(canvas, (0, sb_y), (canvas_w, sb_y), BORDER, 1)

        # Top row: info text
        _put_text(canvas, f"Frame {current_frame_idx}/{total_frames}", 12, sb_y + 16, TEXT_DIM, 0.36)
        _put_text(canvas, f"{src_w}x{src_h}", canvas_w - 90, sb_y + 16, TEXT_DIM, 0.36)
        cv2.circle(canvas, (canvas_w // 2, sb_y + 11), 4, ACCENT, -1)
        _put_text(canvas, "LIVE", canvas_w // 2 + 10, sb_y + 16, ACCENT_DIM, 0.33)

        # Bottom row: wide frame slider spanning full width
        frame_track_pad = 14
        frame_track_x0 = frame_track_pad
        frame_track_x1 = canvas_w - frame_track_pad
        frame_track_y = sb_y + 40
        frame_track_w = frame_track_x1 - frame_track_x0

        _fill_rect(canvas, frame_track_x0, frame_track_y - slider_track_h // 2,
                   frame_track_w, slider_track_h, SLIDER_TRACK)
        f_ratio = (p['frame_idx'] - frame_v_min) / max(1, frame_v_max - frame_v_min)
        f_fill_w = int(frame_track_w * f_ratio)
        if f_fill_w > 0:
            _fill_rect(canvas, frame_track_x0, frame_track_y - slider_track_h // 2,
                       f_fill_w, slider_track_h, SLIDER_FILL)
        f_thumb_x = frame_track_x0 + f_fill_w
        cv2.circle(canvas, (f_thumb_x, frame_track_y), slider_thumb_r, SLIDER_THUMB, -1)
        cv2.circle(canvas, (f_thumb_x, frame_track_y), slider_thumb_r, ACCENT, 1, cv2.LINE_AA)

        # Store frame slider region for hit testing (appended after sidebar sliders)
        new_regions.append((frame_track_x0, frame_track_x1, frame_track_y,
                            frame_v_min, frame_v_max, 'frame_idx'))

        slider_regions = new_regions

        # "UNLOCKED" flash on status bar when secret word is typed
        if doc_state['flash_timer'] and time.time() - doc_state['flash_timer'] < 2.0:
            _put_text(canvas, "UNLOCKED", canvas_w // 2 - 40, sb_y + 16, ACCENT, 0.40, 1)

        # ── Documentation overlay ─────────────────────────────────────
        if doc_state['visible']:
            doc_state['regions'] = _draw_doc_overlay(
                canvas, doc_state['page'], doc_pages,
                doc_state['konami_unlocked'])

        return canvas

    # ── Mouse callback for slider interaction ────────────────────────
    # Canvas is built at the exact target window size, so mouse
    # coordinates map 1:1 to canvas pixels — no scaling needed.

    def _update_slider_from_x(mouse_x, mouse_y):
        """Find which slider the mouse is on and update its value."""
        for i, (x_start, x_end, y_center, v_min, v_max, key) in enumerate(slider_regions):
            if abs(mouse_y - y_center) < slider_row_h // 2 and x_start - 4 <= mouse_x <= x_end + 4:
                ratio = max(0.0, min(1.0, (mouse_x - x_start) / max(1, x_end - x_start)))
                params[key] = int(round(v_min + ratio * (v_max - v_min)))
                dragging['idx'] = i
                return
        dragging['idx'] = -1

    def on_mouse(event, x, y, flags, userdata):
        # Always track mouse position (for rubber-band preview)
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = x
            mouse_pos[1] = y
            # Title hover detection
            if doc_state.get('title_bbox'):
                was_hovered = doc_state['title_hovered']
                doc_state['title_hovered'] = _doc_point_in_rect(
                    x, y, doc_state['title_bbox'])
                if doc_state['title_hovered'] != was_hovered:
                    pass  # dirty flag picks up change

        # When doc overlay is visible, handle overlay clicks only
        if doc_state['visible']:
            if event == cv2.EVENT_LBUTTONDOWN:
                regions = doc_state.get('regions')
                if regions:
                    # Check prev/next arrows
                    if regions['prev_arrow'] and _doc_point_in_rect(x, y, regions['prev_arrow']):
                        if doc_state['page'] > 0:
                            doc_state['page'] -= 1
                        return
                    if regions['next_arrow'] and _doc_point_in_rect(x, y, regions['next_arrow']):
                        vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                        if doc_state['page'] < len(vis) - 1:
                            doc_state['page'] += 1
                        return
                    # Click outside card dismisses overlay
                    if not _doc_point_in_rect(x, y, regions['card']):
                        doc_state['visible'] = False
                        return
            return  # Don't process slider/trail clicks when overlay open

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check MNEMOSKY title click → toggle doc overlay
            if doc_state.get('title_bbox') and _doc_point_in_rect(x, y, doc_state['title_bbox']):
                doc_state['visible'] = not doc_state['visible']
                doc_state['page'] = 0
                return
            # Check if click is inside the Original panel (left column)
            if 0 <= x < orig_w and 0 <= y < orig_h:
                # Map panel coordinates → source frame coordinates
                src_x = int(x / orig_w * src_w)
                src_y = int(y / orig_h * src_h)
                if pending_click[0] is None:
                    if len(marked_trails) < MAX_TRAILS:
                        pending_click[0] = (src_x, src_y)
                else:
                    # Complete the trail — analyse on current grayscale frame
                    gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    analysis = _analyze_trail(pending_click[0], (src_x, src_y),
                                              gray_now, frame)
                    if analysis is not None:
                        marked_trails.append(analysis)
                    pending_click[0] = None
            else:
                # Click is outside Original panel → slider interaction
                _update_slider_from_x(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click on panel tags → open doc overlay to that page
            for tag_bbox, page_idx in doc_state.get('tag_regions', []):
                if _doc_point_in_rect(x, y, tag_bbox):
                    doc_state['visible'] = True
                    doc_state['page'] = page_idx
                    return
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if dragging['idx'] >= 0:
                x_start, x_end, _, v_min, v_max, key = slider_regions[dragging['idx']]
                ratio = max(0.0, min(1.0, (x - x_start) / max(1, x_end - x_start)))
                params[key] = int(round(v_min + ratio * (v_max - v_min)))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging['idx'] = -1

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n" + "=" * 60)
    print("PREPROCESSING PREVIEW")
    print("=" * 60)
    print("Use the Frame slider to find a frame with satellite trail signal.")
    print("Then adjust other sliders to tune preprocessing parameters.")
    print("The goal is to preserve dim satellite trails while reducing noise.")
    print("The MF RESPONSE panel (bottom-centre) shows the directional matched")
    print("filter heatmap — tune MF SNR to control dim-trail sensitivity.")
    print("\nTrail marking  (up to 3 examples):")
    print("  Click on the ORIGINAL panel to set the START of a trail,")
    print("  then click again to set the END. The detector will adapt")
    print("  its parameters to match the brightness/contrast/length of")
    print("  your marked examples.")
    print("\nControls:")
    print("  Click+drag   - Adjust sliders in the sidebar")
    print("  SPACE/ENTER  - Accept current settings and continue")
    print("  ESC          - Cancel and use default settings")
    print("  R            - Reset to default values")
    print("  U            - Undo last marked trail")
    print("  N            - Jump forward 1 second")
    print("  P            - Jump back 1 second")
    print("=" * 60 + "\n")

    first_render = True

    # ── Dirty-flag state tracking ────────────────────────────────────
    # Only recompute expensive operations when inputs actually change.
    # Near-zero CPU when the user is idle (just cv2.waitKey).
    _prev_preproc_key = None   # (clahe_clip, clahe_tile, blur_k, blur_sigma, canny_low, canny_high, frame_idx)
    _prev_display_key = None   # (preproc_key, mf_snr, pending_click_state, mouse_if_pending, num_trails)
    _cached_gray = _cached_enhanced = _cached_blurred = _cached_edges = None
    _cached_display = None

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        # Check if Frame slider was dragged to a new position
        if params['frame_idx'] != current_frame_idx:
            current_frame_idx = params['frame_idx']
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame

        # Build dirty-flag keys
        preproc_key = (
            params['clahe_clip_limit'], params['clahe_tile_size'],
            params['blur_kernel_size'], params['blur_sigma'],
            params['canny_low'], params['canny_high'],
            current_frame_idx,
        )

        # Include mouse position only when rubber-band is active
        _pending_state = pending_click[0]
        _mouse_for_key = (mouse_pos[0], mouse_pos[1]) if _pending_state is not None else None
        display_key = (
            preproc_key,
            params['mf_snr_threshold'],
            _pending_state,
            _mouse_for_key,
            len(marked_trails),
            doc_state['visible'], doc_state['page'],
            doc_state['title_hovered'], doc_state['konami_unlocked'],
        )

        needs_preproc = (preproc_key != _prev_preproc_key)
        needs_display = (display_key != _prev_display_key)

        if needs_preproc:
            _cached_gray, _cached_enhanced, _cached_blurred, _cached_edges = (
                apply_preprocessing(frame, params)
            )
            _prev_preproc_key = preproc_key

        if needs_display or first_render:
            _cached_display = create_display(
                frame, _cached_gray, _cached_enhanced,
                _cached_blurred, _cached_edges, params
            )
            _prev_display_key = display_key

            cv2.imshow(window_name, _cached_display)

            if first_render:
                cv2.resizeWindow(window_name, canvas_w, canvas_h)
                first_render = False

        key = cv2.waitKey(30) & 0xFF

        # ── Secret word tracking (always active) ─────────────────────
        if key != 255 and 97 <= key <= 122:  # a-z lowercase
            doc_state['key_buffer'].append(chr(key))
            doc_state['key_buffer'] = doc_state['key_buffer'][-len(_DOC_SECRET_WORD):]
            if ''.join(doc_state['key_buffer']) == _DOC_SECRET_WORD:
                doc_state['konami_unlocked'] = True
                doc_state['visible'] = True
                vis = [p for p in doc_pages if not p.get('konami_only') or True]
                doc_state['page'] = len(vis) - 1
                doc_state['flash_timer'] = time.time()
                _prev_display_key = None  # force redraw
                continue

        # ── Doc overlay key handling (intercept before normal keys) ───
        if doc_state['visible']:
            if key == 27:  # ESC closes overlay, not the window
                doc_state['visible'] = False
                _prev_display_key = None
            elif key in (ord('a'), ord('A')):  # prev page
                if doc_state['page'] > 0:
                    doc_state['page'] -= 1
                    _prev_display_key = None
            elif key in (ord('d'), ord('D')):  # next page
                vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                if doc_state['page'] < len(vis) - 1:
                    doc_state['page'] += 1
                    _prev_display_key = None
            # Swallow all other keys when overlay is open
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Preview window closed. Using default parameters.")
                cap.release()
                return None
            continue

        # ── ? key toggles doc overlay ─────────────────────────────────
        if key == 63:  # ? (Shift+/)
            doc_state['visible'] = True
            doc_state['page'] = 0
            _prev_display_key = None
            continue

        if key == 27:  # ESC
            print("Preview cancelled. Using default parameters.")
            cv2.destroyWindow(window_name)
            cap.release()
            return None

        elif key in [13, 32]:  # ENTER or SPACE
            final_params = {
                'clahe_clip_limit': max(0.1, params['clahe_clip_limit'] / 10.0),
                'clahe_tile_size': max(2, params['clahe_tile_size']),
                'blur_kernel_size': params['blur_kernel_size'] if params['blur_kernel_size'] % 2 == 1 else params['blur_kernel_size'] + 1,
                'blur_sigma': params['blur_sigma'] / 10.0,
                'canny_low': params['canny_low'],
                'canny_high': params['canny_high'],
                'mf_snr_threshold': params['mf_snr_threshold'] / 10.0,
            }
            # Compute signal envelope from marked trail examples
            envelope = _compute_signal_envelope(marked_trails)
            if envelope is not None:
                final_params['signal_envelope'] = envelope
            print(f"\nAccepted preprocessing parameters:")
            print(f"  CLAHE clip limit: {final_params['clahe_clip_limit']:.1f}")
            print(f"  CLAHE tile size: {final_params['clahe_tile_size']}")
            print(f"  Blur kernel size: {final_params['blur_kernel_size']}")
            print(f"  Blur sigma: {final_params['blur_sigma']:.1f}")
            print(f"  Canny thresholds: {final_params['canny_low']}-{final_params['canny_high']}")
            print(f"  MF SNR threshold: {final_params['mf_snr_threshold']:.1f}")
            if envelope:
                print(f"  Signal envelope: {envelope['num_examples']} trail(s) marked")
                lr = envelope['length_range']
                br = envelope['brightness_range']
                cr = envelope['contrast_range']
                print(f"    Length range:     {lr[0]:.0f} - {lr[1]:.0f} px")
                print(f"    Brightness range: {br[0]:.1f} - {br[1]:.1f}")
                print(f"    Contrast range:   {cr[0]:.2f} - {cr[1]:.2f}")
                print(f"    Angles:           {', '.join(f'{a:.1f}°' for a in envelope['angles'])}")
            cv2.destroyWindow(window_name)
            cap.release()
            return final_params

        elif key == ord('r') or key == ord('R'):  # Reset
            saved_frame_idx = params['frame_idx']
            params = defaults.copy()
            params['frame_idx'] = saved_frame_idx
            marked_trails.clear()
            pending_click[0] = None
            mf_cache['frame_idx'] = -1  # Invalidate MF cache
            mf_lines_cache['frame_idx'] = -1  # Invalidate MF lines cache
            temporal_ref_cache['frame_idx'] = -1  # Invalidate temporal cache
            _prev_preproc_key = None  # Force full recompute
            _prev_display_key = None
            print("Parameters reset to defaults. Trail marks cleared.")

        elif key == ord('u') or key == ord('U'):  # Undo last trail / cancel pending
            if pending_click[0] is not None:
                pending_click[0] = None
                print("Pending trail start point cancelled.")
            elif marked_trails:
                removed = marked_trails.pop()
                print(f"Removed trail #{len(marked_trails)+1} (L={removed['length']:.0f}px)")
            else:
                print("No trails to undo.")

        elif key == ord('n') or key == ord('N'):  # Next frame (jump 1 second)
            current_frame_idx = min(total_frames - 1, current_frame_idx + int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if not ret:
                current_frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                params['frame_idx'] = current_frame_idx

        elif key == ord('p') or key == ord('P'):  # Previous frame (jump 1 second)
            current_frame_idx = max(0, current_frame_idx - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                params['frame_idx'] = current_frame_idx

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Preview window closed. Using default parameters.")
            cap.release()
            return None

    cap.release()
    return None


def show_radon_preview(video_path, initial_params=None):
    """Show an interactive preview window for debugging the Radon detection pipeline.

    Uses the same dark-grey/fluorescent-accent theme as the preprocessing preview.
    All controls — including custom-drawn sliders — live inside a single window.
    Two-row layout: top row has the Original frame with a parameter sidebar;
    bottom row has four diagnostic panels (Residual, Sinogram, LSD Lines,
    Detections) showing the key intermediate stages of the Radon pipeline.

    The six sliders control the most impactful Radon pipeline parameters:
      - Radon SNR threshold (sinogram peak detection sensitivity)
      - PCF Ratio (perpendicular cross-filter strictness)
      - Star Mask sigma (star removal aggressiveness)
      - LSD Significance (LSD detection strictness via log_eps)
      - PCF Kernel (PCF sampling half-width)
      - Min Length (minimum streak length in pixels)

    Controls:
    - Use Frame slider to navigate to a frame containing trail signal
    - Click and drag sliders in the sidebar to adjust parameters
    - Press SPACE or ENTER to accept current settings
    - Press ESC to cancel and use default settings
    - Press 'R' to reset to default values
    - Press 'N' to jump forward 1 second
    - Press 'P' to jump back 1 second

    Args:
        video_path: Path to the input video file
        initial_params: Optional dict with initial parameter values

    Returns:
        Dict with selected Radon parameters, or None if cancelled.
    """

    # ── Theme colours (BGR) — shared with preprocessing preview ─────
    BG_DARK = (30, 30, 30)
    BG_PANEL = (42, 42, 42)
    BG_SIDEBAR = (36, 36, 36)
    BORDER = (58, 58, 58)
    TEXT_PRIMARY = (210, 210, 210)
    TEXT_DIM = (120, 120, 120)
    TEXT_HEADING = (180, 180, 180)
    ACCENT = (200, 255, 80)           # Fluorescent green-yellow
    ACCENT_DIM = (100, 170, 50)
    SLIDER_TRACK = (50, 50, 50)
    SLIDER_FILL = (200, 255, 80)
    SLIDER_THUMB = (240, 255, 160)

    # Panel-specific accent colours
    ACCENT_RESIDUAL = (200, 180, 60)      # Teal for residual panel
    ACCENT_RESIDUAL_STAR = (60, 60, 180)  # Dim red for star mask overlay
    ACCENT_SINOGRAM = (50, 160, 255)      # Amber/orange for sinogram
    ACCENT_SINOGRAM_PEAK = (50, 255, 255) # Bright yellow for peaks
    ACCENT_LSD = (80, 255, 120)           # Green-yellow for LSD lines
    ACCENT_DET_RADON = (50, 180, 255)     # Amber for raw Radon candidates
    ACCENT_DET_PCF = (80, 255, 80)        # Bright green for PCF-confirmed
    ACCENT_DET_REJECT = (80, 80, 140)     # Dim red for rejected

    # Default parameters (stored as ints for slider precision)
    defaults = {
        'radon_snr_threshold': 30,    # ÷10 → 3.0
        'pcf_ratio_threshold': 20,    # ÷10 → 2.0
        'star_mask_sigma': 50,        # ÷10 → 5.0
        'lsd_log_eps': 10,            # ÷10 → 1.0 (range -2.0 to 5.0, stored +20 offset: 0–70)
        'pcf_kernel_len': 31,         # odd int
        'min_streak_length': 50,      # int pixels
    }
    # lsd_log_eps uses offset encoding: stored = (real + 2.0) * 10
    # So stored 0 → real -2.0, stored 10 → real -1.0, stored 30 → real 1.0
    defaults['lsd_log_eps'] = 30      # (1.0 + 2.0) * 10 = 30

    if initial_params:
        for key in defaults:
            if key in initial_params:
                defaults[key] = initial_params[key]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video for Radon preview: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Start at 10% into the video
    current_frame_idx = max(0, int(total_frames * 0.1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        current_frame_idx = 0
        if not ret:
            print("Error: Could not read any frame from video")
            cap.release()
            return None

    src_h, src_w = frame.shape[:2]

    # ── Slider definitions ──────────────────────────────────────────
    frame_v_min = 0
    frame_v_max = max(1, total_frames - 1)
    slider_defs = [
        ('radon_snr_threshold', 'Radon SNR',  lambda v: f"{v/10:.1f}", 10, 80),
        ('pcf_ratio_threshold', 'PCF Ratio',   lambda v: f"{v/10:.1f}", 5, 50),
        ('star_mask_sigma',     'Star Mask σ', lambda v: f"{v/10:.1f}", 20, 100),
        ('lsd_log_eps',         'LSD Signif.', lambda v: f"{(v - 20)/10:.1f}", 0, 70),
        ('pcf_kernel_len',      'PCF Kernel',  lambda v: f"{v if v%2==1 else v+1}", 5, 81),
        ('min_streak_length',   'Min Length',  lambda v: f"{v}px", 10, 200),
    ]

    params = defaults.copy()
    params['frame_idx'] = current_frame_idx

    # ── Detect screen size ──────────────────────────────────────────
    _screen_w, _screen_h = 1920, 1080
    try:
        import subprocess as _sp
        _xdpy = _sp.run(['xdpyinfo'], capture_output=True, text=True, timeout=2)
        for _line in _xdpy.stdout.split('\n'):
            if 'dimensions:' in _line:
                _sw, _sh = _line.split()[1].split('x')
                _screen_w, _screen_h = int(_sw), int(_sh)
                break
    except Exception:
        try:
            import subprocess as _sp
            _xr = _sp.run(['xrandr', '--current'], capture_output=True, text=True, timeout=2)
            for _line in _xr.stdout.split('\n'):
                if ' connected ' in _line and 'x' in _line:
                    import re as _re
                    _m = _re.search(r'(\d{3,5})x(\d{3,5})', _line)
                    if _m:
                        _screen_w, _screen_h = int(_m.group(1)), int(_m.group(2))
                        break
        except Exception:
            pass

    # ── Layout constants (two-row design) ───────────────────────────
    sidebar_w = max(280, min(380, int(_screen_w * 0.18)))
    status_bar_h = 56
    gap = 2

    orig_w = src_w
    orig_h = src_h
    _max_orig_w = _screen_w - sidebar_w - 40
    _max_orig_h = int((_screen_h - status_bar_h - gap - 60) / 1.3)
    _fit_scale = min(1.0, _max_orig_w / orig_w, _max_orig_h / orig_h)
    if _fit_scale < 1.0:
        orig_w = int(orig_w * _fit_scale)
        orig_h = int(orig_h * _fit_scale)

    # Bottom row: 4 panels side-by-side
    bottom_h = int(orig_h * 0.3)
    small_w = (orig_w + sidebar_w - 3 * gap) // 4
    small_h = bottom_h
    canvas_w = orig_w + sidebar_w
    canvas_h = orig_h + gap + bottom_h + status_bar_h
    slider_row_h = 52
    slider_pad_x = 20
    slider_track_h = 5
    slider_thumb_r = 8
    slider_section_top = 72

    # Mutable state for mouse interaction
    dragging = {'idx': -1}
    slider_regions = []
    mouse_pos = [0, 0]

    # ── Documentation overlay state ───────────────────────────────────
    doc_state = {
        'visible': False,
        'page': 0,
        'konami_unlocked': False,
        'key_buffer': [],
        'flash_timer': 0,
        'title_hovered': False,
        'regions': None,
        'tag_regions': [],
    }
    doc_pages = _RADON_DOC_PAGES

    # ── Window setup ────────────────────────────────────────────────
    window_name = "Mnemosky  -  Radon Pipeline Debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ── Helper drawing functions ────────────────────────────────────

    def _fill_rect(img, x, y, w, h, color):
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

    def _draw_border(img, x, y, w, h, color, thickness=1):
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, thickness)

    def _put_text(img, text, x, y, color, scale=0.42, thickness=1):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    def _draw_tag(img, text, x, y, bg_color, text_color, scale=0.38):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        pad_x, pad_y = 6, 4
        _fill_rect(img, x, y - th - pad_y, tw + pad_x * 2, th + pad_y * 2, bg_color)
        _put_text(img, text, x + pad_x, y - 1, text_color, scale)

    # ── Radon pipeline computation ──────────────────────────────────

    # Cache for expensive Radon computation
    _radon_cache = {
        'key': None,
        'residual': None, 'star_mask': None,
        'sinogram_vis': None, 'peak_coords': None,
        'lsd_segments': None, 'lsd_enhanced': None,
        'radon_candidates': None, 'pcf_confirmed': None,
        'pcf_rejected': None,
    }

    def compute_radon_pipeline(gray_frm, p):
        """Run the full Radon pipeline and cache intermediate results."""
        cache_key = (
            current_frame_idx,
            p['radon_snr_threshold'], p['pcf_ratio_threshold'],
            p['star_mask_sigma'], p['lsd_log_eps'],
            p['pcf_kernel_len'], p['min_streak_length'],
        )
        if _radon_cache['key'] == cache_key:
            return

        h, w = gray_frm.shape
        snr_thresh = p['radon_snr_threshold'] / 10.0
        pcf_ratio = p['pcf_ratio_threshold'] / 10.0
        star_sigma = p['star_mask_sigma'] / 10.0
        lsd_eps = (p['lsd_log_eps'] - 20) / 10.0
        pcf_kern = p['pcf_kernel_len']
        if pcf_kern % 2 == 0:
            pcf_kern += 1
        min_len = p['min_streak_length']

        # ── Background subtraction & noise ──────────────────────────
        bg_kernel = min(51, max(3, min(h, w) // 8))
        if bg_kernel % 2 == 0:
            bg_kernel += 1
        bg = cv2.medianBlur(gray_frm, bg_kernel).astype(np.float64)
        residual = gray_frm.astype(np.float64) - bg
        flat = residual.ravel()
        mad = np.median(np.abs(flat - np.median(flat)))
        noise_sigma = max(0.5, mad * 1.4826)

        # ── Star masking (tunable sigma) ────────────────────────────
        high_thresh = star_sigma * noise_sigma
        star_mask_raw = residual > high_thresh
        dilate_size = max(5, int(noise_sigma * 3))
        if dilate_size % 2 == 0:
            dilate_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        star_mask = cv2.dilate(
            star_mask_raw.astype(np.uint8), kernel).astype(bool)
        cleaned = residual.copy()
        cleaned[star_mask] = 0.0

        # ── Downsample for Radon ────────────────────────────────────
        max_area = 500000
        current_area = h * w
        r_scale = min(1.0, np.sqrt(max_area / max(1, current_area)))
        small_h_r = max(64, int(h * r_scale))
        small_w_r = max(64, int(w * r_scale))
        small_cleaned = cv2.resize(
            cleaned.astype(np.float32), (small_w_r, small_h_r),
            interpolation=cv2.INTER_AREA).astype(np.float64)
        small_noise = noise_sigma * r_scale

        # ── Radon transform ─────────────────────────────────────────
        num_angles = 90
        diag = int(np.ceil(np.sqrt(small_h_r**2 + small_w_r**2)))
        pad_h_r = (diag - small_h_r) // 2
        pad_w_r = (diag - small_w_r) // 2
        padded = np.zeros((diag, diag), dtype=np.float32)
        padded[pad_h_r:pad_h_r + small_h_r,
               pad_w_r:pad_w_r + small_w_r] = small_cleaned.astype(np.float32)

        angles = np.linspace(0, 180, num_angles, endpoint=False)
        sinogram = np.zeros((diag, num_angles), dtype=np.float32)
        center = (diag / 2.0, diag / 2.0)

        for i, theta in enumerate(angles):
            M = cv2.getRotationMatrix2D(center, -theta, 1.0)
            rotated = cv2.warpAffine(
                padded, M, (diag, diag),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            sinogram[:, i] = np.sum(rotated, axis=0)
        sinogram = sinogram.astype(np.float64)

        # SNR normalisation
        n_pixels = max(1.0, float(min(small_h_r, small_w_r)))
        noise_per_proj = small_noise * np.sqrt(n_pixels)
        snr_sinogram = sinogram / max(1e-10, noise_per_proj)

        # Baseline removal
        blur_k_s = min(51, max(3, diag // 10))
        if blur_k_s % 2 == 0:
            blur_k_s += 1
        baseline = cv2.GaussianBlur(
            snr_sinogram.astype(np.float32),
            (blur_k_s, 1), 0).astype(np.float64)
        snr_sinogram = snr_sinogram - baseline

        # Peak finding
        peak_mask_arr = snr_sinogram > snr_thresh
        peak_coords_list = []
        radon_candidates = []

        if np.any(peak_mask_arr):
            if _HAS_SCIPY:
                local_max = _scipy_maximum_filter(snr_sinogram, size=(5, 5))
            else:
                kern_nms = np.ones((5, 5), dtype=np.uint8)
                local_max = cv2.dilate(
                    snr_sinogram.astype(np.float32),
                    kern_nms).astype(np.float64)
            peaks = peak_mask_arr & (snr_sinogram == local_max)
            pc = np.argwhere(peaks)
            if len(pc) > 0:
                peak_snrs = np.array(
                    [snr_sinogram[r, c] for r, c in pc])
                order = np.argsort(peak_snrs)[::-1][:20]
                pc = pc[order]
                peak_snrs = peak_snrs[order]
                peak_coords_list = list(pc)

                # Convert peaks to line segments
                center_offset = diag / 2.0
                for idx_p, (off_idx, ang_idx) in enumerate(pc):
                    theta_p = angles[ang_idx]
                    offset = float(off_idx) - center_offset
                    snr_val = peak_snrs[idx_p]
                    theta_rad = np.radians(theta_p)
                    cos_t = np.cos(theta_rad)
                    sin_t = np.sin(theta_rad)
                    cx = offset * cos_t
                    cy = offset * sin_t
                    col_profile = snr_sinogram[:, ang_idx]
                    half_max = snr_val / 2.0
                    streak_len_est = float(np.sum(col_profile >= half_max))
                    if streak_len_est < max(10, int(min_len * r_scale * 0.5)):
                        continue
                    half_len = streak_len_est / 2.0
                    x1 = cx - half_len * (-sin_t)
                    y1 = cy - half_len * cos_t
                    x2 = cx + half_len * (-sin_t)
                    y2 = cy + half_len * cos_t
                    x1 -= pad_w_r
                    y1 -= pad_h_r
                    x2 -= pad_w_r
                    y2 -= pad_h_r
                    # Scale back to full resolution
                    x1_f = np.clip(x1 / r_scale, 0, w - 1)
                    y1_f = np.clip(y1 / r_scale, 0, h - 1)
                    x2_f = np.clip(x2 / r_scale, 0, w - 1)
                    y2_f = np.clip(y2 / r_scale, 0, h - 1)
                    seg_len = np.sqrt(
                        (x2_f - x1_f)**2 + (y2_f - y1_f)**2)
                    if seg_len >= min_len:
                        radon_candidates.append(
                            (float(x1_f), float(y1_f),
                             float(x2_f), float(y2_f), snr_val))

        # ── PCF filtering ───────────────────────────────────────────
        half_w_pcf = pcf_kern // 2
        pcf_confirmed = []
        pcf_rejected = []
        for x1, y1, x2, y2, score in radon_candidates:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            if length < 5:
                pcf_rejected.append((x1, y1, x2, y2, score))
                continue
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            num_samples = max(5, min(20, int(length / 10)))
            par_sum = 0.0
            perp_sum = 0.0
            count = 0
            for i_s in range(num_samples):
                t = (i_s + 1) / (num_samples + 1)
                cx_s = x1 + t * dx
                cy_s = y1 + t * dy
                par_val = 0.0
                par_n = 0
                for d in range(-half_w_pcf, half_w_pcf + 1):
                    px = int(round(cx_s + d * ux))
                    py = int(round(cy_s + d * uy))
                    if 0 <= py < h and 0 <= px < w:
                        par_val += cleaned[py, px]
                        par_n += 1
                perp_val = 0.0
                perp_n = 0
                for d in range(-half_w_pcf, half_w_pcf + 1):
                    px = int(round(cx_s + d * nx))
                    py = int(round(cy_s + d * ny))
                    if 0 <= py < h and 0 <= px < w:
                        perp_val += cleaned[py, px]
                        perp_n += 1
                if par_n > 0 and perp_n > 0:
                    par_sum += par_val / par_n
                    perp_sum += perp_val / perp_n
                    count += 1
            if count == 0:
                pcf_rejected.append((x1, y1, x2, y2, score))
                continue
            mean_par = par_sum / count
            mean_perp = perp_sum / count
            ratio = mean_perp / (abs(mean_par) + 1e-10)
            if ratio >= pcf_ratio:
                pcf_confirmed.append((x1, y1, x2, y2, score))
            else:
                pcf_rejected.append((x1, y1, x2, y2, score))

        # ── LSD detection ───────────────────────────────────────────
        lsd_segments = []
        lsd_enhanced = gray_frm.copy()
        if hasattr(cv2, 'createLineSegmentDetector'):
            clahe_lsd = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(6, 6))
            lsd_enhanced = clahe_lsd.apply(gray_frm)
            lsd = cv2.createLineSegmentDetector(
                refine=cv2.LSD_REFINE_ADV,
                scale=0.8, sigma_scale=0.6, quant=2.0,
                ang_th=22.5, log_eps=lsd_eps,
                density_th=0.5, n_bins=1024)
            lines, widths, precisions, nfas = lsd.detect(lsd_enhanced)
            if lines is not None:
                for i_l, line in enumerate(lines):
                    lx1, ly1, lx2, ly2 = line[0]
                    ll = np.sqrt((lx2 - lx1)**2 + (ly2 - ly1)**2)
                    if ll >= min_len * 0.6:
                        nfa_val = nfas[i_l][0] if nfas is not None else 0.0
                        lsd_segments.append(
                            (float(lx1), float(ly1),
                             float(lx2), float(ly2), nfa_val))
                lsd_segments.sort(key=lambda r: r[4])

        # ── Build sinogram visualisation ────────────────────────────
        sino_vis = np.zeros(
            (snr_sinogram.shape[0], snr_sinogram.shape[1], 3),
            dtype=np.uint8)
        sino_vis[:] = BG_PANEL
        # Normalise SNR sinogram for display
        s_max = max(snr_thresh * 2, np.max(snr_sinogram)) if snr_sinogram.size > 0 else 1.0
        has_signal_s = snr_sinogram > 1.0
        below_s = has_signal_s & (snr_sinogram < snr_thresh)
        above_s = snr_sinogram >= snr_thresh

        _bg_arr = np.array(BG_PANEL, dtype=np.float32)
        _dim_s = np.array(ACCENT_SINOGRAM, dtype=np.float32) * 0.4
        _bright_s = np.array(ACCENT_SINOGRAM, dtype=np.float32)

        if np.any(below_s):
            intensity = np.clip(
                snr_sinogram[below_s] / snr_thresh, 0, 1)
            blend = _bg_arr + (_dim_s - _bg_arr) * intensity[:, np.newaxis]
            sino_vis[below_s] = np.clip(blend, 0, 255).astype(np.uint8)
        if np.any(above_s):
            intensity = np.clip(
                snr_sinogram[above_s] / s_max, 0.3, 1)
            blend = _dim_s + (_bright_s - _dim_s) * intensity[:, np.newaxis]
            sino_vis[above_s] = np.clip(blend, 0, 255).astype(np.uint8)

        # Store results in cache
        _radon_cache['key'] = cache_key
        _radon_cache['residual'] = cleaned
        _radon_cache['star_mask'] = star_mask
        _radon_cache['sinogram_vis'] = sino_vis
        _radon_cache['peak_coords'] = peak_coords_list
        _radon_cache['lsd_segments'] = lsd_segments
        _radon_cache['lsd_enhanced'] = lsd_enhanced
        _radon_cache['radon_candidates'] = radon_candidates
        _radon_cache['pcf_confirmed'] = pcf_confirmed
        _radon_cache['pcf_rejected'] = pcf_rejected

    # ── Composite display builder ───────────────────────────────────

    def create_display(frm, gray_frm, p):
        nonlocal slider_regions

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = BG_DARK

        # ── Run pipeline ────────────────────────────────────────────
        compute_radon_pipeline(gray_frm, p)

        # ── Top row: Original panel ─────────────────────────────────
        orig_panel = cv2.resize(frm, (orig_w, orig_h))
        canvas[0:orig_h, 0:orig_w] = orig_panel
        _draw_border(canvas, 0, 0, orig_w, orig_h, BORDER)

        bot_y = orig_h + gap

        # ── Panel 1: Residual (star-cleaned) ────────────────────────
        res = _radon_cache['residual']
        smask = _radon_cache['star_mask']
        # Normalise residual for display
        r_abs = np.abs(res)
        r_max = max(1.0, np.percentile(r_abs, 99.5))
        res_norm = np.clip(r_abs / r_max, 0, 1)
        res_bgr = np.zeros((res.shape[0], res.shape[1], 3), dtype=np.uint8)
        # Teal tint
        res_bgr[:, :, 0] = (res_norm * ACCENT_RESIDUAL[0]).astype(np.uint8)
        res_bgr[:, :, 1] = (res_norm * ACCENT_RESIDUAL[1]).astype(np.uint8)
        res_bgr[:, :, 2] = (res_norm * ACCENT_RESIDUAL[2]).astype(np.uint8)
        # Overlay star mask
        if smask is not None:
            res_bgr[smask] = ACCENT_RESIDUAL_STAR
        res_panel = cv2.resize(res_bgr, (small_w, small_h))

        # ── Panel 2: Sinogram ───────────────────────────────────────
        sino_vis = _radon_cache['sinogram_vis']
        sino_panel = cv2.resize(sino_vis, (small_w, small_h),
                                interpolation=cv2.INTER_NEAREST)
        # Mark peaks
        if _radon_cache['peak_coords']:
            sino_h_orig, sino_w_orig = sino_vis.shape[:2]
            for pc in _radon_cache['peak_coords']:
                py_s = int(pc[0] * small_h / sino_h_orig)
                px_s = int(pc[1] * small_w / sino_w_orig)
                cv2.circle(sino_panel, (px_s, py_s), 4,
                           ACCENT_SINOGRAM_PEAK, 1, cv2.LINE_AA)
                cv2.circle(sino_panel, (px_s, py_s), 1,
                           ACCENT_SINOGRAM_PEAK, -1)

        # ── Panel 3: LSD Lines ──────────────────────────────────────
        lsd_enh = _radon_cache['lsd_enhanced']
        lsd_bgr = cv2.cvtColor(
            cv2.resize(lsd_enh, (small_w, small_h)), cv2.COLOR_GRAY2BGR)
        # Dim the background slightly
        lsd_bgr = (lsd_bgr.astype(np.float32) * 0.5).astype(np.uint8)
        lsd_scale_x = small_w / src_w
        lsd_scale_y = small_h / src_h
        for lx1, ly1, lx2, ly2, _ in _radon_cache['lsd_segments'][:30]:
            pt1 = (int(lx1 * lsd_scale_x), int(ly1 * lsd_scale_y))
            pt2 = (int(lx2 * lsd_scale_x), int(ly2 * lsd_scale_y))
            cv2.line(lsd_bgr, pt1, pt2, ACCENT_LSD, 2, cv2.LINE_AA)

        # ── Panel 4: Detections (Radon + PCF overlay on original) ───
        det_bgr = cv2.resize(frm, (small_w, small_h))
        det_bgr = (det_bgr.astype(np.float32) * 0.4).astype(np.uint8)
        det_scale_x = small_w / src_w
        det_scale_y = small_h / src_h

        # Rejected candidates (dim red dashes)
        for x1, y1, x2, y2, _ in _radon_cache['pcf_rejected']:
            pt1 = (int(x1 * det_scale_x), int(y1 * det_scale_y))
            pt2 = (int(x2 * det_scale_x), int(y2 * det_scale_y))
            # Dashed line effect: draw short segments
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            seg_l = max(1, int(np.sqrt(dx*dx + dy*dy)))
            for d in range(0, seg_l, 8):
                t1 = d / seg_l
                t2 = min((d + 4) / seg_l, 1.0)
                sp = (int(pt1[0] + t1 * dx), int(pt1[1] + t1 * dy))
                ep = (int(pt1[0] + t2 * dx), int(pt1[1] + t2 * dy))
                cv2.line(det_bgr, sp, ep, ACCENT_DET_REJECT, 1, cv2.LINE_AA)

        # Raw Radon candidates (amber, thin)
        for x1, y1, x2, y2, _ in _radon_cache['radon_candidates']:
            pt1 = (int(x1 * det_scale_x), int(y1 * det_scale_y))
            pt2 = (int(x2 * det_scale_x), int(y2 * det_scale_y))
            cv2.line(det_bgr, pt1, pt2, ACCENT_DET_RADON, 1, cv2.LINE_AA)

        # PCF-confirmed (bright green, thick)
        for x1, y1, x2, y2, _ in _radon_cache['pcf_confirmed']:
            pt1 = (int(x1 * det_scale_x), int(y1 * det_scale_y))
            pt2 = (int(x2 * det_scale_x), int(y2 * det_scale_y))
            cv2.line(det_bgr, pt1, pt2, ACCENT_DET_PCF, 2, cv2.LINE_AA)

        # ── Place 4 bottom panels ───────────────────────────────────
        for px, py, pw, ph, panel in [
            (0, bot_y, small_w, small_h, res_panel),
            (small_w + gap, bot_y, small_w, small_h, sino_panel),
            (2 * (small_w + gap), bot_y, small_w, small_h, lsd_bgr),
            (3 * (small_w + gap), bot_y, small_w, small_h, det_bgr),
        ]:
            # Ensure panel fits the target region
            ph_actual = min(ph, panel.shape[0])
            pw_actual = min(pw, panel.shape[1])
            canvas[py:py + ph_actual, px:px + pw_actual] = \
                panel[:ph_actual, :pw_actual]
            _draw_border(canvas, px, py, pw, ph, BORDER)

        # ── Panel tags (with bbox tracking for right-click doc access) ─
        tag_y_off, tag_x_off = 18, 8
        _tag_regions = []
        _draw_tag(canvas, "ORIGINAL", tag_x_off, tag_y_off,
                  BG_DARK, TEXT_DIM)
        _tag_regions.append((_doc_tag_bbox("ORIGINAL", tag_x_off, tag_y_off), 1))

        n_stars = int(np.sum(_radon_cache['star_mask'])) if _radon_cache['star_mask'] is not None else 0
        star_pct = n_stars / max(1, src_h * src_w) * 100
        res_tag = f"RESIDUAL  stars {star_pct:.1f}%"
        _draw_tag(canvas, res_tag,
                  tag_x_off, bot_y + tag_y_off, BG_DARK, ACCENT_RESIDUAL)
        _tag_regions.append((_doc_tag_bbox(res_tag, tag_x_off, bot_y + tag_y_off), 1))

        n_peaks = len(_radon_cache['peak_coords'])
        snr_v = p['radon_snr_threshold'] / 10.0
        sino_tag = f"SINOGRAM  SNR>={snr_v:.1f}  [{n_peaks}]"
        _draw_tag(canvas, sino_tag,
                  small_w + gap + tag_x_off, bot_y + tag_y_off,
                  BG_DARK, ACCENT_SINOGRAM)
        _tag_regions.append((_doc_tag_bbox(sino_tag, small_w + gap + tag_x_off, bot_y + tag_y_off), 1))

        n_lsd = len(_radon_cache['lsd_segments'])
        eps_v = (p['lsd_log_eps'] - 20) / 10.0
        lsd_tag = f"LSD  eps={eps_v:.1f}  [{n_lsd}]"
        _draw_tag(canvas, lsd_tag,
                  2 * (small_w + gap) + tag_x_off, bot_y + tag_y_off,
                  BG_DARK, ACCENT_LSD)
        _tag_regions.append((_doc_tag_bbox(lsd_tag, 2 * (small_w + gap) + tag_x_off, bot_y + tag_y_off), 1))

        n_conf = len(_radon_cache['pcf_confirmed'])
        n_rej = len(_radon_cache['pcf_rejected'])
        det_tag = f"DETECTIONS  {n_conf} ok  {n_rej} rej"
        _draw_tag(canvas, det_tag,
                  3 * (small_w + gap) + tag_x_off, bot_y + tag_y_off,
                  BG_DARK, ACCENT_DET_PCF)
        _tag_regions.append((_doc_tag_bbox(det_tag, 3 * (small_w + gap) + tag_x_off, bot_y + tag_y_off), 1))
        doc_state['tag_regions'] = _tag_regions

        # ── Sidebar ─────────────────────────────────────────────────
        sb_x = orig_w
        _fill_rect(canvas, sb_x, 0, sidebar_w,
                   canvas_h - status_bar_h, BG_SIDEBAR)
        _draw_border(canvas, sb_x, 0, sidebar_w,
                     canvas_h - status_bar_h, BORDER)

        # Title (with hover hint for doc overlay)
        _title_x, _title_y = sb_x + 14, 24
        _title_bbox = _doc_compute_title_bbox("MNEMOSKY", _title_x, _title_y, 0.52)
        doc_state['title_bbox'] = _title_bbox
        if doc_state['title_hovered']:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, (230, 255, 140), 0.52, 1)
            _put_text(canvas, "(?)", _title_x + _title_bbox[2] + 4, _title_y, (100, 170, 50), 0.36)
        else:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, ACCENT, 0.52, 1)
        _put_text(canvas, "Radon Debug", sb_x + 14, 46,
                  TEXT_HEADING, 0.40)
        cv2.line(canvas, (sb_x + 14, 56),
                 (sb_x + sidebar_w - 14, 56), BORDER, 1)

        # ── Draw sliders ────────────────────────────────────────────
        new_regions = []
        track_x_start = sb_x + slider_pad_x
        track_x_end = sb_x + sidebar_w - slider_pad_x
        track_width = track_x_end - track_x_start

        for i, (key, label, fmt_fn, v_min, v_max) in enumerate(slider_defs):
            row_y = slider_section_top + i * slider_row_h
            val = p[key]
            _put_text(canvas, label, track_x_start, row_y + 14,
                      TEXT_DIM, 0.34)
            _put_text(canvas, fmt_fn(val), track_x_end - 46,
                      row_y + 14, ACCENT, 0.36, 1)
            track_y = row_y + 28
            _fill_rect(canvas, track_x_start,
                       track_y - slider_track_h // 2,
                       track_width, slider_track_h, SLIDER_TRACK)
            ratio = (val - v_min) / max(1, v_max - v_min)
            fill_w = int(track_width * ratio)
            if fill_w > 0:
                _fill_rect(canvas, track_x_start,
                           track_y - slider_track_h // 2,
                           fill_w, slider_track_h, SLIDER_FILL)
            thumb_x = track_x_start + fill_w
            cv2.circle(canvas, (thumb_x, track_y),
                       slider_thumb_r, SLIDER_THUMB, -1)
            cv2.circle(canvas, (thumb_x, track_y),
                       slider_thumb_r, ACCENT, 1, cv2.LINE_AA)
            new_regions.append(
                (track_x_start, track_x_end, track_y,
                 v_min, v_max, key))

        # ── Pipeline stats section ──────────────────────────────────
        stats_y = slider_section_top + len(slider_defs) * slider_row_h + 12
        cv2.line(canvas, (sb_x + 14, stats_y - 6),
                 (sb_x + sidebar_w - 14, stats_y - 6), BORDER, 1)
        _put_text(canvas, "PIPELINE STATS", sb_x + 14,
                  stats_y + 10, TEXT_HEADING, 0.36)
        stats_y += 26
        for label_s, val_s in [
            ("Radon candidates", str(len(_radon_cache['radon_candidates']))),
            ("PCF confirmed", str(len(_radon_cache['pcf_confirmed']))),
            ("PCF rejected", str(len(_radon_cache['pcf_rejected']))),
            ("LSD segments", str(len(_radon_cache['lsd_segments']))),
            ("Sinogram peaks", str(len(_radon_cache['peak_coords']))),
        ]:
            _put_text(canvas, label_s, sb_x + 14, stats_y,
                      TEXT_DIM, 0.30)
            _put_text(canvas, val_s, sb_x + sidebar_w - 50,
                      stats_y, ACCENT, 0.32, 1)
            stats_y += 16
        stats_y += 8

        # ── Controls help ───────────────────────────────────────────
        cv2.line(canvas, (sb_x + 14, stats_y - 6),
                 (sb_x + sidebar_w - 14, stats_y - 6), BORDER, 1)
        _put_text(canvas, "CONTROLS", sb_x + 14, stats_y + 10,
                  TEXT_HEADING, 0.36)
        stats_y += 26
        for key_str, desc in [
            ("SPACE / ENTER", "Accept"),
            ("ESC", "Cancel"),
            ("R", "Reset"),
            ("N / P", "Next / Prev frame"),
        ]:
            _put_text(canvas, key_str, sb_x + 14, stats_y,
                      ACCENT_DIM, 0.30)
            _put_text(canvas, desc, sb_x + 126, stats_y,
                      TEXT_DIM, 0.30)
            stats_y += 16

        # ── Status bar with frame slider ────────────────────────────
        sb_y = canvas_h - status_bar_h
        _fill_rect(canvas, 0, sb_y, canvas_w, status_bar_h, BG_PANEL)
        cv2.line(canvas, (0, sb_y), (canvas_w, sb_y), BORDER, 1)

        _put_text(canvas, f"Frame {current_frame_idx}/{total_frames}",
                  12, sb_y + 16, TEXT_DIM, 0.36)
        _put_text(canvas, f"{src_w}x{src_h}",
                  canvas_w - 90, sb_y + 16, TEXT_DIM, 0.36)
        cv2.circle(canvas, (canvas_w // 2, sb_y + 11), 4, ACCENT, -1)
        _put_text(canvas, "RADON", canvas_w // 2 + 10, sb_y + 16,
                  ACCENT_DIM, 0.33)

        frame_track_pad = 14
        frame_track_x0 = frame_track_pad
        frame_track_x1 = canvas_w - frame_track_pad
        frame_track_y = sb_y + 40
        frame_track_w = frame_track_x1 - frame_track_x0

        _fill_rect(canvas, frame_track_x0,
                   frame_track_y - slider_track_h // 2,
                   frame_track_w, slider_track_h, SLIDER_TRACK)
        f_ratio = ((p['frame_idx'] - frame_v_min)
                   / max(1, frame_v_max - frame_v_min))
        f_fill_w = int(frame_track_w * f_ratio)
        if f_fill_w > 0:
            _fill_rect(canvas, frame_track_x0,
                       frame_track_y - slider_track_h // 2,
                       f_fill_w, slider_track_h, SLIDER_FILL)
        f_thumb_x = frame_track_x0 + f_fill_w
        cv2.circle(canvas, (f_thumb_x, frame_track_y),
                   slider_thumb_r, SLIDER_THUMB, -1)
        cv2.circle(canvas, (f_thumb_x, frame_track_y),
                   slider_thumb_r, ACCENT, 1, cv2.LINE_AA)

        new_regions.append(
            (frame_track_x0, frame_track_x1, frame_track_y,
             frame_v_min, frame_v_max, 'frame_idx'))

        slider_regions = new_regions

        # "UNLOCKED" flash on status bar
        if doc_state['flash_timer'] and time.time() - doc_state['flash_timer'] < 2.0:
            _put_text(canvas, "UNLOCKED", canvas_w // 2 - 40, sb_y + 16, ACCENT, 0.40, 1)

        # ── Documentation overlay ─────────────────────────────────────
        if doc_state['visible']:
            doc_state['regions'] = _draw_doc_overlay(
                canvas, doc_state['page'], doc_pages,
                doc_state['konami_unlocked'])

        return canvas

    # ── Mouse callback ──────────────────────────────────────────────

    def _update_slider_from_x(mouse_x, mouse_y):
        for i, (x_start, x_end, y_center, v_min, v_max, key) in enumerate(slider_regions):
            if (abs(mouse_y - y_center) < slider_row_h // 2
                    and x_start - 4 <= mouse_x <= x_end + 4):
                ratio = max(0.0, min(1.0,
                    (mouse_x - x_start) / max(1, x_end - x_start)))
                params[key] = int(round(v_min + ratio * (v_max - v_min)))
                dragging['idx'] = i
                return
        dragging['idx'] = -1

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = x
            mouse_pos[1] = y
            # Title hover detection
            if doc_state.get('title_bbox'):
                doc_state['title_hovered'] = _doc_point_in_rect(
                    x, y, doc_state['title_bbox'])

        # When doc overlay is visible, handle overlay clicks only
        if doc_state['visible']:
            if event == cv2.EVENT_LBUTTONDOWN:
                regions = doc_state.get('regions')
                if regions:
                    if regions['prev_arrow'] and _doc_point_in_rect(x, y, regions['prev_arrow']):
                        if doc_state['page'] > 0:
                            doc_state['page'] -= 1
                        return
                    if regions['next_arrow'] and _doc_point_in_rect(x, y, regions['next_arrow']):
                        vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                        if doc_state['page'] < len(vis) - 1:
                            doc_state['page'] += 1
                        return
                    if not _doc_point_in_rect(x, y, regions['card']):
                        doc_state['visible'] = False
                        return
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check MNEMOSKY title click
            if doc_state.get('title_bbox') and _doc_point_in_rect(x, y, doc_state['title_bbox']):
                doc_state['visible'] = not doc_state['visible']
                doc_state['page'] = 0
                return
            _update_slider_from_x(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            for tag_bbox, page_idx in doc_state.get('tag_regions', []):
                if _doc_point_in_rect(x, y, tag_bbox):
                    doc_state['visible'] = True
                    doc_state['page'] = page_idx
                    return
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            if dragging['idx'] >= 0:
                x_start, x_end, _, v_min, v_max, key = \
                    slider_regions[dragging['idx']]
                ratio = max(0.0, min(1.0,
                    (x - x_start) / max(1, x_end - x_start)))
                params[key] = int(round(v_min + ratio * (v_max - v_min)))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging['idx'] = -1

    cv2.setMouseCallback(window_name, on_mouse)

    print("\n" + "=" * 60)
    print("RADON PIPELINE DEBUG PREVIEW")
    print("=" * 60)
    print("Use the Frame slider to find a frame with satellite trail signal.")
    print("Adjust sliders to tune the Radon detection pipeline parameters.")
    print()
    print("Bottom panels (left to right):")
    print("  RESIDUAL  — Background-subtracted, star-cleaned image (teal)")
    print("  SINOGRAM  — Radon SNR heatmap; peaks = detected streaks (amber)")
    print("  LSD LINES — LSD line segments overlaid on enhanced frame (green)")
    print("  DETECTIONS — Final results: green=PCF ok, amber=raw, red=rejected")
    print()
    print("Controls:")
    print("  Click+drag   - Adjust sliders in the sidebar")
    print("  SPACE/ENTER  - Accept current settings and continue")
    print("  ESC          - Cancel and use default settings")
    print("  R            - Reset to default values")
    print("  N            - Jump forward 1 second")
    print("  P            - Jump back 1 second")
    print("=" * 60 + "\n")

    first_render = True
    _prev_cache_key = None
    _cached_display = None

    # ── Main loop ───────────────────────────────────────────────────
    while True:
        # Check if Frame slider was dragged
        if params['frame_idx'] != current_frame_idx:
            current_frame_idx = params['frame_idx']
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Dirty-flag: recompute only when parameters or frame change
        cache_key = (
            current_frame_idx,
            params['radon_snr_threshold'], params['pcf_ratio_threshold'],
            params['star_mask_sigma'], params['lsd_log_eps'],
            params['pcf_kernel_len'], params['min_streak_length'],
            doc_state['visible'], doc_state['page'],
            doc_state['title_hovered'], doc_state['konami_unlocked'],
        )

        needs_redraw = (cache_key != _prev_cache_key) or first_render

        if needs_redraw:
            _cached_display = create_display(frame, gray, params)
            _prev_cache_key = cache_key

            cv2.imshow(window_name, _cached_display)

            if first_render:
                cv2.resizeWindow(window_name, canvas_w, canvas_h)
                first_render = False

        key = cv2.waitKey(30) & 0xFF

        # ── Secret word tracking (always active) ─────────────────────
        if key != 255 and 97 <= key <= 122:
            doc_state['key_buffer'].append(chr(key))
            doc_state['key_buffer'] = doc_state['key_buffer'][-len(_DOC_SECRET_WORD):]
            if ''.join(doc_state['key_buffer']) == _DOC_SECRET_WORD:
                doc_state['konami_unlocked'] = True
                doc_state['visible'] = True
                vis = [p for p in doc_pages if not p.get('konami_only') or True]
                doc_state['page'] = len(vis) - 1
                doc_state['flash_timer'] = time.time()
                _prev_cache_key = None
                continue

        # ── Doc overlay key handling ──────────────────────────────────
        if doc_state['visible']:
            if key == 27:
                doc_state['visible'] = False
                _prev_cache_key = None
            elif key in (ord('a'), ord('A')):
                if doc_state['page'] > 0:
                    doc_state['page'] -= 1
                    _prev_cache_key = None
            elif key in (ord('d'), ord('D')):
                vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                if doc_state['page'] < len(vis) - 1:
                    doc_state['page'] += 1
                    _prev_cache_key = None
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Radon preview window closed. Using default parameters.")
                cap.release()
                return None
            continue

        if key == 63:  # ? (Shift+/)
            doc_state['visible'] = True
            doc_state['page'] = 0
            _prev_cache_key = None
            continue

        if key == 27:  # ESC
            print("Radon preview cancelled. Using default parameters.")
            cv2.destroyWindow(window_name)
            cap.release()
            return None

        elif key in [13, 32]:  # ENTER or SPACE
            final_params = {
                'radon_snr_threshold': params['radon_snr_threshold'] / 10.0,
                'pcf_ratio_threshold': params['pcf_ratio_threshold'] / 10.0,
                'star_mask_sigma': params['star_mask_sigma'] / 10.0,
                'lsd_log_eps': (params['lsd_log_eps'] - 20) / 10.0,
                'pcf_kernel_len': params['pcf_kernel_len'] if params['pcf_kernel_len'] % 2 == 1 else params['pcf_kernel_len'] + 1,
                'min_streak_length': params['min_streak_length'],
            }
            print(f"\nAccepted Radon pipeline parameters:")
            print(f"  Radon SNR threshold: {final_params['radon_snr_threshold']:.1f}")
            print(f"  PCF ratio threshold: {final_params['pcf_ratio_threshold']:.1f}")
            print(f"  Star mask sigma:     {final_params['star_mask_sigma']:.1f}")
            print(f"  LSD log_eps:         {final_params['lsd_log_eps']:.1f}")
            print(f"  PCF kernel length:   {final_params['pcf_kernel_len']}")
            print(f"  Min streak length:   {final_params['min_streak_length']}px")
            cv2.destroyWindow(window_name)
            cap.release()
            return final_params

        elif key == ord('r') or key == ord('R'):
            saved_frame_idx = params['frame_idx']
            params = defaults.copy()
            params['frame_idx'] = saved_frame_idx
            _radon_cache['key'] = None
            _prev_cache_key = None
            print("Radon parameters reset to defaults.")

        elif key == ord('n') or key == ord('N'):
            current_frame_idx = min(total_frames - 1,
                                    current_frame_idx + int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if not ret:
                current_frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                params['frame_idx'] = current_frame_idx

        elif key == ord('p') or key == ord('P'):
            current_frame_idx = max(0, current_frame_idx - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                params['frame_idx'] = current_frame_idx

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Radon preview window closed. Using default parameters.")
            cap.release()
            return None

    cap.release()
    return None


def show_nn_preview(video_path, nn_params=None):
    """Show interactive preview for tuning neural network detection parameters.

    Uses the same dark-grey/fluorescent-accent theme as other preview windows.
    Shows live model detections with adjustable confidence/NMS thresholds.

    Layout:
    - Left column (~65%): Frame with detection overlays (GOLD=satellite, ORANGE=airplane)
    - Right sidebar (~35%): Model info card, sliders for confidence and NMS IoU,
      class mapping display, inference stats (FPS, detection count)
    - Bottom status bar: frame slider

    Controls:
    - Click and drag sliders in the sidebar to adjust parameters
    - SPACE or ENTER to accept current settings
    - ESC to cancel and use default settings
    - R to reset to defaults
    - N/P to jump forward/back 1 second
    - M to type a new model path (typed into status bar, ENTER to confirm)

    Args:
        video_path: Path to the input video file
        nn_params: Dict with NN parameters (model_path, backend, confidence, etc.)

    Returns:
        Dict with tuned nn parameters, or None if cancelled.
    """
    if nn_params is None or nn_params.get('model_path') is None:
        print("Error: --model is required for NN preview.")
        return None

    # ── Theme colours (BGR) — shared with other previews ─────────────
    BG_DARK = (30, 30, 30)
    BG_PANEL = (42, 42, 42)
    BG_SIDEBAR = (36, 36, 36)
    BORDER = (58, 58, 58)
    TEXT_PRIMARY = (210, 210, 210)
    TEXT_DIM = (120, 120, 120)
    TEXT_HEADING = (180, 180, 180)
    ACCENT = (200, 255, 80)           # Fluorescent green-yellow
    ACCENT_DIM = (100, 170, 50)
    SLIDER_TRACK = (50, 50, 50)
    SLIDER_FILL = (200, 255, 80)
    SLIDER_THUMB = (240, 255, 160)
    COLOR_SATELLITE = (0, 185, 255)   # GOLD (BGR)
    COLOR_AIRPLANE = (0, 140, 255)    # ORANGE (BGR)
    COLOR_CONF_BAR = (80, 200, 80)    # Green for confidence bars

    def _fill_rect(img, x, y, w, h, color):
        cv2.rectangle(img, (int(x), int(y)),
                      (int(x + w), int(y + h)), color, -1)

    def _put_text(img, text, x, y, color, scale=0.42, thickness=1):
        cv2.putText(img, str(text), (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_tag(img, text, x, y, bg_color, text_color, scale=0.38):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        pad_x, pad_y = 6, 3
        _fill_rect(img, x, y - th - pad_y, tw + pad_x * 2, th + pad_y * 2, bg_color)
        _put_text(img, text, x + pad_x, y - 1, text_color, scale)

    # ── Parameters (stored as ints for slider precision) ─────────────
    # Confidence: 5-95 (÷100 → 0.05-0.95)
    # NMS IoU:    10-90 (÷100 → 0.10-0.90)
    params = {
        'confidence': int(nn_params.get('confidence', 0.25) * 100),
        'nms_iou': int(nn_params.get('nms_iou', 0.45) * 100),
    }
    defaults = dict(params)

    # Slider definitions: (key, label, min_val, max_val, step)
    slider_defs = [
        ('confidence', 'Confidence', 5, 95, 5),
        ('nms_iou', 'NMS IoU', 10, 90, 5),
    ]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video for NN preview: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    current_frame_idx = 0

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Load model backend for live inference
    try:
        backend = _NNBackend(
            model_path=nn_params['model_path'],
            backend=nn_params['backend'],
            device=nn_params.get('device', 'auto'),
            confidence=nn_params.get('confidence', 0.25),
            nms_iou=nn_params.get('nms_iou', 0.45),
            input_size=nn_params.get('input_size', 640),
            half_precision=nn_params.get('half_precision', False),
            no_gpu=nn_params.get('device') == 'cpu',
        )
        model_info = backend.get_model_info()
    except Exception as e:
        print(f"Error loading model for NN preview: {e}")
        cap.release()
        return None

    # Build class_id → trail_type mapping
    class_map = nn_params.get('class_map', {'satellite': [0], 'airplane': [1]})
    cls_to_type = {}
    for ttype, cids in class_map.items():
        for cid in cids:
            cls_to_type[cid] = ttype

    # ── Canvas layout ────────────────────────────────────────────────
    WIN_W, WIN_H = 1280, 780
    STATUS_H = 56
    SIDEBAR_W = 280
    FRAME_W = WIN_W - SIDEBAR_W
    FRAME_H = WIN_H - STATUS_H

    window_name = 'Mnemosky - NN Detection Preview'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIN_W, WIN_H)

    # ── Mouse state ──────────────────────────────────────────────────
    mouse_state = {'x': 0, 'y': 0, 'down': False, 'active_slider': None}

    # ── Documentation overlay state ───────────────────────────────────
    doc_state = {
        'visible': False,
        'page': 0,
        'konami_unlocked': False,
        'key_buffer': [],
        'flash_timer': 0,
        'title_hovered': False,
        'regions': None,
        'tag_regions': [],
        'title_bbox': None,
    }
    doc_pages = _NN_DOC_PAGES

    def _mouse_callback(event, x, y, flags, _):
        mouse_state['x'] = x
        mouse_state['y'] = y

        # Title hover detection
        if event == cv2.EVENT_MOUSEMOVE and doc_state.get('title_bbox'):
            doc_state['title_hovered'] = _doc_point_in_rect(
                x, y, doc_state['title_bbox'])

        # When doc overlay is visible, handle overlay clicks only
        if doc_state['visible']:
            if event == cv2.EVENT_LBUTTONDOWN:
                regions = doc_state.get('regions')
                if regions:
                    if regions['prev_arrow'] and _doc_point_in_rect(x, y, regions['prev_arrow']):
                        if doc_state['page'] > 0:
                            doc_state['page'] -= 1
                        return
                    if regions['next_arrow'] and _doc_point_in_rect(x, y, regions['next_arrow']):
                        vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                        if doc_state['page'] < len(vis) - 1:
                            doc_state['page'] += 1
                        return
                    if not _doc_point_in_rect(x, y, regions['card']):
                        doc_state['visible'] = False
                        return
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Check MNEMOSKY title click
            if doc_state.get('title_bbox') and _doc_point_in_rect(x, y, doc_state['title_bbox']):
                doc_state['visible'] = not doc_state['visible']
                doc_state['page'] = 0
                return
            mouse_state['down'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            for tag_bbox, page_idx in doc_state.get('tag_regions', []):
                if _doc_point_in_rect(x, y, tag_bbox):
                    doc_state['visible'] = True
                    doc_state['page'] = page_idx
                    return
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_state['down'] = False
            mouse_state['active_slider'] = None

    cv2.setMouseCallback(window_name, _mouse_callback)

    # ── Main preview loop ────────────────────────────────────────────
    need_redraw = True
    last_detections = []
    inference_ms = 0.0

    while True:
        if need_redraw:
            # Update backend parameters from sliders
            backend.confidence = params['confidence'] / 100.0
            backend.nms_iou = params['nms_iou'] / 100.0

            # Run inference
            t0 = time.time()
            raw_dets = backend.predict(frame)
            inference_ms = (time.time() - t0) * 1000.0
            last_detections = raw_dets
            need_redraw = False

        # ── Build canvas ─────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), BG_DARK[0], dtype=np.uint8)

        # Main frame panel with detection overlays
        disp = frame.copy()
        sat_count, plane_count = 0, 0
        for det in last_detections:
            ttype = cls_to_type.get(det['class_id'])
            if ttype is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in det['bbox']]
            color = COLOR_SATELLITE if ttype == 'satellite' else COLOR_AIRPLANE
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            label = f"{ttype[:3].upper()} {det['confidence']:.0%}"
            cv2.putText(disp, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            if ttype == 'satellite':
                sat_count += 1
            else:
                plane_count += 1

        # Scale frame to fit panel
        fh, fw = disp.shape[:2]
        scale = min(FRAME_W / fw, FRAME_H / fh)
        disp_w, disp_h = int(fw * scale), int(fh * scale)
        if disp_w > 0 and disp_h > 0:
            disp_resized = cv2.resize(disp, (disp_w, disp_h))
            ox = (FRAME_W - disp_w) // 2
            oy = (FRAME_H - disp_h) // 2
            canvas[oy:oy + disp_h, ox:ox + disp_w] = disp_resized

        # Panel tag (with bbox tracking for right-click doc access)
        nn_tag_text = f"NN DETECTIONS  [{sat_count}S {plane_count}A]"
        _draw_tag(canvas, nn_tag_text, 8, 18, BG_DARK, ACCENT)
        doc_state['tag_regions'] = [(_doc_tag_bbox(nn_tag_text, 8, 18), 0)]

        # ── Sidebar ──────────────────────────────────────────────────
        sb_x = FRAME_W
        _fill_rect(canvas, sb_x, 0, SIDEBAR_W, WIN_H - STATUS_H, BG_SIDEBAR)
        cv2.line(canvas, (sb_x, 0), (sb_x, WIN_H - STATUS_H), BORDER, 1)

        # Title (with hover hint for doc overlay)
        _title_x, _title_y = sb_x + 12, 18
        _title_bbox = _doc_compute_title_bbox("MNEMOSKY", _title_x, _title_y, 0.45)
        doc_state['title_bbox'] = _title_bbox
        if doc_state['title_hovered']:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, (230, 255, 140), 0.45, 1)
            _put_text(canvas, "(?)", _title_x + _title_bbox[2] + 4, _title_y, ACCENT_DIM, 0.32)
        else:
            _put_text(canvas, "MNEMOSKY", _title_x, _title_y, ACCENT, 0.45, 1)

        sy = 36
        _put_text(canvas, "MODEL INFO", sb_x + 12, sy, TEXT_HEADING, 0.45, 1)
        sy += 22

        # Model path (truncated)
        mpath = Path(nn_params['model_path']).name
        _put_text(canvas, f"  {mpath}", sb_x + 8, sy, TEXT_PRIMARY, 0.36)
        sy += 18
        _put_text(canvas, f"  Backend: {nn_params['backend']}", sb_x + 8, sy, TEXT_DIM, 0.34)
        sy += 16
        _put_text(canvas, f"  Device: {model_info.get('device', '?')}", sb_x + 8, sy, TEXT_DIM, 0.34)
        sy += 16
        _put_text(canvas, f"  Input: {nn_params.get('input_size', 640)}px", sb_x + 8, sy, TEXT_DIM, 0.34)
        sy += 16
        num_classes = len(model_info.get('class_names', {}))
        _put_text(canvas, f"  Classes: {num_classes}", sb_x + 8, sy, TEXT_DIM, 0.34)
        sy += 24

        # Draw sliders
        slider_margin = 14
        slider_track_w = SIDEBAR_W - 2 * slider_margin - 24
        slider_rects = {}

        for skey, slabel, smin, smax, sstep in slider_defs:
            # Label + value
            real_val = params[skey] / 100.0
            _put_text(canvas, f"{slabel}: {real_val:.2f}", sb_x + slider_margin, sy,
                      ACCENT, 0.38, 1)
            sy += 18

            # Track
            tx = sb_x + slider_margin + 12
            track_y = sy
            _fill_rect(canvas, tx, track_y, slider_track_w, 8, SLIDER_TRACK)

            # Fill
            frac = (params[skey] - smin) / max(1, smax - smin)
            fill_w = int(frac * slider_track_w)
            _fill_rect(canvas, tx, track_y, fill_w, 8, SLIDER_FILL)

            # Thumb
            thumb_x = tx + fill_w
            cv2.circle(canvas, (int(thumb_x), int(track_y + 4)), 7,
                       SLIDER_THUMB, -1)

            slider_rects[skey] = (tx, track_y - 8, slider_track_w, 24,
                                  smin, smax, sstep)
            sy += 28

        # Class mapping
        sy += 8
        _put_text(canvas, "CLASS MAP", sb_x + 12, sy, TEXT_HEADING, 0.45, 1)
        sy += 20
        for ttype, cids in class_map.items():
            color = COLOR_SATELLITE if ttype == 'satellite' else COLOR_AIRPLANE
            _put_text(canvas, f"  {cids} -> {ttype}", sb_x + 8, sy, color, 0.34)
            sy += 16

        # Confidence histogram of current detections
        sy += 16
        _put_text(canvas, "DETECTIONS", sb_x + 12, sy, TEXT_HEADING, 0.45, 1)
        sy += 20

        conf_threshold = params['confidence'] / 100.0
        for det in last_detections[:8]:  # max 8 shown
            ttype = cls_to_type.get(det['class_id'], '?')
            cname = det.get('class_name', '')
            label = cname or ttype[:3]
            conf = det['confidence']
            bar_w = int((conf) * (SIDEBAR_W - 60))
            color = COLOR_SATELLITE if ttype == 'satellite' else COLOR_AIRPLANE
            _fill_rect(canvas, sb_x + 45, sy - 8, bar_w, 12, color)
            _put_text(canvas, f"{label}", sb_x + 8, sy, TEXT_PRIMARY, 0.32)
            _put_text(canvas, f"{conf:.0%}", sb_x + 45 + bar_w + 4, sy,
                      TEXT_DIM, 0.30)
            sy += 16

        # Stats
        sy = WIN_H - STATUS_H - 60
        _put_text(canvas, f"Inference: {inference_ms:.0f}ms "
                  f"({1000.0 / max(1, inference_ms):.0f} FPS)",
                  sb_x + 12, sy, ACCENT, 0.36)
        sy += 18
        _put_text(canvas, f"Detections: {len(last_detections)}", sb_x + 12, sy,
                  TEXT_PRIMARY, 0.36)
        sy += 18
        _put_text(canvas, "SPACE/ENTER=Accept  ESC=Cancel  R=Reset",
                  sb_x + 8, sy, TEXT_DIM, 0.28)

        # ── Status bar with frame slider ─────────────────────────────
        bar_y = WIN_H - STATUS_H
        _fill_rect(canvas, 0, bar_y, WIN_W, STATUS_H, BG_PANEL)
        cv2.line(canvas, (0, bar_y), (WIN_W, bar_y), BORDER, 1)

        # Frame slider
        fs_x, fs_y = 60, bar_y + 28
        fs_w = WIN_W - 120
        _fill_rect(canvas, fs_x, fs_y - 3, fs_w, 6, SLIDER_TRACK)
        if total_frames > 1:
            frac = current_frame_idx / max(1, total_frames - 1)
            fill_w = int(frac * fs_w)
            _fill_rect(canvas, fs_x, fs_y - 3, fill_w, 6, SLIDER_FILL)
            cv2.circle(canvas, (fs_x + fill_w, fs_y), 6, SLIDER_THUMB, -1)

        _put_text(canvas, f"Frame {current_frame_idx}/{total_frames}",
                  fs_x, bar_y + 14, TEXT_PRIMARY, 0.35)

        # ── Handle slider interaction ────────────────────────────────
        if mouse_state['down']:
            mx, my = mouse_state['x'], mouse_state['y']

            # Check parameter sliders
            for skey, (rx, ry, rw, rh, smin, smax, sstep) in slider_rects.items():
                if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                    frac = max(0.0, min(1.0, (mx - rx) / max(1, rw)))
                    raw = smin + frac * (smax - smin)
                    snapped = round(raw / sstep) * sstep
                    snapped = max(smin, min(smax, snapped))
                    if params[skey] != int(snapped):
                        params[skey] = int(snapped)
                        need_redraw = True

            # Check frame slider
            if fs_x <= mx <= fs_x + fs_w and bar_y + 10 <= my <= bar_y + 46:
                frac = max(0.0, min(1.0, (mx - fs_x) / max(1, fs_w)))
                new_idx = int(frac * max(1, total_frames - 1))
                if new_idx != current_frame_idx:
                    current_frame_idx = new_idx
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                    ret, new_frame = cap.read()
                    if ret:
                        frame = new_frame
                        need_redraw = True

        # "UNLOCKED" flash on status bar
        if doc_state['flash_timer'] and time.time() - doc_state['flash_timer'] < 2.0:
            _put_text(canvas, "UNLOCKED", WIN_W // 2 - 40, bar_y + 14, ACCENT, 0.40, 1)

        # ── Documentation overlay ─────────────────────────────────────
        if doc_state['visible']:
            doc_state['regions'] = _draw_doc_overlay(
                canvas, doc_state['page'], doc_pages,
                doc_state['konami_unlocked'])

        # ── Display ──────────────────────────────────────────────────
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        # ── Secret word tracking (always active) ─────────────────────
        if key != 255 and 97 <= key <= 122:
            doc_state['key_buffer'].append(chr(key))
            doc_state['key_buffer'] = doc_state['key_buffer'][-len(_DOC_SECRET_WORD):]
            if ''.join(doc_state['key_buffer']) == _DOC_SECRET_WORD:
                doc_state['konami_unlocked'] = True
                doc_state['visible'] = True
                vis = [p for p in doc_pages if not p.get('konami_only') or True]
                doc_state['page'] = len(vis) - 1
                doc_state['flash_timer'] = time.time()
                continue

        # ── Doc overlay key handling ──────────────────────────────────
        if doc_state['visible']:
            if key == 27:
                doc_state['visible'] = False
            elif key in (ord('a'), ord('A')):
                if doc_state['page'] > 0:
                    doc_state['page'] -= 1
            elif key in (ord('d'), ord('D')):
                vis = [p for p in doc_pages if not p.get('konami_only') or doc_state['konami_unlocked']]
                if doc_state['page'] < len(vis) - 1:
                    doc_state['page'] += 1
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                return None
            continue

        if key == 63:  # ? (Shift+/)
            doc_state['visible'] = True
            doc_state['page'] = 0
            continue

        if key in (32, 13):  # SPACE or ENTER — accept
            cv2.destroyWindow(window_name)
            cap.release()
            return {
                'confidence': params['confidence'] / 100.0,
                'nms_iou': params['nms_iou'] / 100.0,
            }
        elif key == 27:  # ESC — cancel
            cv2.destroyWindow(window_name)
            cap.release()
            return None
        elif key in (ord('r'), ord('R')):  # Reset
            params = dict(defaults)
            need_redraw = True
        elif key in (ord('n'), ord('N')):  # Next 1s
            jump = max(1, int(fps))
            current_frame_idx = min(total_frames - 1, current_frame_idx + jump)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                need_redraw = True
        elif key in (ord('p'), ord('P')):  # Prev 1s
            jump = max(1, int(fps))
            current_frame_idx = max(0, current_frame_idx - jump)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                need_redraw = True

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            return None

    cap.release()
    return None


class TemporalFrameBuffer:
    """Rolling buffer of grayscale frames for temporal background estimation.

    With long-exposure astrophotography (e.g. 13 s per frame), persistent
    objects (stars, sky glow, vignetting, hot pixels) appear in every frame
    while transient events (satellite/airplane trails, meteors, cosmic rays)
    appear in only one or two.  The per-pixel *temporal median* of N
    surrounding frames is a near-perfect background model that removes all
    persistent structure, leaving only transients in the difference image.

    This is the same principle used by professional sky surveys (ZTF, LSST)
    for transient detection, adapted here for video.

    Usage::

        buf = TemporalFrameBuffer(capacity=7)
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            buf.add(gray)
            if buf.is_ready():
                ctx = buf.get_temporal_context(gray)
                # ctx['diff_image'], ctx['noise_map'], ctx['reference']
    """

    def __init__(self, capacity=7):
        """
        Args:
            capacity: Number of frames to keep in the buffer.  Should be odd
                so the current frame sits at the centre.  Larger values give
                cleaner backgrounds but use more RAM (~4 MB per 1080p frame).
                7 frames ≈ 28 MB for 1080p — a reasonable default.
        """
        self.capacity = capacity
        self._frames = []           # List of uint8 grayscale arrays
        self._reference = None      # Cached temporal median (float32)
        self._noise_map = None      # Cached per-pixel MAD noise (float32)
        self._dirty = True          # True if buffer changed since last compute
        # Pre-allocated arrays to avoid repeated ~58MB allocations per frame
        self._stack = None          # float32 (capacity, H, W) — reused
        self._abs_dev = None        # float32 (capacity, H, W) — reused

    def add(self, gray_frame):
        """Add a grayscale frame to the buffer, evicting the oldest if full."""
        self._frames.append(gray_frame)
        if len(self._frames) > self.capacity:
            self._frames.pop(0)
        self._dirty = True

    def is_ready(self):
        """True once the buffer has at least 5 frames for a meaningful median."""
        return len(self._frames) >= 5

    @property
    def count(self):
        return len(self._frames)

    def _compute(self):
        """Compute temporal median and noise map from the current buffer.

        Uses pre-allocated arrays to avoid ~116MB of allocations per frame.
        """
        if not self._dirty:
            return

        n = len(self._frames)
        h, w = self._frames[0].shape[:2]

        # Allocate or resize the pre-allocated stack (only on first call or shape change)
        if (self._stack is None or self._stack.shape[0] < n or
                self._stack.shape[1] != h or self._stack.shape[2] != w):
            self._stack = np.empty((self.capacity, h, w), dtype=np.float32)
            self._abs_dev = np.empty((self.capacity, h, w), dtype=np.float32)

        # Fill stack from current frames (only the active slice)
        for i, frame in enumerate(self._frames):
            np.copyto(self._stack[i], frame, casting='unsafe')

        active = self._stack[:n]

        # Per-pixel temporal median — stars and fixed pattern noise vanish
        self._reference = np.median(active, axis=0)

        # Per-pixel MAD (Median Absolute Deviation) → robust noise estimate
        np.subtract(active, self._reference[np.newaxis, :, :], out=self._abs_dev[:n])
        np.abs(self._abs_dev[:n], out=self._abs_dev[:n])
        mad = np.median(self._abs_dev[:n], axis=0)
        self._noise_map = mad  # reuse the array from median
        self._noise_map *= 1.4826  # Gaussian σ equivalent
        # Floor at 0.5 to avoid division by zero in SNR calculations
        np.maximum(self._noise_map, 0.5, out=self._noise_map)

        self._dirty = False

    def get_temporal_context(self, current_gray):
        """Build a temporal context dict for the current frame.

        Args:
            current_gray: The grayscale frame to subtract from the reference.

        Returns:
            Dict with keys:
                'diff_image': float32 background-subtracted image (≥ 0).
                    Stars, sky gradients, vignetting removed.  Only transient
                    features (trails, cosmic rays) remain.
                'noise_map':  float32 per-pixel noise σ (spatially varying).
                    Brighter sky regions have higher noise; dark zenith lower.
                'reference':  float32 temporal median reference frame.
                'buffer_depth': int, how many frames contributed.
        """
        self._compute()

        diff = current_gray.astype(np.float32) - self._reference
        diff = np.clip(diff, 0, None)

        return {
            'diff_image': diff,
            'noise_map': self._noise_map,
            'reference': self._reference,
            'buffer_depth': len(self._frames),
        }


class DetectionTracker:
    """Temporal consistency filter and tracklet builder for satellite detections.

    Maintains a sliding window of per-frame detections and applies two
    post-detection filters:

    1. **Temporal consistency**: A detection is confirmed only if a
       geometrically consistent detection (similar angle + close position)
       appears in >= `min_hits` of the last `window` frames.  Single-frame
       noise transients (cosmic rays, hot pixels, cloud edges) are suppressed.

    2. **Tracklet formation**: Detections linked across 3+ consecutive
       frames with consistent motion form a *tracklet*.  Tracklet-confirmed
       trails get a confidence multiplier and a ``tracklet_id`` tag.

    The tracker runs in the main process after detection results arrive
    (works with both sequential and parallel modes).

    Usage::

        tracker = DetectionTracker(window=4, min_hits=2)
        for frame_idx, detections in detection_stream:
            confirmed = tracker.update(frame_idx, detections)
            # use confirmed detections for drawing / export
    """

    def __init__(self, window=4, min_hits=2, angle_thresh=10, dist_thresh=30):
        """
        Args:
            window: How many recent frames to keep for consistency check.
            min_hits: Minimum number of frames (within the window) a detection
                must appear in to be confirmed.
            angle_thresh: Max angle difference (degrees) to consider two
                detections as the same trail across frames.
            dist_thresh: Max perpendicular distance (pixels) between trail
                midpoints across frames to consider them the same trail.
        """
        self.window = window
        self.min_hits = min_hits
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

        # Ring buffer: list of (frame_idx, detections) tuples
        self._history = []

        # Active tracklets: list of dicts with keys:
        #   'id', 'detections' (list of (frame_idx, trail_type, info)),
        #   'last_frame', 'angle', 'center'
        self._tracklets = []
        self._next_tracklet_id = 1

    @staticmethod
    def _trail_angle(info):
        """Extract angle in [0, 180) from detection info."""
        if 'angle' in info:
            return info['angle'] % 180
        line = info.get('line')
        if line is not None:
            import math
            x1, y1, x2, y2 = line[:4] if len(line) >= 4 else line[0]
            return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        return 0.0

    @staticmethod
    def _trail_center(info):
        """Extract trail midpoint from detection info."""
        if 'center' in info:
            return info['center']
        b = info['bbox']
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def _is_same_trail(self, info_a, info_b):
        """Check if two detections (possibly from different frames) describe
        the same physical trail using angle + perpendicular distance."""
        a1 = self._trail_angle(info_a)
        a2 = self._trail_angle(info_b)
        angle_diff = abs(a1 - a2) % 180
        angle_diff = min(angle_diff, 180 - angle_diff)
        if angle_diff > self.angle_thresh:
            return False

        c1 = self._trail_center(info_a)
        c2 = self._trail_center(info_b)

        # Perpendicular distance from c2 to the line through c1 at angle a1
        import math
        a_rad = math.radians(a1)
        nx, ny = -math.sin(a_rad), math.cos(a_rad)  # perpendicular
        perp_dist = abs((c2[0] - c1[0]) * nx + (c2[1] - c1[1]) * ny)
        return perp_dist < self.dist_thresh

    def _update_tracklets(self, frame_idx, detections):
        """Link current detections to existing tracklets or start new ones."""
        matched = set()  # indices into detections that matched a tracklet

        for tracklet in self._tracklets:
            best_match = None
            best_dist = float('inf')
            for i, (trail_type, info) in enumerate(detections):
                if i in matched:
                    continue
                if self._is_same_trail(tracklet['detections'][-1][2], info):
                    c = self._trail_center(info)
                    tc = tracklet['center']
                    dist = (c[0] - tc[0]) ** 2 + (c[1] - tc[1]) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_match = i

            if best_match is not None:
                trail_type, info = detections[best_match]
                tracklet['detections'].append((frame_idx, trail_type, info))
                tracklet['last_frame'] = frame_idx
                tracklet['center'] = self._trail_center(info)
                tracklet['angle'] = self._trail_angle(info)
                matched.add(best_match)

        # Start new tracklets for unmatched detections
        for i, (trail_type, info) in enumerate(detections):
            if i not in matched:
                self._tracklets.append({
                    'id': self._next_tracklet_id,
                    'detections': [(frame_idx, trail_type, info)],
                    'last_frame': frame_idx,
                    'center': self._trail_center(info),
                    'angle': self._trail_angle(info),
                })
                self._next_tracklet_id += 1

        # Expire tracklets not seen for > window frames
        self._tracklets = [
            t for t in self._tracklets
            if frame_idx - t['last_frame'] <= self.window
        ]

    def update(self, frame_idx, detections):
        """Add detections for the current frame and return confirmed detections.

        Args:
            frame_idx: Current frame index (1-based from process_video).
            detections: List of (trail_type, detection_info) from detect_trails().

        Returns:
            List of (trail_type, detection_info) — only detections that have
            temporal support (appeared in >= min_hits recent frames) or belong
            to tracklets with 3+ frames.  Each confirmed detection_info gets
            an extra ``'temporal_hits'`` key and optionally ``'tracklet_id'``.
        """
        # Store in history ring buffer
        self._history.append((frame_idx, detections))
        if len(self._history) > self.window:
            self._history.pop(0)

        # Update tracklets
        self._update_tracklets(frame_idx, detections)

        # Not enough history — pass everything through
        if len(self._history) < 2:
            return detections

        # Score each current detection by temporal support
        confirmed = []
        for trail_type, info in detections:
            hits = 1  # current frame
            for past_fidx, past_dets in self._history[:-1]:
                for _, past_info in past_dets:
                    if self._is_same_trail(info, past_info):
                        hits += 1
                        break

            # Find tracklet for this detection (if any)
            tracklet_id = None
            tracklet_len = 0
            for t in self._tracklets:
                if (t['last_frame'] == frame_idx and
                        t['detections'][-1][2] is info):
                    tracklet_id = t['id']
                    tracklet_len = len(t['detections'])
                    break

            # Confirm if: enough temporal hits OR strong tracklet
            if hits >= self.min_hits or tracklet_len >= 3:
                enriched_info = dict(info)
                enriched_info['temporal_hits'] = hits
                if tracklet_id is not None:
                    enriched_info['tracklet_id'] = tracklet_id
                    enriched_info['tracklet_length'] = tracklet_len
                confirmed.append((trail_type, enriched_info))

            elif trail_type == 'airplane':
                # Airplanes are bright and distinctive — don't require
                # temporal confirmation (they'd pass on the first frame anyway)
                confirmed.append((trail_type, info))

        return confirmed


# ═══════════════════════════════════════════════════════════════════════
#  STS-Inspired Components: Translation Ledger, Loss Profiles
# ═══════════════════════════════════════════════════════════════════════

class TranslationLedger:
    """Accounting for what the detector rejected and why (Callon).

    Every transformation from raw pixels to classified detection *loses*
    something.  The translation ledger tracks aggregate rejection statistics
    so users can see what the detector discarded, making the filtering
    assumptions transparent and auditable.

    Inspired by Michel Callon's sociology of translation — each stage of
    the detection pipeline is a translation that recruits some phenomena
    and excludes others.  The ledger makes these exclusions visible.
    """

    def __init__(self):
        self.total_lines_detected = 0
        self.rejected_too_short = 0
        self.rejected_too_long = 0
        self.rejected_full_frame = 0
        self.rejected_low_contrast = 0
        self.rejected_too_dark = 0
        self.rejected_too_few_pixels = 0
        self.rejected_cloud_texture = 0
        self.rejected_too_bright = 0
        self.rejected_segment_variation = 0
        self.rejected_aspect_ratio = 0
        self.rejected_unclassifiable = 0
        self.classified_satellite = 0
        self.classified_airplane = 0
        self.classified_anomalous = 0
        self.supplementary_lines = 0
        self.example_rejections = []   # up to 5 examples with reasons
        self._max_examples = 5

    def record_rejection(self, reason, line=None):
        """Increment a rejection counter and optionally store an example."""
        attr = f'rejected_{reason}'
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)
        if line is not None and len(self.example_rejections) < self._max_examples:
            self.example_rejections.append({
                'reason': reason, 'line': tuple(int(v) for v in line[0])})

    def record_classification(self, trail_type):
        """Increment the counter for a successful classification."""
        attr = f'classified_{trail_type}'
        if hasattr(self, attr):
            setattr(self, attr, getattr(self, attr) + 1)

    @property
    def total_rejected(self):
        return sum(getattr(self, a) for a in dir(self) if a.startswith('rejected_'))

    @property
    def total_classified(self):
        return (self.classified_satellite + self.classified_airplane +
                self.classified_anomalous)

    def summary_lines(self):
        """Return a list of formatted summary strings for display."""
        total = self.total_lines_detected + self.supplementary_lines
        if total == 0:
            return ["Translation Ledger: no lines detected"]
        lines = []
        lines.append("=== Translation Ledger (Callon) ===")
        lines.append(f"Primary lines detected (Hough):   {self.total_lines_detected:>6}")
        if self.supplementary_lines:
            lines.append(f"Supplementary lines (matched flt): {self.supplementary_lines:>5}")
        lines.append(f"  Rejected (too short):            {self.rejected_too_short:>5}  "
                      f"({self._pct(self.rejected_too_short, total)})")
        lines.append(f"  Rejected (too long):             {self.rejected_too_long:>5}  "
                      f"({self._pct(self.rejected_too_long, total)})")
        lines.append(f"  Rejected (full-frame artifact):  {self.rejected_full_frame:>5}  "
                      f"({self._pct(self.rejected_full_frame, total)})")
        lines.append(f"  Rejected (low contrast):         {self.rejected_low_contrast:>5}  "
                      f"({self._pct(self.rejected_low_contrast, total)})")
        lines.append(f"  Rejected (too dark):             {self.rejected_too_dark:>5}  "
                      f"({self._pct(self.rejected_too_dark, total)})")
        lines.append(f"  Rejected (cloud/texture):        {self.rejected_cloud_texture:>5}  "
                      f"({self._pct(self.rejected_cloud_texture, total)})")
        lines.append(f"  Rejected (too bright):           {self.rejected_too_bright:>5}  "
                      f"({self._pct(self.rejected_too_bright, total)})")
        lines.append(f"  Rejected (aspect ratio):         {self.rejected_aspect_ratio:>5}  "
                      f"({self._pct(self.rejected_aspect_ratio, total)})")
        lines.append(f"  Rejected (segment variation):    {self.rejected_segment_variation:>5}  "
                      f"({self._pct(self.rejected_segment_variation, total)})")
        lines.append(f"  Rejected (unclassifiable):       {self.rejected_unclassifiable:>5}  "
                      f"({self._pct(self.rejected_unclassifiable, total)})")
        lines.append(f"  {'─' * 43}")
        lines.append(f"  Classified as satellite:         {self.classified_satellite:>5}  "
                      f"({self._pct(self.classified_satellite, total)})")
        lines.append(f"  Classified as airplane:          {self.classified_airplane:>5}  "
                      f"({self._pct(self.classified_airplane, total)})")
        if self.classified_anomalous:
            lines.append(f"  Classified as anomalous:         {self.classified_anomalous:>5}  "
                          f"({self._pct(self.classified_anomalous, total)})")
        lines.append(f"  {'─' * 43}")
        survived = self.total_classified
        lines.append(f"  Survival rate: {self._pct(survived, total)} "
                      f"of detected lines → classified trails")
        return lines

    def to_dict(self):
        """Serialize for JSON storage in annotation database."""
        return {k: getattr(self, k) for k in [
            'total_lines_detected', 'supplementary_lines',
            'rejected_too_short', 'rejected_too_long', 'rejected_full_frame',
            'rejected_low_contrast', 'rejected_too_dark',
            'rejected_too_few_pixels', 'rejected_cloud_texture',
            'rejected_too_bright', 'rejected_segment_variation',
            'rejected_aspect_ratio', 'rejected_unclassifiable',
            'classified_satellite', 'classified_airplane',
            'classified_anomalous', 'example_rejections',
        ]}

    @staticmethod
    def _pct(n, total):
        return f"{n / total * 100:.1f}%" if total > 0 else "0.0%"


# ═══════════════════════════════════════════════════════════════════════
#  STS-Inspired: Situated Loss Profiles (Winner / Jasanoff)
# ═══════════════════════════════════════════════════════════════════════
#
#  The loss function weights are *political choices* disguised as
#  engineering defaults.  Making them a named user-facing choice with
#  explicit value statements embodies Langdon Winner's insight that
#  artifacts have politics.

LOSS_PROFILES = {
    'discovery': {
        'weights': {'fp': 0.5, 'fn': 3.0, 'mc': 0.3},
        'description': 'Maximizes recall — prioritizes not missing faint trails',
    },
    'precision': {
        'weights': {'fp': 3.0, 'fn': 0.5, 'mc': 1.0},
        'description': 'Minimizes false alarms — prioritizes confident detections',
    },
    'balanced': {
        'weights': {'fp': 1.0, 'fn': 2.0, 'mc': 0.5},
        'description': 'Balanced — slight preference for recall over precision',
    },
    'catalog': {
        'weights': {'fp': 1.0, 'fn': 1.0, 'mc': 3.0},
        'description': 'Prioritizes correct satellite vs airplane classification',
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  HITL (Human-in-the-Loop) Reinforcement Learning Components
# ═══════════════════════════════════════════════════════════════════════

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

CORRECTION_RULES = {
    ('reject', 'satellite'): [
        {
            'param': 'satellite_contrast_min',
            'diagnostic': lambda meta: meta.get('contrast_ratio'),
            'direction': 'raise',
            'target_fn': lambda diag_val, current: max(current, diag_val + 0.01),
        },
        {
            'param': 'satellite_min_length',
            'diagnostic': lambda meta: meta.get('length'),
            'direction': 'raise',
            'condition': lambda meta, params: meta.get('length', 999) < params.get('satellite_min_length', 60) * 1.5,
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
    ('add_missed', 'airplane'): [
        {
            'param': 'airplane_brightness_min',
            'diagnostic': lambda meta: meta.get('avg_brightness'),
            'direction': 'lower',
            'target_fn': lambda diag_val, current: min(current, max(20, diag_val - 5)),
        },
    ],
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


class AnnotationDatabase:
    """COCO-compatible annotation database with correction tracking.

    Manages detection annotations, human corrections, and session metadata.
    Supports loading/saving to JSON and export to pure COCO format.
    """

    CATEGORY_MAP = {0: 'satellite', 1: 'airplane', 2: 'anomalous'}
    CATEGORY_ID = {'satellite': 0, 'airplane': 1, 'anomalous': 2}

    def __init__(self, path=None):
        """Load existing database or create empty one."""
        self._path = Path(path) if path else None
        self._next_image_id = 1
        self._next_annotation_id = 1
        self._next_missed_id = 1
        self._next_correction_id = 1
        self._current_session_id = None

        self.data = {
            'info': {
                'description': 'Mnemosky HITL annotation database',
                'version': '1.0',
                'date_created': datetime.now(timezone.utc).isoformat(),
            },
            'categories': [
                {'id': 0, 'name': 'satellite', 'supercategory': 'trail'},
                {'id': 1, 'name': 'airplane', 'supercategory': 'trail'},
                {'id': 2, 'name': 'anomalous', 'supercategory': 'trail'},
            ],
            'images': [],
            'annotations': [],
            'missed_annotations': [],
            'corrections': [],
            'sessions': [],
            'learned_parameters': {
                'current': {},
                'update_count': 0,
                'last_updated': None,
                'history': [],
            },
        }

        if path and Path(path).exists():
            self._load(path)

    def _load(self, path):
        """Load database from JSON file, with validation and corruption recovery."""
        try:
            with open(path, 'r') as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Annotation database corrupted ({e}), starting fresh.")
            return
        except OSError as e:
            print(f"Warning: Cannot read annotation database ({e}), starting fresh.")
            return
        # Validate expected structure before merging
        for key in list(self.data.keys()):
            if key in loaded and isinstance(loaded[key], type(self.data[key])):
                self.data[key] = loaded[key]
        # Rebuild ID counters
        if self.data['images']:
            self._next_image_id = max((img.get('id', 0) for img in self.data['images']), default=0) + 1
        if self.data['annotations']:
            self._next_annotation_id = max((a.get('id', 0) for a in self.data['annotations']), default=0) + 1
        if self.data['missed_annotations']:
            self._next_missed_id = max((m.get('id', 0) for m in self.data['missed_annotations']), default=0) + 1
        if self.data['corrections']:
            self._next_correction_id = max((c.get('id', 0) for c in self.data['corrections']), default=0) + 1

    def add_image(self, frame_index, video_source, width, height):
        """Register a frame image, return image_id."""
        # Check if image already exists for this frame
        for img in self.data['images']:
            if img['frame_index'] == frame_index and img['video_source'] == video_source:
                return img['id']

        img_id = self._next_image_id
        self._next_image_id += 1
        stem = Path(video_source).stem
        self.data['images'].append({
            'id': img_id,
            'file_name': f'{stem}_f{frame_index:06d}.jpg',
            'width': width,
            'height': height,
            'frame_index': frame_index,
            'video_source': video_source,
            'session_id': self._current_session_id,
        })
        return img_id

    def add_detection(self, image_id, category_id, bbox_xyxy, detection_info,
                      params_snapshot, confidence):
        """Add a detector-produced annotation, return annotation_id.
        Converts from internal (x_min, y_min, x_max, y_max) to COCO (x, y, w, h)."""
        ann_id = self._next_annotation_id
        self._next_annotation_id += 1
        x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy]
        coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        area = coco_bbox[2] * coco_bbox[3]

        # Build detection metadata from detection_info
        det_meta = {}
        for key in ('angle', 'center', 'length', 'avg_brightness', 'max_brightness',
                     'line', 'contrast_ratio', 'brightness_std', 'trail_snr',
                     'has_dotted_pattern', 'avg_saturation',
                     'epistemic_profile', 'inscription'):
            if key in detection_info:
                val = detection_info[key]
                # Convert numpy types to native Python for JSON
                if isinstance(val, (np.integer,)):
                    val = int(val)
                elif isinstance(val, (np.floating,)):
                    val = float(val)
                elif isinstance(val, tuple):
                    val = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in val]
                det_meta[key] = val

        self.data['annotations'].append({
            'id': ann_id,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': coco_bbox,
            'area': area,
            'iscrowd': 0,
            'mnemosky_ext': {
                'source': 'detector',
                'status': 'pending',
                'confidence': round(confidence, 4),
                'review_action': None,
                'reviewed_at': None,
                'detection_meta': det_meta,
                'original_category_id': category_id,
                'original_bbox': coco_bbox[:],
                'parameters_snapshot': dict(params_snapshot) if params_snapshot else {},
            },
        })
        return ann_id

    def add_missed(self, image_id, category_id, bbox_xyxy, estimated_meta=None):
        """Add a user-marked missed detection, return missed_annotation_id."""
        mid = self._next_missed_id
        self._next_missed_id += 1
        x_min, y_min, x_max, y_max = [int(v) for v in bbox_xyxy]
        coco_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        entry = {
            'id': mid,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': coco_bbox,
            'area': coco_bbox[2] * coco_bbox[3],
            'marked_at': datetime.now(timezone.utc).isoformat(),
            'session_id': self._current_session_id,
            'estimated_meta': estimated_meta or {},
        }
        self.data['missed_annotations'].append(entry)

        # Record as correction
        self.record_correction(None, 'add_missed',
                               new_category_id=category_id,
                               new_bbox=coco_bbox,
                               missed_annotation_id=mid)
        return mid

    def record_correction(self, annotation_id, action,
                          new_category_id=None, new_bbox=None, notes=None,
                          missed_annotation_id=None):
        """Record a human correction action."""
        cid = self._next_correction_id
        self._next_correction_id += 1

        prev_status = None
        prev_cat = None
        new_status = None

        if annotation_id is not None:
            ann = self._get_annotation(annotation_id)
            if ann:
                prev_status = ann['mnemosky_ext']['status']
                prev_cat = ann['category_id']
                if action == 'accept':
                    ann['mnemosky_ext']['status'] = 'confirmed'
                    ann['mnemosky_ext']['review_action'] = 'accepted'
                    new_status = 'confirmed'
                elif action == 'reject':
                    ann['mnemosky_ext']['status'] = 'rejected'
                    ann['mnemosky_ext']['review_action'] = 'rejected'
                    new_status = 'rejected'
                elif action == 'reclassify':
                    ann['mnemosky_ext']['status'] = 'confirmed'
                    ann['mnemosky_ext']['review_action'] = 'reclassified'
                    if new_category_id is not None:
                        ann['category_id'] = new_category_id
                    new_status = 'confirmed'
                elif action == 'adjust_bbox':
                    if new_bbox is not None:
                        ann['bbox'] = new_bbox[:]
                        ann['area'] = new_bbox[2] * new_bbox[3]
                    new_status = ann['mnemosky_ext']['status']
                ann['mnemosky_ext']['reviewed_at'] = datetime.now(timezone.utc).isoformat()
        elif action == 'add_missed':
            new_status = 'confirmed'

        correction = {
            'id': cid,
            'annotation_id': annotation_id,
            'session_id': self._current_session_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'action': action,
            'previous_status': prev_status,
            'new_status': new_status,
            'previous_category_id': prev_cat,
            'new_category_id': new_category_id,
            'previous_bbox': None,
            'new_bbox': new_bbox,
            'notes': notes,
        }
        if missed_annotation_id is not None:
            correction['missed_annotation_id'] = missed_annotation_id
        self.data['corrections'].append(correction)
        return cid

    def _get_annotation(self, annotation_id):
        """Get annotation by ID."""
        for ann in self.data['annotations']:
            if ann['id'] == annotation_id:
                return ann
        return None

    def get_calibration_set(self):
        """Build calibration set for the learning loop.
        Returns list of (detection_meta, true_label, detector_label) tuples."""
        cal_set = []

        for ann in self.data['annotations']:
            ext = ann['mnemosky_ext']
            meta = ext.get('detection_meta', {})
            detector_label = self.CATEGORY_MAP.get(ext['original_category_id'])

            if ext['status'] == 'confirmed':
                true_label = self.CATEGORY_MAP.get(ann['category_id'])
                cal_set.append((meta, true_label, detector_label))
            elif ext['status'] == 'rejected':
                cal_set.append((meta, None, detector_label))

        for missed in self.data['missed_annotations']:
            meta = missed.get('estimated_meta', {})
            true_label = self.CATEGORY_MAP.get(missed['category_id'])
            cal_set.append((meta, true_label, None))

        return cal_set

    def get_pending_annotations(self, image_id=None):
        """Get annotations awaiting review, sorted by confidence (low first)."""
        pending = []
        for ann in self.data['annotations']:
            if ann['mnemosky_ext']['status'] != 'pending':
                continue
            if image_id is not None and ann['image_id'] != image_id:
                continue
            pending.append(ann)
        pending.sort(key=lambda a: a['mnemosky_ext'].get('confidence', 0.5))
        return pending

    def get_frames_needing_review(self):
        """Get image_ids sorted by minimum detection confidence."""
        frame_min_conf = {}
        for ann in self.data['annotations']:
            if ann['mnemosky_ext']['status'] != 'pending':
                continue
            img_id = ann['image_id']
            conf = ann['mnemosky_ext'].get('confidence', 0.5)
            if img_id not in frame_min_conf or conf < frame_min_conf[img_id]:
                frame_min_conf[img_id] = conf
        return sorted(frame_min_conf.keys(), key=lambda k: frame_min_conf[k])

    def get_image_by_id(self, image_id):
        """Get image entry by ID."""
        for img in self.data['images']:
            if img['id'] == image_id:
                return img
        return None

    def get_annotations_for_image(self, image_id):
        """Get all annotations for a given image."""
        return [a for a in self.data['annotations'] if a['image_id'] == image_id]

    def get_missed_for_image(self, image_id):
        """Get all missed annotations for a given image."""
        return [m for m in self.data['missed_annotations'] if m['image_id'] == image_id]

    def start_session(self, video_source, sensitivity, algorithm, params,
                      observer_context=None, loss_profile='balanced'):
        """Begin a new review session."""
        ts = datetime.now(timezone.utc)
        self._current_session_id = f"sess_{ts.strftime('%Y%m%d_%H%M%S')}"
        self.data['sessions'].append({
            'id': self._current_session_id,
            'video_source': video_source,
            'started_at': ts.isoformat(),
            'completed_at': None,
            'sensitivity': sensitivity,
            'algorithm': algorithm,
            'software_version': __version__,
            'observer_context': observer_context,
            'loss_profile': loss_profile,
            'frames_reviewed': 0,
            'total_frames': 0,
            'corrections_count': 0,
            'accepted_count': 0,
            'rejected_count': 0,
            'reclassified_count': 0,
            'missed_count': 0,
            'parameters_before': dict(params),
            'parameters_after': None,
        })

    def end_session(self, params_after=None):
        """Finalize current session with post-learning parameters."""
        if not self._current_session_id:
            return
        for sess in self.data['sessions']:
            if sess['id'] == self._current_session_id:
                sess['completed_at'] = datetime.now(timezone.utc).isoformat()
                if params_after:
                    sess['parameters_after'] = dict(params_after)
                # Count corrections for this session
                counts = {'accept': 0, 'reject': 0, 'reclassify': 0, 'add_missed': 0}
                n_sess = 0
                for c in self.data['corrections']:
                    if c['session_id'] == self._current_session_id:
                        n_sess += 1
                        action = c.get('action')
                        if action in counts:
                            counts[action] += 1
                sess['corrections_count'] = n_sess
                sess['accepted_count'] = counts['accept']
                sess['rejected_count'] = counts['reject']
                sess['reclassified_count'] = counts['reclassify']
                sess['missed_count'] = counts['add_missed']
                break

    def save(self, path=None):
        """Write database to JSON file (atomic: write to temp, then rename)."""
        save_path = Path(path) if path else self._path
        if not save_path:
            return
        # Warn if annotation database is getting very large
        n_ann = len(self.data.get('annotations', []))
        n_corr = len(self.data.get('corrections', []))
        if n_ann + n_corr > 50000:
            print(f"Warning: Annotation database has {n_ann} annotations and "
                  f"{n_corr} corrections. Consider exporting and archiving.")
        # Atomic write: dump to temp file in same directory, then os.replace
        fd, tmp_path = tempfile.mkstemp(
            dir=str(save_path.parent), suffix='.tmp', prefix=save_path.stem)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
            os.replace(tmp_path, str(save_path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def export_coco(self, path):
        """Export pure COCO format (confirmed annotations only)."""
        coco = {
            'images': self.data['images'][:],
            'annotations': [],
            'categories': self.data['categories'][:],
        }
        ann_id = 1
        for ann in self.data['annotations']:
            if ann['mnemosky_ext']['status'] == 'confirmed':
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': ann['image_id'],
                    'category_id': ann['category_id'],
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': 0,
                })
                ann_id += 1
        # Include missed annotations as regular annotations
        for missed in self.data['missed_annotations']:
            coco['annotations'].append({
                'id': ann_id,
                'image_id': missed['image_id'],
                'category_id': missed['category_id'],
                'bbox': missed['bbox'],
                'area': missed['area'],
                'iscrowd': 0,
            })
            ann_id += 1
        with open(path, 'w') as f:
            json.dump(coco, f, indent=2)

    def undo_last_correction(self):
        """Undo the most recent correction. Returns True if successful."""
        if not self.data['corrections']:
            return False
        correction = self.data['corrections'].pop()
        ann_id = correction.get('annotation_id')
        if ann_id is not None:
            ann = self._get_annotation(ann_id)
            if ann:
                # Restore previous status
                if correction['previous_status']:
                    ann['mnemosky_ext']['status'] = correction['previous_status']
                if correction['previous_category_id'] is not None:
                    ann['category_id'] = correction['previous_category_id']
                ann['mnemosky_ext']['review_action'] = None
                ann['mnemosky_ext']['reviewed_at'] = None
        if correction['action'] == 'add_missed' and correction.get('missed_annotation_id'):
            # Remove the missed annotation
            self.data['missed_annotations'] = [
                m for m in self.data['missed_annotations']
                if m['id'] != correction['missed_annotation_id']
            ]
        return True


class ParameterAdapter:
    """Adapts detection parameters from human corrections.

    Tier 1: Immediate EMA adaptation per correction.
    Tier 2: Batch coordinate-wise golden section search over calibration set.
    """

    def __init__(self, initial_params, safety_bounds=None, loss_profile='balanced'):
        """Initialize with starting parameters and safety bounds.

        Args:
            initial_params: Starting detection parameter dict.
            safety_bounds: Hard min/max per parameter (prevents drift).
            loss_profile: Named loss profile from LOSS_PROFILES (Winner).
                Controls the FP/FN/misclassification tradeoff during Tier 2 learning.
        """
        self.params = dict(initial_params)
        self.safety_bounds = safety_bounds or PARAMETER_SAFETY_BOUNDS
        self.update_counts = {p: 0 for p in initial_params}
        self.base_lr = 0.3
        self.decay_rate = 0.1
        # STS: Situated loss weights (Winner — artifacts have politics)
        profile = LOSS_PROFILES.get(loss_profile, LOSS_PROFILES['balanced'])
        self.loss_weights = profile['weights']
        self.loss_profile_name = loss_profile

    def apply_correction(self, correction_action, trail_type, detection_meta):
        """Tier 1: Apply single correction via EMA. Returns {param: new_value} updates."""
        key = (correction_action, trail_type)
        rules = CORRECTION_RULES.get(key, [])

        updates = {}
        for rule in rules:
            param = rule['param']
            if param not in self.params:
                continue
            if 'condition' in rule and not rule['condition'](detection_meta, self.params):
                continue
            diag_val = rule['diagnostic'](detection_meta)
            if diag_val is None:
                continue

            target = rule['target_fn'](diag_val, self.params[param])
            n = self.update_counts.get(param, 0)
            alpha = self.base_lr / (1.0 + n * self.decay_rate)

            new_val = self.params[param] + alpha * (target - self.params[param])
            lo, hi = self.safety_bounds.get(param, (float('-inf'), float('inf')))
            new_val = max(lo, min(hi, new_val))

            self.params[param] = new_val
            self.update_counts[param] = n + 1
            updates[param] = new_val

        return updates

    def optimize_batch(self, calibration_set):
        """Tier 2: Run coordinate-wise golden section search.
        Uses the situated loss weights from the active loss profile (Winner).
        Returns optimized parameters dict."""
        if len(calibration_set) < 10:
            return dict(self.params)

        params = dict(self.params)
        best_loss = self._compute_loss(params, calibration_set, weights=self.loss_weights)

        phi = (1 + 5 ** 0.5) / 2
        resphi = 2 - phi

        optimizable = [p for p in [
            'satellite_contrast_min', 'satellite_min_length', 'satellite_max_length',
            'canny_low', 'canny_high', 'hough_threshold', 'min_line_length',
            'max_line_gap', 'brightness_threshold', 'airplane_brightness_min',
            'airplane_saturation_min', 'min_aspect_ratio', 'mf_snr_threshold',
        ] if p in params and p in self.safety_bounds]

        for _ in range(3):
            improved = False
            for param_name in optimizable:
                lo, hi = self.safety_bounds[param_name]
                a, b = lo, hi
                x1 = a + resphi * (b - a)
                x2 = b - resphi * (b - a)

                test = dict(params)
                test[param_name] = x1
                f1 = self._compute_loss(test, calibration_set, weights=self.loss_weights)
                test[param_name] = x2
                f2 = self._compute_loss(test, calibration_set, weights=self.loss_weights)

                for __ in range(15):
                    if f1 < f2:
                        b = x2
                        x2 = x1
                        f2 = f1
                        x1 = a + resphi * (b - a)
                        test[param_name] = x1
                        f1 = self._compute_loss(test, calibration_set, weights=self.loss_weights)
                    else:
                        a = x1
                        x1 = x2
                        f1 = f2
                        x2 = b - resphi * (b - a)
                        test[param_name] = x2
                        f2 = self._compute_loss(test, calibration_set, weights=self.loss_weights)

                optimal = (a + b) / 2
                test[param_name] = optimal
                new_loss = self._compute_loss(test, calibration_set, weights=self.loss_weights)
                if new_loss < best_loss:
                    params[param_name] = optimal
                    best_loss = new_loss
                    improved = True

            if not improved:
                break

        # Reversion safety: reject if >20% worse
        original_loss = self._compute_loss(self.params, calibration_set, weights=self.loss_weights)
        if best_loss > original_loss * 1.2:
            return dict(self.params)

        self.params = params
        return dict(params)

    @staticmethod
    def _simulate_classify(meta, params):
        """Simulate whether a detection would pass classification with given params."""
        contrast = meta.get('contrast_ratio', 1.0)
        length = meta.get('length', 0)
        brightness = meta.get('avg_brightness', 0)
        brightness_std = meta.get('brightness_std', 0)
        has_dotted = meta.get('has_dotted_pattern', False)

        if contrast < params.get('satellite_contrast_min', 1.08):
            return False, None
        if length < params.get('satellite_min_length', 60):
            return False, None
        if length > params.get('satellite_max_length', 1400):
            return False, None

        is_bright = brightness > params.get('airplane_brightness_min', 45)
        if has_dotted and is_bright:
            return True, 'airplane'

        is_dim = brightness < params.get('airplane_brightness_min', 45)
        brightness_var = brightness_std / (brightness + 1e-5)
        is_smooth = brightness_var < 0.40
        if is_dim and is_smooth:
            return True, 'satellite'

        return False, None

    @staticmethod
    def _compute_loss(params, calibration_set, weights=None):
        """Evaluate a parameter vector against the calibration set."""
        if weights is None:
            weights = {'fp': 1.0, 'fn': 2.0, 'mc': 0.5}
        fp_count = fn_count = mc_count = 0

        for meta, true_label, detector_label in calibration_set:
            would_detect, predicted_label = ParameterAdapter._simulate_classify(meta, params)
            if true_label is None:
                if would_detect:
                    fp_count += 1
            elif detector_label is None:
                if not would_detect:
                    fn_count += 1
            else:
                if not would_detect:
                    fn_count += 1
                elif predicted_label != true_label:
                    mc_count += 1

        return (weights['fp'] * fp_count +
                weights['fn'] * fn_count +
                weights['mc'] * mc_count)

    def get_params(self):
        """Get current parameter values."""
        return dict(self.params)

    def save_profile(self, profile_name='default'):
        """Save learned parameters to ~/.mnemosky/learned_params.json."""
        try:
            profile_dir = Path.home() / '.mnemosky'
            profile_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot create profile directory ({e}), parameters not saved.")
            return
        profile_path = profile_dir / 'learned_params.json'

        profiles = {'version': 1, 'profiles': {}, 'active_profile': profile_name}
        if profile_path.exists():
            try:
                with open(profile_path, 'r') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profiles = loaded
            except (json.JSONDecodeError, OSError):
                pass

        profiles['profiles'][profile_name] = {
            'parameters': dict(self.params),
            'calibration_stats': {
                'total_corrections': sum(self.update_counts.values()),
            },
            'last_updated': datetime.now(timezone.utc).isoformat(),
        }
        profiles['active_profile'] = profile_name

        with open(profile_path, 'w') as f:
            json.dump(profiles, f, indent=2)

    def load_profile(self, profile_name='default'):
        """Load learned parameters from profile. Returns True if found."""
        profile_path = Path.home() / '.mnemosky' / 'learned_params.json'
        if not profile_path.exists():
            return False
        try:
            with open(profile_path, 'r') as f:
                profiles = json.load(f)
            if not isinstance(profiles, dict):
                return False
            if profile_name in profiles.get('profiles', {}):
                learned = profiles['profiles'][profile_name].get('parameters', {})
                for k, v in learned.items():
                    if k in self.params:
                        self.params[k] = v
                return True
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
        return False

    @staticmethod
    def compute_confidence(detection_info, params):
        """Compute pseudo-confidence score for a detection."""
        contrast = detection_info.get('contrast_ratio', 1.0)
        snr = detection_info.get('trail_snr', 0.0) or 0.0
        length = detection_info.get('length', 0.0)
        brightness = detection_info.get('avg_brightness', 0.0)
        brightness_std = detection_info.get('brightness_std', 0.0)

        contrast_min = params.get('satellite_contrast_min', 1.08)
        contrast_margin = (contrast - contrast_min) / max(contrast_min, 0.01)
        snr_margin = (snr - 2.5) / 2.5 if snr > 0 else 0

        min_len = params.get('satellite_min_length', 60)
        max_len = params.get('satellite_max_length', 1400)
        len_mid = (min_len + max_len) / 2
        len_range = max_len - min_len
        length_score = 1.0 - min(1.0, 2 * abs(length - len_mid) / max(len_range, 1)) if len_range > 0 else 0.5

        smoothness = 1.0 - min(1.0, brightness_std / (brightness + 1e-5) / 0.4)

        raw = (0.30 * max(0, min(1, contrast_margin * 5))
             + 0.25 * max(0, min(1, snr_margin * 2))
             + 0.20 * max(0, length_score)
             + 0.25 * max(0, smoothness))

        return 1.0 / (1.0 + np.exp(-6 * (raw - 0.5)))


class ReviewUI:
    """Interactive review interface for HITL correction workflow.

    Single OpenCV window with dark theme matching the existing preview GUI.
    Displays detections for review, handles keyboard/mouse interaction,
    and feeds corrections to AnnotationDatabase and ParameterAdapter.
    """

    # Theme colours (BGR)
    BG_COLOR = (30, 30, 30)
    PANEL_COLOR = (42, 42, 42)
    TEXT_COLOR = (210, 210, 210)
    DIM_TEXT = (120, 120, 120)
    ACCENT = (200, 255, 80)          # Fluorescent green-yellow
    CONFIRMED_COLOR = (80, 200, 80)  # Green
    REJECTED_COLOR = (80, 80, 180)   # Dim red
    MISSED_COLOR = (255, 100, 200)   # Magenta
    AMBER = (0, 200, 255)            # Warning amber

    SIDEBAR_W = 280
    STATUS_H = 56
    WINDOW_NAME = 'Mnemosky Review'

    def __init__(self, video_path, detections_by_frame, detector,
                 annotation_db, param_adapter=None, auto_accept_threshold=0.9):
        self.video_path = video_path
        self.detections_by_frame = detections_by_frame or {}
        self.detector = detector
        self.ann_db = annotation_db
        self.param_adapter = param_adapter
        self.auto_accept_threshold = auto_accept_threshold

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video for review: {video_path}")
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.current_frame_idx = -1
        self.current_frame = None
        self.selected_ann_idx = 0
        self.current_annotations = []
        self.current_missed = []

        self.mark_mode = False
        self.mark_start = None
        self.mark_end = None
        self.mark_pending_bbox = None    # (x_min, y_min, x_max, y_max) awaiting category
        self.mark_pending_meta = None    # estimated metadata awaiting category
        self.mark_pending_img_id = None  # image ID awaiting category

        # Frame index list ordered by review priority
        self.review_queue = []
        self._build_review_queue()

        # Session stats
        self.stats = {'reviewed': 0, 'accepted': 0, 'rejected': 0,
                      'reclassified': 0, 'missed': 0,
                      'param_adjustments': 0, 'learn_runs': 0,
                      'params_optimized': 0}

        # Toast notification queue: [(text, color, expire_tick)]
        self._toasts = []
        self._tick = 0

        # Total pending count for progress tracking
        self._total_detections = sum(len(dets) for dets in self.detections_by_frame.values())
        self._frames_with_detections = len(self.detections_by_frame)

        # Frame loading state
        self._loading_frame = False

        self.running = True
        self.help_visible = False
        self.full_frame_mode = False

    def _toast(self, text, color=None, duration=60):
        """Show a transient notification overlay (duration in ticks, ~30fps)."""
        if color is None:
            color = self.ACCENT
        self._toasts.append((text, color, self._tick + duration))

    def _count_reviewed_frames(self):
        """Count frames where all detections have been reviewed."""
        reviewed = 0
        for im in self.ann_db.data['images']:
            pending = self.ann_db.get_pending_annotations(im['id'])
            if not pending:
                reviewed += 1
        return reviewed

    def _build_review_queue(self):
        """Build frame review queue sorted by minimum confidence."""
        frames_with_detections = sorted(self.detections_by_frame.keys())
        if not frames_with_detections:
            self.review_queue = []
            return

        # Build confidence map from annotation DB
        frame_confs = {}
        for img in self.ann_db.data['images']:
            fi = img['frame_index']
            pending = self.ann_db.get_pending_annotations(img['id'])
            if pending:
                confs = [a.get('mnemosky_ext', {}).get('confidence', 0.5) for a in pending]
                if confs:
                    frame_confs[fi] = min(confs)

        # Sort by confidence (lowest first), then by frame index
        self.review_queue = sorted(
            frames_with_detections,
            key=lambda fi: (frame_confs.get(fi, 1.0), fi))

    def run(self):
        """Main event loop."""
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        display_w = self.vid_w + (0 if self.full_frame_mode else self.SIDEBAR_W)
        display_h = self.vid_h + self.STATUS_H
        cv2.resizeWindow(self.WINDOW_NAME, min(display_w, 1920), min(display_h, 1136))
        cv2.setMouseCallback(self.WINDOW_NAME, self._handle_mouse)

        # Navigate to first frame with detections
        if self.review_queue:
            self._navigate_to_frame(self.review_queue[0])
        elif self.total_frames > 0:
            self._navigate_to_frame(0)

        while self.running:
            self._tick += 1
            # Expire old toasts
            self._toasts = [(t, c, e) for t, c, e in self._toasts if e > self._tick]
            canvas = self._render()
            cv2.imshow(self.WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key != 255:
                self._handle_key(key)
            # Detect window closed via X button
            try:
                if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.running = False
            except cv2.error:
                self.running = False

        self.cap.release()
        cv2.destroyWindow(self.WINDOW_NAME)

    def _navigate_to_frame(self, frame_idx):
        """Seek video to frame and load annotations."""
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        if frame_idx == self.current_frame_idx and self.current_frame is not None:
            return

        # Show loading indicator
        self._loading_frame = True
        if self.current_frame is not None:
            canvas = self._render()
            cv2.imshow(self.WINDOW_NAME, canvas)
            cv2.waitKey(1)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        self._loading_frame = False
        if not ret:
            self._toast("Failed to read frame", self.REJECTED_COLOR)
            return
        self.current_frame_idx = frame_idx
        self.current_frame = frame
        self.selected_ann_idx = 0

        # Load annotations for this frame
        img = None
        for im in self.ann_db.data['images']:
            if im['frame_index'] == frame_idx:
                img = im
                break
        if img:
            self.current_annotations = self.ann_db.get_annotations_for_image(img['id'])
            self.current_missed = self.ann_db.get_missed_for_image(img['id'])
        else:
            self.current_annotations = []
            self.current_missed = []

        # Sort by confidence (lowest first for review priority)
        self.current_annotations.sort(
            key=lambda a: a['mnemosky_ext'].get('confidence', 0.5))

    def _render(self):
        """Render current frame with overlays, sidebar, and status bar."""
        sidebar_w = 0 if self.full_frame_mode else self.SIDEBAR_W
        canvas_w = self.vid_w + sidebar_w
        canvas_h = self.vid_h + self.STATUS_H
        canvas = np.full((canvas_h, canvas_w, 3), self.BG_COLOR, dtype=np.uint8)

        if self.current_frame is not None:
            frame_display = self.current_frame.copy()
            self._draw_overlays(frame_display)
            canvas[0:self.vid_h, 0:self.vid_w] = frame_display

        if not self.full_frame_mode:
            self._render_sidebar(canvas)

        self._render_status_bar(canvas)

        if self.help_visible:
            self._render_help(canvas)

        # Draw toast notifications (bottom-left, stacked upward)
        if self._toasts:
            toast_y = self.vid_h - 15
            for text, color, expire in reversed(self._toasts):
                # Fade out in last 15 ticks
                remaining = expire - self._tick
                alpha = min(1.0, remaining / 15.0)
                faded = tuple(int(c * alpha) for c in color)
                # Background pill
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(canvas, (8, toast_y - th - 6), (tw + 20, toast_y + 6),
                              self.BG_COLOR, -1)
                cv2.rectangle(canvas, (8, toast_y - th - 6), (tw + 20, toast_y + 6),
                              faded, 1)
                cv2.putText(canvas, text, (14, toast_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, faded, 1, cv2.LINE_AA)
                toast_y -= th + 18

        return canvas

    def _draw_overlays(self, frame):
        """Draw detection boxes on the frame."""
        for i, ann in enumerate(self.current_annotations):
            ext = ann['mnemosky_ext']
            status = ext['status']
            is_selected = (i == self.selected_ann_idx)

            # COCO bbox to xyxy
            bx, by, bw, bh = ann['bbox']
            x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)

            if status == 'confirmed':
                color = self.CONFIRMED_COLOR
                thickness = 3 if is_selected else 1
            elif status == 'rejected':
                color = self.REJECTED_COLOR
                thickness = 1
            else:  # pending
                color = self.ACCENT if is_selected else tuple(int(c * 0.7) for c in self.ACCENT)
                thickness = 3 if is_selected else 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Detection number label
            label = f"#{i+1}"
            cat_name = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], '?')[:3].upper()
            label += f" {cat_name}"
            conf = ext.get('confidence', 0)
            if conf < 0.5:
                label += " *"
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw missed annotations
        for missed in self.current_missed:
            bx, by, bw, bh = missed['bbox']
            x1, y1, x2, y2 = int(bx), int(by), int(bx + bw), int(by + bh)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.MISSED_COLOR, 2)
            cat_label = "MISSED SAT" if missed.get('category_id', 0) == 0 else "MISSED AIR"
            cv2.putText(frame, cat_label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.MISSED_COLOR, 1, cv2.LINE_AA)

        # Draw bbox being marked
        if self.mark_mode and self.mark_start and self.mark_end:
            cv2.rectangle(frame, self.mark_start, self.mark_end, self.MISSED_COLOR, 2)

        # Draw pending mark-missed bbox with pulsing effect
        if self.mark_pending_bbox is not None:
            px1, py1, px2, py2 = self.mark_pending_bbox
            pulse = 0.5 + 0.5 * abs((self._tick % 30) - 15) / 15.0
            pcolor = tuple(int(c * pulse) for c in self.MISSED_COLOR)
            cv2.rectangle(frame, (px1, py1), (px2, py2), pcolor, 2)
            cv2.putText(frame, "S/P?", (px1, py1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pcolor, 1, cv2.LINE_AA)

    def _render_sidebar(self, canvas):
        """Draw detection cards, session stats, and parameter display."""
        sx = self.vid_w
        sy = 0
        sw = self.SIDEBAR_W
        sh = self.vid_h

        # Background
        canvas[sy:sy+sh, sx:sx+sw] = self.PANEL_COLOR

        # Title
        cv2.putText(canvas, "DETECTIONS", (sx + 10, sy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ACCENT, 1, cv2.LINE_AA)
        count_text = f"({len(self.current_annotations)})"
        cv2.putText(canvas, count_text, (sx + 130, sy + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.DIM_TEXT, 1, cv2.LINE_AA)

        y = sy + 45
        for i, ann in enumerate(self.current_annotations):
            if y + 70 > sy + sh - 280:
                cv2.putText(canvas, f"... +{len(self.current_annotations) - i} more",
                            (sx + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            self.DIM_TEXT, 1, cv2.LINE_AA)
                break

            ext = ann['mnemosky_ext']
            is_selected = (i == self.selected_ann_idx)
            card_color = self.ACCENT if is_selected else self.BG_COLOR
            cv2.rectangle(canvas, (sx + 5, y), (sx + sw - 5, y + 65), card_color,
                          2 if is_selected else 1)

            cat_name = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], '?')[:3].upper()
            conf = ext.get('confidence', 0.5)
            status = ext['status']
            status_label = {'pending': 'REVIEW', 'confirmed': 'ACCEPT', 'rejected': 'REJECT'}.get(status, status)
            status_color = {'pending': self.AMBER, 'confirmed': self.CONFIRMED_COLOR,
                            'rejected': self.REJECTED_COLOR}.get(status, self.DIM_TEXT)

            cv2.putText(canvas, f"#{i+1} {cat_name}", (sx + 12, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"conf {conf:.2f}", (sx + 12, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.DIM_TEXT, 1, cv2.LINE_AA)

            meta = ext.get('detection_meta', {})
            info_parts = []
            if 'length' in meta:
                info_parts.append(f"L={meta['length']:.0f}")
            if 'angle' in meta:
                info_parts.append(f"A={meta['angle']:.0f}")
            if info_parts:
                cv2.putText(canvas, "  ".join(info_parts), (sx + 12, y + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.DIM_TEXT, 1, cv2.LINE_AA)

            # Status badge
            cv2.putText(canvas, f"[{status_label}]", (sx + 170, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, status_color, 1, cv2.LINE_AA)

            if conf < 0.5:
                cv2.putText(canvas, "*", (sx + sw - 20, y + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.AMBER, 1, cv2.LINE_AA)

            y += 72

        # --- Review progress bar ---
        prog_y = sy + sh - 270
        cv2.line(canvas, (sx + 10, prog_y), (sx + sw - 10, prog_y), self.DIM_TEXT, 1)
        prog_y += 20
        cv2.putText(canvas, "PROGRESS", (sx + 10, prog_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT, 1, cv2.LINE_AA)
        prog_y += 18
        reviewed_frames = self._count_reviewed_frames()
        total_frames = max(1, self._frames_with_detections)
        pct = int(100 * reviewed_frames / total_frames)
        bar_x1, bar_x2 = sx + 10, sx + sw - 10
        bar_w_total = bar_x2 - bar_x1
        cv2.rectangle(canvas, (bar_x1, prog_y), (bar_x2, prog_y + 12), self.BG_COLOR, -1)
        fill_w = int(bar_w_total * reviewed_frames / total_frames)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x1, prog_y), (bar_x1 + fill_w, prog_y + 12),
                          self.CONFIRMED_COLOR, -1)
        cv2.rectangle(canvas, (bar_x1, prog_y), (bar_x2, prog_y + 12), self.DIM_TEXT, 1)
        cv2.putText(canvas, f"{pct}%", (bar_x1 + bar_w_total // 2 - 10, prog_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.TEXT_COLOR, 1, cv2.LINE_AA)
        prog_y += 20
        cv2.putText(canvas, f"  Frames: {reviewed_frames}/{total_frames}",
                    (sx + 10, prog_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)
        prog_y += 17
        cv2.putText(canvas, f"  Detections: {self.stats['reviewed']}/{self._total_detections}",
                    (sx + 10, prog_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)

        # --- Session stats ---
        stats_y = prog_y + 20
        cv2.line(canvas, (sx + 10, stats_y), (sx + sw - 10, stats_y), self.DIM_TEXT, 1)
        stats_y += 20
        cv2.putText(canvas, "SESSION STATS", (sx + 10, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT, 1, cv2.LINE_AA)
        for label, key, color in [
            ("Accepted", "accepted", self.CONFIRMED_COLOR),
            ("Rejected", "rejected", self.REJECTED_COLOR),
            ("Reclassed", "reclassified", self.AMBER),
            ("Missed added", "missed", self.MISSED_COLOR),
        ]:
            stats_y += 18
            cv2.putText(canvas, f"  {label}: {self.stats[key]}", (sx + 10, stats_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # --- Improvements ---
        stats_y += 22
        cv2.line(canvas, (sx + 10, stats_y), (sx + sw - 10, stats_y), self.DIM_TEXT, 1)
        stats_y += 20
        cv2.putText(canvas, "IMPROVEMENTS", (sx + 10, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT, 1, cv2.LINE_AA)
        stats_y += 18
        cv2.putText(canvas, f"  Param tweaks: {self.stats['param_adjustments']}",
                    (sx + 10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)
        stats_y += 18
        cv2.putText(canvas, f"  Learn runs: {self.stats['learn_runs']}",
                    (sx + 10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)
        stats_y += 18
        cv2.putText(canvas, f"  Params optimized: {self.stats['params_optimized']}",
                    (sx + 10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    self.TEXT_COLOR, 1, cv2.LINE_AA)

        # Controls hint
        stats_y += 22
        cv2.putText(canvas, "[H] Help  [Q] Quit", (sx + 10, stats_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.DIM_TEXT, 1, cv2.LINE_AA)

    def _render_status_bar(self, canvas):
        """Draw bottom bar with frame slider and progress."""
        bar_y = self.vid_h
        bar_w = canvas.shape[1]
        canvas[bar_y:bar_y + self.STATUS_H, :] = self.BG_COLOR

        # Frame info + loading indicator
        if self._loading_frame:
            mode_text = "LOADING..."
            mode_color = self.AMBER
        elif self.mark_pending_bbox is not None:
            mode_text = "S=Satellite  P=Airplane  Esc=Cancel"
            mode_color = self.MISSED_COLOR
        elif self.mark_mode:
            mode_text = "MARK MISSED"
            mode_color = self.MISSED_COLOR
        else:
            mode_text = "REVIEW"
            mode_color = self.ACCENT
        cv2.putText(canvas, f"  {mode_text}", (10, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_color, 1, cv2.LINE_AA)

        # Frame counter + pending on this frame
        pending_here = sum(1 for a in self.current_annotations
                           if a['mnemosky_ext']['status'] == 'pending')
        frame_text = f"frame {self.current_frame_idx + 1}/{self.total_frames}"
        if pending_here > 0:
            frame_text += f"  ({pending_here} pending)"
        cv2.putText(canvas, frame_text, (10, bar_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.DIM_TEXT, 1, cv2.LINE_AA)

        # Frame slider with detection markers
        slider_x = 200
        slider_w = bar_w - 400
        slider_y = bar_y + 28
        if slider_w > 100 and self.total_frames > 0:
            cv2.line(canvas, (slider_x, slider_y), (slider_x + slider_w, slider_y),
                     self.DIM_TEXT, 2)
            # Draw tick marks for frames with detections
            for fi in self.detections_by_frame:
                tick_x = int(slider_x + (fi / max(1, self.total_frames - 1)) * slider_w)
                cv2.line(canvas, (tick_x, slider_y - 4), (tick_x, slider_y + 4),
                         self.DIM_TEXT, 1)
            # Current position
            pos = int(slider_x + (self.current_frame_idx / max(1, self.total_frames - 1)) * slider_w)
            cv2.circle(canvas, (pos, slider_y), 8, self.ACCENT, -1)

        # Right-side buttons
        btn_x = bar_w - 280
        cv2.putText(canvas, "[E] EXPORT", (btn_x, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[L] LEARN", (btn_x + 100, bar_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, "[Ctrl+S] SAVE", (btn_x + 100, bar_y + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.DIM_TEXT, 1, cv2.LINE_AA)

    def _render_help(self, canvas):
        """Draw help overlay."""
        h, w = canvas.shape[:2]
        overlay = canvas.copy()
        cv2.rectangle(overlay, (50, 50), (w - 50, h - 50), self.BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (50, 50), (w - 50, h - 50), self.ACCENT, 2)

        lines = [
            "KEYBOARD SHORTCUTS",
            "",
            "A        Accept selected detection",
            "R        Reject selected detection",
            "S        Reclassify as satellite",
            "P        Reclassify as airplane (plane)",
            "M        Mark missed (draw bbox, then S/P)",
            "Escape   Cancel / exit mark mode",
            "Tab      Next detection",
            "Right/D  Next frame",
            "Left/W   Previous frame",
            "N        Next unreviewed frame",
            "Space    Accept all on frame",
            "X        Reject all on frame",
            "1-9      Select detection by number",
            "Z        Undo last correction",
            "L        Run learning",
            "E        Export YOLO dataset",
            "F        Toggle full frame view",
            "H        Toggle this help",
            "Q        Quit (prompts save)",
        ]
        y = 90
        for line in lines:
            color = self.ACCENT if line == "KEYBOARD SHORTCUTS" else self.TEXT_COLOR
            size = 0.5 if line == "KEYBOARD SHORTCUTS" else 0.4
            cv2.putText(canvas, line, (80, y), cv2.FONT_HERSHEY_SIMPLEX,
                        size, color, 1, cv2.LINE_AA)
            y += 22

    def _handle_key(self, key):
        """Process keyboard input."""
        # Intercept category selection when a mark-missed bbox is pending
        if self.mark_pending_bbox is not None:
            if key == ord('s') or key == ord('S'):
                self._commit_mark_missed(0)  # satellite
            elif key == ord('p') or key == ord('P'):
                self._commit_mark_missed(1)  # airplane
            elif key == 27:  # Escape - cancel
                self.mark_pending_bbox = None
                self.mark_pending_meta = None
                self.mark_pending_img_id = None
                self._toast("Mark cancelled", self.DIM_TEXT)
            # Ignore all other keys during pending state
            return

        if key == ord('q') or key == ord('Q'):
            self._prompt_save_and_quit()
        elif key == ord('h') or key == ord('H'):
            self.help_visible = not self.help_visible
        elif key == ord('f') or key == ord('F'):
            self.full_frame_mode = not self.full_frame_mode
        elif key == 27:  # Escape
            if self.mark_mode:
                self.mark_mode = False
                self.mark_start = None
                self.mark_end = None
            elif self.help_visible:
                self.help_visible = False
        elif key == ord('a') or key == ord('A'):
            self._accept_selected()
        elif key == ord('r'):
            self._reject_selected()
        elif key == ord('s'):
            self._reclassify_selected(0)  # satellite
        elif key == ord('p'):
            self._reclassify_selected(1)  # airplane
        elif key == ord('m') or key == ord('M'):
            self.mark_mode = not self.mark_mode
            self.mark_start = None
            self.mark_end = None
        elif key == 9:  # Tab
            if self.current_annotations:
                self.selected_ann_idx = (self.selected_ann_idx + 1) % len(self.current_annotations)
        elif key == ord('d') or key == ord('D') or key == 83 or key == 3:  # Right arrow
            self._navigate_to_frame(self.current_frame_idx + 1)
        elif key == ord('w') or key == 81 or key == 2:  # Left arrow  (w or left arrow)
            self._navigate_to_frame(self.current_frame_idx - 1)
        elif key == ord('n') or key == ord('N'):
            self._next_unreviewed_frame()
        elif key == ord(' '):  # Space = accept all
            self._accept_all_on_frame()
        elif key == ord('x') or key == ord('X'):
            self._reject_all_on_frame()
        elif key == ord('z') or key == ord('Z'):
            self._undo()
        elif key == ord('l') or key == ord('L'):
            self._run_learning()
        elif key == ord('e') or key == ord('E'):
            self._export_dataset()
        elif key == 19:  # Ctrl+S
            self.ann_db.save()
            if self.param_adapter:
                self.param_adapter.save_profile()
            self._toast("Annotations + profile saved", self.CONFIRMED_COLOR)
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self.current_annotations):
                self.selected_ann_idx = idx

    def _handle_mouse(self, event, x, y, flags, param):
        """Process mouse input."""
        if self.mark_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mark_start = (x, y)
                self.mark_end = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.mark_start:
                self.mark_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.mark_start:
                self.mark_end = (x, y)
                self._finish_mark_missed()
            return

        # Click on status bar slider
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.vid_h:
            slider_x = 200
            sidebar_w = 0 if self.full_frame_mode else self.SIDEBAR_W
            slider_w = (self.vid_w + sidebar_w) - 400
            if slider_w > 100 and slider_x <= x <= slider_x + slider_w:
                frac = (x - slider_x) / slider_w
                target = int(frac * (self.total_frames - 1))
                self._navigate_to_frame(target)
            return

        # Click on main frame area to select detection
        if event == cv2.EVENT_LBUTTONDOWN and x < self.vid_w and y < self.vid_h:
            for i, ann in enumerate(self.current_annotations):
                bx, by_, bw, bh = ann['bbox']
                if bx <= x <= bx + bw and by_ <= y <= by_ + bh:
                    self.selected_ann_idx = i
                    return

        # Click on sidebar card
        if (event == cv2.EVENT_LBUTTONDOWN and not self.full_frame_mode
                and x >= self.vid_w):
            card_y = 45
            for i in range(len(self.current_annotations)):
                if card_y <= y <= card_y + 65:
                    self.selected_ann_idx = i
                    return
                card_y += 72

    def _accept_selected(self):
        """Accept the currently selected detection."""
        if not self.current_annotations or self.selected_ann_idx >= len(self.current_annotations):
            return
        ann = self.current_annotations[self.selected_ann_idx]
        if ann['mnemosky_ext']['status'] == 'pending':
            cat = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], '?')[:3].upper()
            self.ann_db.record_correction(ann['id'], 'accept')
            self.stats['accepted'] += 1
            self.stats['reviewed'] += 1
            self._toast(f"Accepted #{self.selected_ann_idx+1} {cat}", self.CONFIRMED_COLOR)
            self._reload_current_frame_annotations()

    def _reject_selected(self):
        """Reject the currently selected detection."""
        if not self.current_annotations or self.selected_ann_idx >= len(self.current_annotations):
            return
        ann = self.current_annotations[self.selected_ann_idx]
        if ann['mnemosky_ext']['status'] == 'pending':
            cat = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], '?')[:3].upper()
            self.ann_db.record_correction(ann['id'], 'reject')
            self.stats['rejected'] += 1
            self.stats['reviewed'] += 1
            # Apply Tier 1 learning
            if self.param_adapter:
                trail_type = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], 'satellite')
                meta = ann['mnemosky_ext'].get('detection_meta', {})
                self.param_adapter.apply_correction('reject', trail_type, meta)
                self.stats['param_adjustments'] += 1
            self._toast(f"Rejected #{self.selected_ann_idx+1} {cat}", self.REJECTED_COLOR)
            self._reload_current_frame_annotations()

    def _reclassify_selected(self, new_category_id):
        """Reclassify the currently selected detection."""
        if not self.current_annotations or self.selected_ann_idx >= len(self.current_annotations):
            return
        ann = self.current_annotations[self.selected_ann_idx]
        if ann['category_id'] == new_category_id:
            return
        old_type = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], 'satellite')
        new_type = AnnotationDatabase.CATEGORY_MAP.get(new_category_id, 'satellite')
        self.ann_db.record_correction(ann['id'], 'reclassify', new_category_id=new_category_id)
        self.stats['reclassified'] += 1
        self.stats['reviewed'] += 1
        # Apply Tier 1 learning
        if self.param_adapter:
            meta = ann['mnemosky_ext'].get('detection_meta', {})
            action = f'reclassify_to_{new_type}'
            self.param_adapter.apply_correction(action, old_type, meta)
            self.stats['param_adjustments'] += 1
        self._toast(f"Reclassed #{self.selected_ann_idx+1} -> {new_type}", self.AMBER)
        self._reload_current_frame_annotations()

    def _accept_all_on_frame(self):
        """Accept all pending detections on current frame."""
        count = 0
        for ann in self.current_annotations:
            if ann['mnemosky_ext']['status'] == 'pending':
                self.ann_db.record_correction(ann['id'], 'accept')
                self.stats['accepted'] += 1
                self.stats['reviewed'] += 1
                count += 1
        if count:
            self._toast(f"Accepted all {count} detections", self.CONFIRMED_COLOR)
        self._reload_current_frame_annotations()

    def _reject_all_on_frame(self):
        """Reject all pending detections on current frame."""
        count = 0
        for ann in self.current_annotations:
            if ann['mnemosky_ext']['status'] == 'pending':
                self.ann_db.record_correction(ann['id'], 'reject')
                self.stats['rejected'] += 1
                self.stats['reviewed'] += 1
                count += 1
                if self.param_adapter:
                    trail_type = AnnotationDatabase.CATEGORY_MAP.get(ann['category_id'], 'satellite')
                    meta = ann['mnemosky_ext'].get('detection_meta', {})
                    self.param_adapter.apply_correction('reject', trail_type, meta)
                    self.stats['param_adjustments'] += 1
        if count:
            self._toast(f"Rejected all {count} detections", self.REJECTED_COLOR)
        self._reload_current_frame_annotations()

    def _finish_mark_missed(self):
        """Complete marking a missed detection."""
        if not self.mark_start or not self.mark_end:
            self.mark_mode = False
            return
        x1, y1 = self.mark_start
        x2, y2 = self.mark_end
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        if x_max - x_min < 10 or y_max - y_min < 5:
            self.mark_mode = False
            self.mark_start = None
            self.mark_end = None
            return

        # Find or create image entry
        img = None
        for im in self.ann_db.data['images']:
            if im['frame_index'] == self.current_frame_idx:
                img = im
                break
        if not img:
            video_src = self.ann_db.data['sessions'][-1]['video_source'] if self.ann_db.data['sessions'] else self.video_path
            img_id = self.ann_db.add_image(self.current_frame_idx, video_src, self.vid_w, self.vid_h)
        else:
            img_id = img['id']

        # Estimate metadata from the marked region
        estimated_meta = {}
        if self.current_frame is not None:
            roi = self.current_frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
                estimated_meta['avg_brightness'] = float(np.mean(gray_roi))
                estimated_meta['length'] = float(max(x_max - x_min, y_max - y_min))
                estimated_meta['angle'] = 0.0 if (x_max - x_min) > (y_max - y_min) else 90.0

        # Store pending state — user must choose category (S=satellite, P=airplane)
        self.mark_pending_bbox = (x_min, y_min, x_max, y_max)
        self.mark_pending_meta = estimated_meta
        self.mark_pending_img_id = img_id

        self.mark_mode = False
        self.mark_start = None
        self.mark_end = None
        self._toast("S=Satellite  P=Airplane  Esc=Cancel", self.MISSED_COLOR, duration=120)

    def _commit_mark_missed(self, category_id):
        """Commit a pending missed detection with the chosen category."""
        if self.mark_pending_bbox is None:
            return
        trail_type = AnnotationDatabase.CATEGORY_MAP.get(category_id, 'satellite')
        self.ann_db.add_missed(self.mark_pending_img_id, category_id,
                               self.mark_pending_bbox, self.mark_pending_meta)
        self.stats['missed'] += 1
        self.stats['reviewed'] += 1

        # Apply Tier 1 learning for missed trail
        if self.param_adapter:
            self.param_adapter.apply_correction('add_missed', trail_type,
                                                self.mark_pending_meta)
            self.stats['param_adjustments'] += 1

        length = self.mark_pending_meta.get('length', 0)
        self._toast(f"Marked missed {trail_type} (L={length:.0f}px)", self.MISSED_COLOR)

        self.mark_pending_bbox = None
        self.mark_pending_meta = None
        self.mark_pending_img_id = None
        self._reload_current_frame_annotations()

    def _reload_current_frame_annotations(self):
        """Reload annotations for current frame after changes."""
        img = None
        for im in self.ann_db.data['images']:
            if im['frame_index'] == self.current_frame_idx:
                img = im
                break
        if img:
            self.current_annotations = self.ann_db.get_annotations_for_image(img['id'])
            self.current_missed = self.ann_db.get_missed_for_image(img['id'])
            self.current_annotations.sort(
                key=lambda a: a['mnemosky_ext'].get('confidence', 0.5))
        if self.selected_ann_idx >= len(self.current_annotations):
            self.selected_ann_idx = max(0, len(self.current_annotations) - 1)

    def _next_unreviewed_frame(self):
        """Jump to next frame with pending detections."""
        for fi in self.review_queue:
            if fi > self.current_frame_idx:
                for im in self.ann_db.data['images']:
                    if im['frame_index'] == fi:
                        pending = self.ann_db.get_pending_annotations(im['id'])
                        if pending:
                            self._toast(f"-> frame {fi+1} ({len(pending)} pending)", self.ACCENT)
                            self._navigate_to_frame(fi)
                            return
        # Wrap around
        for fi in self.review_queue:
            if fi != self.current_frame_idx:
                for im in self.ann_db.data['images']:
                    if im['frame_index'] == fi:
                        pending = self.ann_db.get_pending_annotations(im['id'])
                        if pending:
                            self._toast(f"-> frame {fi+1} (wrapped, {len(pending)} pending)", self.ACCENT)
                            self._navigate_to_frame(fi)
                            return
        self._toast("All frames reviewed!", self.CONFIRMED_COLOR, duration=90)

    def _run_learning(self):
        """Trigger Tier 2 batch optimization."""
        if not self.param_adapter:
            self._toast("Learning disabled (--no-learn)", self.REJECTED_COLOR)
            return
        cal_set = self.ann_db.get_calibration_set()
        if len(cal_set) < 10:
            self._toast(f"Need 10+ corrections (have {len(cal_set)})", self.AMBER)
            return
        self._toast("Running batch optimization...", self.ACCENT, duration=90)
        optimized = self.param_adapter.optimize_batch(cal_set)
        # Update detector params
        n_updated = 0
        for k, v in optimized.items():
            if k in self.detector.params:
                self.detector.params[k] = v
                n_updated += 1
        self.stats['learn_runs'] += 1
        self.stats['params_optimized'] += n_updated
        self._toast(f"Optimized {n_updated} params from {len(cal_set)} corrections",
                    self.CONFIRMED_COLOR, duration=90)

    def _export_dataset(self):
        """Export YOLO dataset from confirmed + missed annotations."""
        # Count exportable annotations
        n_confirmed = sum(1 for a in self.ann_db.data['annotations']
                          if a.get('mnemosky_ext', {}).get('status') == 'confirmed')
        n_missed = len(self.ann_db.data.get('missed_annotations', []))
        total = n_confirmed + n_missed
        if total == 0:
            self._toast("No confirmed/missed annotations to export", self.AMBER)
            return
        # Save DB first to ensure export reads latest data
        self.ann_db.save()
        self._toast(f"Exporting {total} annotations...", self.ACCENT, duration=120)
        # Render so user sees the toast
        canvas = self._render()
        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(1)
        try:
            ann_path = self.ann_db._path
            if not ann_path:
                self._toast("No annotation file path — save first", self.REJECTED_COLOR)
                return
            export_dataset_from_annotations(
                input_path=self.video_path,
                annotations_path=str(ann_path),
            )
            self._toast(f"Exported YOLO dataset ({total} annotations)", self.CONFIRMED_COLOR, duration=90)
        except Exception as e:
            self._toast(f"Export failed: {e}", self.REJECTED_COLOR, duration=90)

    def _undo(self):
        """Undo last correction."""
        if self.ann_db.undo_last_correction():
            # Adjust stats (approximate -- just decrement reviewed)
            self.stats['reviewed'] = max(0, self.stats['reviewed'] - 1)
            self._reload_current_frame_annotations()
            self._toast("Undid last correction", self.AMBER)
        else:
            self._toast("Nothing to undo", self.DIM_TEXT)

    def _prompt_save_and_quit(self):
        """Save and quit."""
        self.ann_db.save()
        if self.param_adapter:
            self.param_adapter.save_profile()
        self._toast("Saved! Exiting...", self.CONFIRMED_COLOR, duration=30)
        # Show one final render so user sees the toast
        canvas = self._render()
        cv2.imshow(self.WINDOW_NAME, canvas)
        cv2.waitKey(500)
        print("Annotations saved. Exiting review mode.")
        self.running = False


class SatelliteTrailDetector:
    """
    Satellite and airplane trail detector with configurable sensitivity presets.

    Detects and classifies satellite and airplane trails in video frames using
    line detection and morphological operations.

    This class provides high-level detection interface with sensitivity presets
    (low, medium, high) and optional custom preprocessing parameters.
    """

    def __init__(self, sensitivity='medium', preprocessing_params=None, skip_aspect_ratio_check=False, signal_envelope=None):
        """
        Initialize detector with sensitivity level and optional custom preprocessing.

        Args:
            sensitivity: 'low', 'medium', or 'high' - affects detection thresholds
            preprocessing_params: Optional dict with custom preprocessing parameters:
                - clahe_clip_limit: CLAHE clip limit (default: 4.0)
                - clahe_tile_size: CLAHE tile grid size (default: 6)
                - blur_kernel_size: Gaussian blur kernel size (default: 3, must be odd)
                - blur_sigma: Gaussian blur sigma (default: 0.3)
                - canny_low: Canny edge detection low threshold
                - canny_high: Canny edge detection high threshold
            skip_aspect_ratio_check: If True, disables aspect ratio filtering (default: False)
            signal_envelope: Optional dict from user-marked trail examples (computed by
                show_preprocessing_preview). Contains brightness, contrast, length, and
                angle ranges measured on real satellite trails, used to dynamically adapt
                detection thresholds.
        """
        # Store custom preprocessing parameters
        self.preprocessing_params = preprocessing_params
        self.skip_aspect_ratio_check = skip_aspect_ratio_check
        self.signal_envelope = signal_envelope

        # Sensitivity presets - rebalanced to reduce false positives
        # Satellite length ranges are generous because trails can span large
        # portions of the frame depending on exposure and satellite altitude.
        presets = {
            'low': {
                'min_line_length': 80,  # Longer minimum to reduce noise
                'max_line_gap': 40,  # Moderate gap tolerance
                'canny_low': 8,  # Less sensitive to reduce edge noise
                'canny_high': 60,
                'hough_threshold': 45,  # Higher threshold for fewer false detections
                'min_aspect_ratio': 4,  # Stricter aspect ratio for true trails
                'brightness_threshold': 25,
                'airplane_brightness_min': 90,
                'airplane_saturation_min': 10,
                'satellite_min_length': 120,  # Satellite trail length range (1920x1080)
                'satellite_max_length': 700,
                'satellite_contrast_min': 1.10,  # Minimum trail-to-background contrast
            },
            'medium': {
                'min_line_length': 50,  # Lower to catch dim trail fragments
                'max_line_gap': 35,  # Moderate gap tolerance (MF handles fragmented dim trails)
                'canny_low': 4,  # Slightly more sensitive for dim trails
                'canny_high': 45,
                'hough_threshold': 30,  # Lower threshold to catch dim trails
                'min_aspect_ratio': 4,  # Require trails to be relatively long and thin
                'brightness_threshold': 18,
                'airplane_brightness_min': 75,
                'airplane_saturation_min': 8,
                'satellite_min_length': 100,  # Satellites can be shorter segments
                'satellite_max_length': 1000,  # Long trails but not full-frame artifacts
                'satellite_contrast_min': 1.08,  # Lower contrast for dim satellites
            },
            'high': {
                'min_line_length': 35,  # Catches shorter trail fragments
                'max_line_gap': 60,  # Very tolerant of breaks in dim trails
                'canny_low': 2,  # Very sensitive edge detection
                'canny_high': 35,
                'hough_threshold': 20,  # Lower threshold for more detections
                'min_aspect_ratio': 3,  # More relaxed but not too permissive
                'brightness_threshold': 12,
                'airplane_brightness_min': 45,
                'airplane_saturation_min': 2,
                'satellite_min_length': 60,  # Very short fragments allowed
                'satellite_max_length': 1400,  # Very long trails but reject full-frame artifacts
                'satellite_contrast_min': 1.03,  # Very dim trails allowed (groundtruth ~1.03x)
                'mf_sigma_perp': 0.7,  # Narrower kernel to match dim trail PSF (~1.3-2px FWHM)
            }
        }

        self.params = presets.get(sensitivity, presets['medium'])
        # Yellowish complementary color palette (BGR format)
        self.satellite_color = (0, 185, 255)  # Gold/yellow for satellites
        self.airplane_color = (0, 140, 255)   # Orange/amber for airplanes
        self.anomalous_color = (200, 50, 200)  # Magenta for anomalous (Bowker & Star)
        self.ledger = None  # TranslationLedger instance (set externally if --ledger)
        self.sensitivity = sensitivity  # Store for inscription metadata
        self.box_thickness = 1
        self.dot_length = 8  # Length of each dash in dotted line
        self.gap_length = 4  # Gap between dashes

        # Apply custom thresholds from preview to params if provided
        if self.preprocessing_params:
            if 'canny_low' in self.preprocessing_params:
                self.params['canny_low'] = self.preprocessing_params['canny_low']
            if 'canny_high' in self.preprocessing_params:
                self.params['canny_high'] = self.preprocessing_params['canny_high']
            if 'mf_snr_threshold' in self.preprocessing_params:
                self.params['mf_snr_threshold'] = self.preprocessing_params['mf_snr_threshold']

        # Per-operation GPU flags — a failure in one GPU path (e.g. median
        # blur) should not disable unrelated GPU ops (e.g. warpAffine).
        self._use_gpu = _HAS_CUDA
        self._use_gpu_filter = _HAS_CUDA      # filter2D (matched filter)
        self._use_gpu_warp = _HAS_CUDA         # warpAffine (Radon)
        self._use_gpu_median = _HAS_CUDA       # median blur (star mask / bg)

        # Pre-compute matched filter kernel bank (48 kernels: 24 angles x 2 lengths)
        # These depend only on num_angles, kernel_lengths, and sigma_perp —
        # all fixed at init time.  Saves ~5-15ms per frame by avoiding
        # repeated numpy allocations and trig calls.
        sigma_perp = self.params.get('mf_sigma_perp', 1.0)
        self._mf_kernel_bank = []
        for klen in [21, 51]:
            for i in range(24):
                angle_deg = i * 180.0 / 24
                kernel = self._create_matched_filter_kernel(klen, angle_deg, sigma_perp=sigma_perp)
                noise_factor = np.sqrt(np.sum(kernel ** 2))
                self._mf_kernel_bank.append((angle_deg, kernel, noise_factor))
        # Pre-create GPU filter objects if CUDA is available
        self._gpu_filters = None
        if self._use_gpu_filter:
            try:
                self._gpu_filters = []
                for angle_deg, kernel, noise_factor in self._mf_kernel_bank:
                    filt = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, kernel)
                    self._gpu_filters.append((angle_deg, filt, noise_factor))
            except Exception:
                self._gpu_filters = None

        # Pre-create CLAHE object (reused every frame with same parameters)
        if self.preprocessing_params:
            clip_limit = self.preprocessing_params.get('clahe_clip_limit', 6.0)
            tile_size = self.preprocessing_params.get('clahe_tile_size', 6)
        else:
            clip_limit = 6.0
            tile_size = 6
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

        # Adapt detection parameters from user-marked trail signal envelope
        if self.signal_envelope:
            self._apply_signal_envelope(self.signal_envelope)

    def _apply_signal_envelope(self, env):
        """Dynamically adapt detection thresholds to match user-marked trail examples.

        The envelope provides measured brightness, contrast, length, and smoothness
        from real satellite trails in the video. We widen the detector's acceptance
        windows so that trails similar to the marked examples are reliably detected.
        """
        # ── Length range: ensure detector covers the marked examples ──
        env_min_len, env_max_len = env['length_range']
        self.params['satellite_min_length'] = min(
            self.params['satellite_min_length'], max(30, int(env_min_len)))
        self.params['satellite_max_length'] = max(
            self.params['satellite_max_length'], int(env_max_len))

        # Also lower min_line_length if examples are short
        self.params['min_line_length'] = min(
            self.params['min_line_length'], max(20, int(env_min_len * 0.5)))

        # ── Contrast: lower threshold to admit dimmer trails ──────────
        env_min_contrast = env['contrast_range'][0]
        self.params['satellite_contrast_min'] = min(
            self.params['satellite_contrast_min'],
            max(1.01, env_min_contrast))

        # ── Brightness: ensure marked-trail brightness isn't rejected ─
        env_max_brightness = env['brightness_range'][1]
        # Raise brightness_threshold ceiling so dim examples aren't lost
        self.params['brightness_threshold'] = max(
            self.params['brightness_threshold'],
            int(env_max_brightness))
        # Push airplane brightness floor above the satellite range so marked
        # examples are not mis-classified as airplanes
        self.params['airplane_brightness_min'] = max(
            self.params['airplane_brightness_min'],
            int(env_max_brightness * 1.5))

        # ── Hough: be more lenient to catch fragments of similar trails ─
        self.params['max_line_gap'] = max(
            self.params['max_line_gap'],
            min(80, int(env_max_len * 0.10)))

    @staticmethod
    def _rotated_kernel_endpoints(size, angle_deg):
        """Return two endpoint tuples for a line through the center of a (size x size) grid."""
        import math
        cx = cy = size // 2
        half = size // 2
        rad = math.radians(angle_deg)
        dx = int(round(half * math.cos(rad)))
        dy = int(round(half * math.sin(rad)))
        return (cx - dx, cy - dy), (cx + dx, cy + dy)

    def preprocess_frame(self, frame):
        """Convert frame to grayscale and enhance for trail detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get preprocessing parameters (use custom if available, otherwise defaults)
        if self.preprocessing_params:
            blur_kernel = self.preprocessing_params.get('blur_kernel_size', 3)
            blur_sigma = self.preprocessing_params.get('blur_sigma', 0.3)
        else:
            blur_kernel = 3
            blur_sigma = 0.3

        # Ensure blur kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # Apply CLAHE for contrast enhancement (reuse cached object)
        enhanced = self._clahe.apply(gray)

        # Apply Gaussian blur - smoothing to reduce sensor noise
        if blur_kernel >= 1 and blur_sigma > 0:
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), blur_sigma)
        else:
            blurred = enhanced

        return gray, blurred

    def detect_lines(self, preprocessed):
        """Detect lines using Canny edge detection and Hough transform."""
        # Edge detection
        edges = cv2.Canny(
            preprocessed,
            self.params['canny_low'],
            self.params['canny_high']
        )

        # Morphological operations to connect broken trails
        kernel = np.ones((3, 3), np.uint8)

        # Dilate to connect gaps in dim trails, then erode to clean up
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Directional dilation to bridge gaps in linear features.
        # Use 2 orientations (0° and 90°) — the Hough transform handles
        # intermediate angles, so full 4-angle bridging is redundant.
        for angle in [0, 90]:
            line_kernel = np.zeros((7, 7), dtype=np.uint8)
            cv2.line(line_kernel, *self._rotated_kernel_endpoints(7, angle), 1, thickness=1)
            edges = cv2.dilate(edges, line_kernel, iterations=1)

        # Clean up directional dilation
        edges = cv2.erode(edges, kernel, iterations=1)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.params['hough_threshold'],
            minLineLength=self.params['min_line_length'],
            maxLineGap=self.params['max_line_gap']
        )

        # Cap line count to avoid O(N) classify_trail bottleneck.
        # Keep the longest lines (most likely to be real trails).
        max_lines = 100
        if lines is not None and len(lines) > max_lines:
            lengths = np.sqrt((lines[:, 0, 2] - lines[:, 0, 0]).astype(np.float64)**2 +
                              (lines[:, 0, 3] - lines[:, 0, 1]).astype(np.float64)**2)
            top_indices = np.argpartition(lengths, -max_lines)[-max_lines:]
            lines = lines[top_indices]

        return lines, edges

    # ------------------------------------------------------------------
    #  Supplementary dim-trail detection: Directional Matched Filtering
    # ------------------------------------------------------------------
    #
    #  The primary pipeline (Canny → Hough) cannot detect trails whose
    #  brightness is too low to produce edges.  The methods below implement
    #  a *matched filter* approach — the theoretically optimal linear
    #  detector for a known signal shape (straight line) in additive
    #  Gaussian noise.
    #
    #  Pipeline:
    #    1. Local background subtraction (large-kernel median)
    #    2. Robust noise estimation (MAD → σ)
    #    3. Oriented filter bank (averaging kernels at 5° steps)
    #    4. SNR thresholding on the filter response map
    #    5. Hough extraction of line segments from the thresholded map
    #    6. Per-trail SNR confirmation using perpendicular flank sampling
    #    7. Duplicate suppression vs. primary detections
    #
    #  This gives ~√L SNR improvement (L = kernel length), allowing
    #  detection of trails 3–6× dimmer than the Canny threshold.
    # ------------------------------------------------------------------

    @staticmethod
    def _create_matched_filter_kernel(length, angle_deg, sigma_perp=1.0):
        """Create a Gaussian-profile oriented line kernel.

        Uses an analytic Gaussian cross-section perpendicular to the line
        direction — no integer-rasterisation artifacts, consistent effective
        width at every orientation.  The along-line weight is uniform.

        This matches real trail PSFs better than a 1-pixel-wide rasterised
        line and produces orientation-independent filter responses.

        Args:
            length: Kernel side length (will be rounded up to odd)
            angle_deg: Line orientation in degrees (0 = horizontal)
            sigma_perp: Gaussian width (std dev) perpendicular to the line.
                Controls the effective trail width sensitivity.
                1.0 ≈ 2.4 px FWHM (good for typical satellite trails).

        Returns:
            Normalized float32 kernel array
        """
        import math

        ksize = length if length % 2 == 1 else length + 1
        center = ksize // 2
        rad = math.radians(angle_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Vectorised computation — no per-pixel loop
        y_coords, x_coords = np.mgrid[:ksize, :ksize]
        dx = (x_coords - center).astype(np.float32)
        dy = (y_coords - center).astype(np.float32)

        # Perpendicular distance to the line through the centre
        perp_dist = np.abs(-sin_a * dx + cos_a * dy)
        # Distance along the line from the centre
        along_dist = np.abs(cos_a * dx + sin_a * dy)

        # Gaussian profile perpendicular, uniform along the line
        kernel = np.exp(-0.5 * (perp_dist / sigma_perp) ** 2).astype(np.float32)
        # Zero out beyond the half-length
        kernel[along_dist > center] = 0

        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum

        return kernel

    def _detect_dim_lines_matched_filter(self, gray_frame, existing_lines=None,
                                         temporal_context=None):
        """Supplementary line detection using directional matched filtering.

        Detects dim linear features that Canny + Hough miss by directly
        searching for oriented brightness ridges in the background-subtracted
        image.  A bank of oriented averaging kernels at 5-degree steps is
        applied and the maximum filter response at each pixel is compared
        against a statistical noise threshold (SNR-based).

        When a temporal_context is provided (from TemporalFrameBuffer), it
        replaces the spatial median background subtraction with the far
        superior temporal median difference image.  Stars, vignetting, and
        sky gradients are removed perfectly, and the per-pixel noise map
        gives spatially adaptive thresholding for free.

        The matched filter is the theoretically optimal linear detector for
        a known signal shape (straight line) in additive Gaussian noise,
        providing SNR improvement proportional to sqrt(kernel_length).

        Args:
            gray_frame: Original grayscale frame (not CLAHE-enhanced)
            existing_lines: Lines already found by primary Hough detection,
                used to suppress duplicate detections.  HoughLinesP format.
            temporal_context: Optional dict from TemporalFrameBuffer with keys
                'diff_image' (float32), 'noise_map' (float32).  When provided,
                the spatial median background subtraction and MAD noise
                estimation are skipped — the temporal versions are strictly
                superior.

        Returns:
            Additional lines in HoughLinesP format [[[x1, y1, x2, y2]], ...],
            or None if no additional lines found.
        """
        import math

        h, w = gray_frame.shape

        # --- Downsample for performance (1/2 resolution) ---
        scale = 0.5
        small_w, small_h = int(w * scale), int(h * scale)

        if temporal_context is not None:
            # ── Temporal mode: use pre-computed difference image + noise map
            # The temporal median has already removed stars, sky gradients,
            # vignetting, and consistent hot pixels.  The difference image
            # contains only transient features (trails, cosmic rays, noise).
            diff_full = temporal_context['diff_image']
            noise_full = temporal_context['noise_map']
            signal = cv2.resize(diff_full, (small_w, small_h),
                                interpolation=cv2.INTER_AREA)
            noise_map_small = cv2.resize(noise_full, (small_w, small_h),
                                         interpolation=cv2.INTER_LINEAR)
            # For the scalar noise path below, use median of the noise map
            noise_std = float(np.median(noise_map_small))
            if noise_std < 0.5:
                noise_std = 0.5
            use_noise_map = True
        else:
            # ── Spatial mode: original single-frame background subtraction
            small = cv2.resize(gray_frame, (small_w, small_h),
                               interpolation=cv2.INTER_AREA)

            # Background subtraction using large-kernel median
            bg_kernel = 51  # Must be odd; larger kernel removes sky gradients better
            bg = cv2.medianBlur(small, bg_kernel)
            signal = small.astype(np.float32) - bg.astype(np.float32)
            signal = np.clip(signal, 0, None)

            # MAD noise estimation (single value for entire frame)
            flat = signal.ravel()
            median_val = np.median(flat)
            mad = np.median(np.abs(flat - median_val))
            noise_std = mad * 1.4826
            if noise_std < 0.5:
                noise_std = 0.5
            noise_map_small = None
            use_noise_map = False

        # --- Multi-scale directional matched filter bank ---
        # Uses pre-computed kernel bank from __init__ (24 angles x 2 lengths).
        best_snr = np.zeros_like(signal)
        best_angle = np.zeros_like(signal)

        # GPU-accelerated path: upload signal once, run all filters on GPU
        if self._use_gpu_filter and self._gpu_filters is not None:
            try:
                gpu_signal = cv2.cuda_GpuMat()
                gpu_signal.upload(signal)
                for angle_deg, filt, noise_factor in self._gpu_filters:
                    gpu_response = filt.apply(gpu_signal)
                    response = gpu_response.download()

                    if use_noise_map:
                        snr = response / (noise_map_small * noise_factor + 1e-10)
                    else:
                        snr = response / (noise_std * noise_factor + 1e-10)

                    better = snr > best_snr
                    best_snr[better] = snr[better]
                    best_angle[better] = angle_deg
            except Exception:
                # CUDA failed at runtime — fall back to CPU for filter2D only
                self._use_gpu_filter = False
                self._gpu_filters = None
                best_snr = np.zeros_like(signal)
                best_angle = np.zeros_like(signal)
                # Re-run on CPU (fall through below)

        # CPU path (default or CUDA fallback)
        if not self._use_gpu or self._gpu_filters is None:
            for angle_deg, kernel, noise_factor in self._mf_kernel_bank:
                response = cv2.filter2D(signal, cv2.CV_32F, kernel)

                if use_noise_map:
                    snr = response / (noise_map_small * noise_factor + 1e-10)
                else:
                    snr = response / (noise_std * noise_factor + 1e-10)

                better = snr > best_snr
                best_snr[better] = snr[better]
                best_angle[better] = angle_deg

        # --- SNR-based thresholding ---
        snr_threshold = self.params.get('mf_snr_threshold', 2.5)
        significant = (best_snr > snr_threshold).astype(np.uint8) * 255

        # Light morphological cleanup
        cleanup_kernel = np.ones((3, 3), np.uint8)
        significant = cv2.dilate(significant, cleanup_kernel, iterations=1)
        significant = cv2.erode(significant, cleanup_kernel, iterations=1)

        # --- Extract line segments via Hough on the thresholded map ---
        min_len = max(15, int(self.params['min_line_length'] * scale * 0.6))
        max_gap = int(self.params['max_line_gap'] * scale)
        hough_thresh = max(15, int(self.params['hough_threshold'] * 0.6))

        lines = cv2.HoughLinesP(
            significant,
            rho=1,
            theta=np.pi / 180,
            threshold=hough_thresh,
            minLineLength=min_len,
            maxLineGap=max_gap
        )

        if lines is None:
            return None

        # --- Scale back to original resolution ---
        inv_scale = 1.0 / scale
        scaled_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            scaled_lines.append(np.array([[
                int(x1 * inv_scale), int(y1 * inv_scale),
                int(x2 * inv_scale), int(y2 * inv_scale)
            ]]))

        # --- Filter duplicates vs existing primary detections ---
        if existing_lines is not None and len(existing_lines) > 0:
            filtered = []
            for new_line in scaled_lines:
                nx1, ny1, nx2, ny2 = new_line[0]
                ncx = (nx1 + nx2) / 2.0
                ncy = (ny1 + ny2) / 2.0
                n_angle = math.degrees(math.atan2(abs(ny2 - ny1), abs(nx2 - nx1)))

                is_dup = False
                for existing in existing_lines:
                    ex1, ey1, ex2, ey2 = existing[0]
                    ecx = (ex1 + ex2) / 2.0
                    ecy = (ey1 + ey2) / 2.0
                    e_angle = math.degrees(math.atan2(abs(ey2 - ey1), abs(ex2 - ex1)))

                    dist = math.sqrt((ncx - ecx)**2 + (ncy - ecy)**2)
                    angle_diff = abs(n_angle - e_angle)
                    angle_diff = min(angle_diff, 180 - angle_diff)

                    if dist < 80 and angle_diff < 15:
                        is_dup = True
                        break

                if not is_dup:
                    filtered.append(new_line)

            if not filtered:
                return None
            scaled_lines = filtered

        if not scaled_lines:
            return None
        # Cap supplementary lines to avoid classify_trail bottleneck
        result = np.array(scaled_lines)
        max_supp = 50
        if len(result) > max_supp:
            lengths = np.sqrt((result[:, 0, 2] - result[:, 0, 0]).astype(np.float64)**2 +
                              (result[:, 0, 3] - result[:, 0, 1]).astype(np.float64)**2)
            top_idx = np.argpartition(lengths, -max_supp)[-max_supp:]
            result = result[top_idx]
        return result

    def _compute_trail_snr(self, gray_frame, line):
        """Compute signal-to-noise ratio of a candidate trail vs local background.

        Measures brightness along the trail and in flanking regions
        perpendicular to the trail direction.

            SNR = (trail_mean − background_mean) / background_std

        A positive SNR indicates the trail is brighter than its surroundings.
        SNR >= 2 is a marginal detection; SNR >= 3 is reliable.

        Args:
            gray_frame: Grayscale frame
            line: Line in HoughLinesP format [[x1, y1, x2, y2]]

        Returns:
            SNR value (float).  Returns 0.0 if the trail is too short or
            has insufficient samples.
        """
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if length < 10:
            return 0.0

        n_samples = max(20, min(200, int(length / 3)))

        h, w = gray_frame.shape
        dx_n = (x2 - x1) / length
        dy_n = (y2 - y1) / length
        perp_dx = -dy_n  # Perpendicular direction
        perp_dy = dx_n
        flank_dist = 8  # Pixels from trail center to flank sample

        # Vectorized sampling: compute all coordinates at once
        ts = np.linspace(0, 1, n_samples)
        cx_all = x1 + ts * (x2 - x1)
        cy_all = y1 + ts * (y2 - y1)

        # Trail pixels
        tx = np.round(cx_all).astype(int)
        ty = np.round(cy_all).astype(int)
        t_valid = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)
        trail_values = gray_frame[ty[t_valid], tx[t_valid]].astype(np.float64)

        # Flank pixels (both sides)
        fx_neg = np.round(cx_all - flank_dist * perp_dx).astype(int)
        fy_neg = np.round(cy_all - flank_dist * perp_dy).astype(int)
        fx_pos = np.round(cx_all + flank_dist * perp_dx).astype(int)
        fy_pos = np.round(cy_all + flank_dist * perp_dy).astype(int)
        fn_valid = (fx_neg >= 0) & (fx_neg < w) & (fy_neg >= 0) & (fy_neg < h)
        fp_valid = (fx_pos >= 0) & (fx_pos < w) & (fy_pos >= 0) & (fy_pos < h)
        flank_values = np.concatenate([
            gray_frame[fy_neg[fn_valid], fx_neg[fn_valid]].astype(np.float64),
            gray_frame[fy_pos[fp_valid], fx_pos[fp_valid]].astype(np.float64),
        ])

        if len(trail_values) < 5 or len(flank_values) < 5:
            return 0.0

        trail_mean = np.mean(trail_values)
        bg_mean = np.mean(flank_values)
        bg_std = np.std(flank_values)

        if bg_std < 0.5:
            bg_std = 0.5

        return (trail_mean - bg_mean) / bg_std

    # ------------------------------------------------------------------
    #  Post-detection enrichment: Streak photometry, curvature, velocity
    # ------------------------------------------------------------------

    def _analyze_streak_photometry(self, gray_frame, line):
        """Extract the brightness profile along a trail and classify its lightcurve.

        With 13-second exposures, the brightness along a satellite trail encodes
        the object's lightcurve during the pass:

        - **Steady brightness** → stabilised satellite or debris
        - **Periodic sinusoidal modulation** → tumbling object
          (period_pixels / trail_length × exposure_time = tumble period)
        - **Discrete bright/dark segments** → specular flash (Iridium-like)
        - **Sharp ~1 Hz pulses** → airplane strobe lights

        The analysis uses FFT to find the dominant periodicity and its strength
        relative to the DC component (the mean brightness).

        Args:
            gray_frame: Grayscale frame
            line: Line in HoughLinesP format [[x1, y1, x2, y2]]

        Returns:
            Dict with photometry results, or None if trail too short::

                {
                    'profile': np.array,       # raw brightness samples
                    'profile_detrended': np.array,  # mean-subtracted
                    'mean_brightness': float,
                    'std_brightness': float,
                    'dominant_period_px': float or None,  # pixels per cycle
                    'periodicity_strength': float,  # 0-1, ratio of peak FFT power to total
                    'num_cycles': float,       # how many full cycles fit in the trail
                    'classification': str,     # 'steady', 'tumbling', 'flashing', 'strobing'
                }
        """
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 30:
            return None

        # Dense sampling: every 2 pixels along the trail
        n_samples = max(20, int(length / 2))
        n_samples = min(n_samples, 500)

        h, w = gray_frame.shape
        samples = []
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0
            px = int(round(x1 + t * (x2 - x1)))
            py = int(round(y1 + t * (y2 - y1)))
            if 0 <= px < w and 0 <= py < h:
                # Average 3×3 neighbourhood to reduce single-pixel noise
                y_lo = max(0, py - 1)
                y_hi = min(h, py + 2)
                x_lo = max(0, px - 1)
                x_hi = min(w, px + 2)
                samples.append(float(np.mean(gray_frame[y_lo:y_hi, x_lo:x_hi])))

        if len(samples) < 20:
            return None

        profile = np.array(samples, dtype=np.float64)
        mean_br = np.mean(profile)
        std_br = np.std(profile)

        # Detrend: subtract the mean so FFT focuses on oscillation
        detrended = profile - mean_br

        # FFT analysis for periodicity
        n = len(detrended)
        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2
        # Ignore DC (index 0) and very low frequencies (index 1)
        if len(power) > 2:
            power_no_dc = power[2:]
            total_power = np.sum(power_no_dc)
            if total_power > 0:
                peak_idx = np.argmax(power_no_dc) + 2  # offset by 2 for skipped bins
                peak_power = power[peak_idx]
                periodicity_strength = peak_power / (total_power + 1e-10)
                # Period in pixels: n_samples / frequency_index
                # Frequency index maps to (peak_idx / n) cycles per sample
                # Period in samples = n / peak_idx
                period_samples = n / peak_idx if peak_idx > 0 else None
                # Convert from samples to pixels
                px_per_sample = length / n_samples
                dominant_period_px = period_samples * px_per_sample if period_samples else None
                num_cycles = length / dominant_period_px if dominant_period_px and dominant_period_px > 0 else 0
            else:
                periodicity_strength = 0.0
                dominant_period_px = None
                num_cycles = 0
        else:
            periodicity_strength = 0.0
            dominant_period_px = None
            num_cycles = 0

        # Classify the lightcurve pattern
        variation_coeff = std_br / (mean_br + 1e-5)

        if variation_coeff < 0.10:
            classification = 'steady'
        elif periodicity_strength > 0.25 and num_cycles >= 2:
            # Check if the modulation is smooth (tumbling) or sharp (strobing)
            # Sharp strobe pulses produce many harmonics; smooth sinusoidal
            # concentrates power in one frequency.
            if periodicity_strength > 0.5:
                classification = 'tumbling'   # strong single frequency → sinusoidal
            else:
                classification = 'strobing'   # weaker peak → sharp pulses with harmonics
        elif variation_coeff > 0.30 and periodicity_strength < 0.15:
            classification = 'flashing'   # irregular brightness changes (specular flash)
        elif periodicity_strength > 0.15 and num_cycles >= 1.5:
            classification = 'tumbling'
        else:
            classification = 'steady'

        return {
            'profile': profile,
            'profile_detrended': detrended,
            'mean_brightness': mean_br,
            'std_brightness': std_br,
            'dominant_period_px': dominant_period_px,
            'periodicity_strength': periodicity_strength,
            'num_cycles': num_cycles,
            'classification': classification,
        }

    def _fit_trail_curvature(self, gray_frame, line, diff_image=None):
        """Fit a quadratic to the trail and measure curvature.

        In a long exposure, LEO satellites trace a measurable arc due to
        Earth's curvature and orbital mechanics.  Fitting y = ax² + bx + c
        (in the trail's local coordinate system) gives a curvature value |a|
        that is proportional to 1/altitude.

        - High curvature → low orbit (LEO)
        - Low curvature  → high orbit (MEO/GEO) or airplane (straight-line
          motion at the scale of the frame)

        The fit is performed on the brightness-weighted ridge of the trail,
        not just the endpoint geometry, so sub-pixel accuracy is achievable.

        Args:
            gray_frame: Grayscale frame (or diff_image if available)
            line: Line in HoughLinesP format [[x1, y1, x2, y2]]
            diff_image: Optional background-subtracted image (preferred if
                available, as stars are removed)

        Returns:
            Dict with curvature results, or None if trail too short::

                {
                    'curvature': float,       # |a| coefficient of x² term
                    'residual_rms': float,    # RMS of fit residuals (pixels)
                    'is_curved': bool,        # True if curvature is significant
                    'fit_coeffs': (a, b, c),  # quadratic coefficients
                }
        """
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if length < 60:
            return None

        img = diff_image if diff_image is not None else gray_frame.astype(np.float32)
        h, w = img.shape[:2]

        # Sample perpendicular brightness profiles to find the ridge
        # (sub-pixel trail center at each position along the line)
        n_slices = max(20, int(length / 4))
        n_slices = min(n_slices, 200)

        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        perp_dx = -dy
        perp_dy = dx
        perp_half = 6  # pixels to each side of the trail

        along_coords = []   # t parameter (0 to length)
        perp_offsets = []    # sub-pixel perpendicular offset from nominal line

        for i in range(n_slices):
            t = i / (n_slices - 1) if n_slices > 1 else 0
            cx = x1 + t * (x2 - x1)
            cy = y1 + t * (y2 - y1)

            # Sample perpendicular profile
            profile = []
            offsets = []
            for j in range(-perp_half, perp_half + 1):
                sx = int(round(cx + j * perp_dx))
                sy = int(round(cy + j * perp_dy))
                if 0 <= sx < w and 0 <= sy < h:
                    profile.append(img[sy, sx])
                    offsets.append(j)

            if len(profile) < 5:
                continue

            profile = np.array(profile, dtype=np.float64)
            offsets = np.array(offsets, dtype=np.float64)

            # Brightness-weighted centroid gives sub-pixel ridge position
            profile_shifted = profile - np.min(profile)
            total = np.sum(profile_shifted)
            if total > 0:
                centroid = np.sum(offsets * profile_shifted) / total
            else:
                centroid = 0.0

            along_coords.append(t * length)
            perp_offsets.append(centroid)

        if len(along_coords) < 10:
            return None

        along = np.array(along_coords)
        perp = np.array(perp_offsets)

        # Fit quadratic: perp_offset = a*along² + b*along + c
        # where a is the curvature coefficient
        coeffs = np.polyfit(along, perp, 2)
        a, b, c = coeffs

        # Compute residuals
        fitted = np.polyval(coeffs, along)
        residuals = perp - fitted
        residual_rms = np.sqrt(np.mean(residuals ** 2))

        curvature = abs(a)

        # Is the curvature statistically significant?
        # Compare quadratic fit residuals to linear fit residuals
        linear_coeffs = np.polyfit(along, perp, 1)
        linear_fitted = np.polyval(linear_coeffs, along)
        linear_residual_rms = np.sqrt(np.mean((perp - linear_fitted) ** 2))

        # Curvature is significant if the quadratic fit reduces residuals
        # by at least 20% compared to the linear fit, and the curvature
        # causes at least 0.5 pixel deflection over the trail length
        max_deflection = curvature * (length ** 2) / 4  # peak deflection at midpoint
        is_curved = (
            linear_residual_rms > 0.3 and
            residual_rms < linear_residual_rms * 0.8 and
            max_deflection > 0.5
        )

        return {
            'curvature': curvature,
            'residual_rms': residual_rms,
            'is_curved': is_curved,
            'fit_coeffs': (a, b, c),
        }

    @staticmethod
    def _estimate_angular_velocity(trail_length_px, frame_width_px,
                                   exposure_time=13.0, fov_degrees=None):
        """Estimate angular velocity from trail length and exposure time.

        If the field of view (FOV) is known, returns degrees/second.
        Otherwise returns pixels/second (still useful for relative comparison
        and orbit-class discrimination).

        Typical angular velocities (degrees/second):

        =====================  ==================  ========================
        Object                 Angular velocity    Trail in 13 s (60° FOV)
        =====================  ==================  ========================
        LEO satellite          0.3 – 1.5 °/s       200 – 1000+ px
        MEO satellite          0.01 – 0.1 °/s      6 – 65 px
        GEO satellite          ~0 °/s               ~0 px (point)
        Airplane (high alt)    0.05 – 0.5 °/s       32 – 320 px
        Meteor                 5 – 70 °/s            full frame+
        =====================  ==================  ========================

        Args:
            trail_length_px: Trail length in pixels
            frame_width_px: Frame width in pixels (for plate scale)
            exposure_time: Exposure time in seconds (default: 13.0)
            fov_degrees: Horizontal field of view in degrees (optional).
                If None, returns velocity in px/s only.

        Returns:
            Dict with velocity estimates::

                {
                    'px_per_sec': float,           # pixels / second
                    'deg_per_sec': float or None,  # degrees / second (if FOV known)
                    'orbit_class': str,            # 'LEO', 'MEO', 'GEO', 'meteor', 'unknown'
                }
        """
        if exposure_time <= 0:
            exposure_time = 13.0

        px_per_sec = trail_length_px / exposure_time

        deg_per_sec = None
        if fov_degrees is not None and frame_width_px > 0:
            plate_scale = fov_degrees / frame_width_px  # degrees per pixel
            deg_per_sec = px_per_sec * plate_scale

        # Classify orbit based on angular velocity
        # Use pixel velocity normalised to 1080p-equivalent for classification
        # (assumes ~60° FOV for a typical wide-angle setup)
        norm_px_per_sec = px_per_sec * (1920.0 / max(1, frame_width_px))

        if norm_px_per_sec > 250:
            orbit_class = 'meteor'      # extremely fast
        elif norm_px_per_sec > 15:
            orbit_class = 'LEO'         # fast movers
        elif norm_px_per_sec > 2:
            orbit_class = 'MEO'         # slow but visible trail
        elif norm_px_per_sec > 0.3:
            orbit_class = 'GEO'         # nearly stationary
        else:
            orbit_class = 'unknown'

        return {
            'px_per_sec': px_per_sec,
            'deg_per_sec': deg_per_sec,
            'orbit_class': orbit_class,
        }

    def detect_point_features(self, line, gray_frame, return_debug_info=False):
        """
        Detect point-like features (bright spots) along a trail using spatial analysis.
        Returns the number of distinct bright points detected.

        Args:
            line: Detected line coordinates
            gray_frame: Grayscale frame
            return_debug_info: If True, return detailed debug information

        Returns:
            If return_debug_info is False: number of peaks
            If return_debug_info is True: (num_peaks, debug_dict)
        """
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if length < 30:
            if return_debug_info:
                return 0, {'sample_points': [], 'peak_indices': [], 'brightness_samples': []}
            return 0

        # Sample points along the line
        num_samples = int(length / 5)  # Sample every 5 pixels
        num_samples = max(10, min(num_samples, 100))  # Between 10 and 100 samples

        brightness_samples = []
        sample_points = []  # Store (x, y) coordinates for debug visualization

        for i in range(num_samples):
            t = i / (num_samples - 1) if num_samples > 1 else 0
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))

            # Check bounds
            if 0 <= py < gray_frame.shape[0] and 0 <= px < gray_frame.shape[1]:
                # Get brightness in a small neighborhood (3x3) around the point
                y_min = max(0, py - 1)
                y_max = min(gray_frame.shape[0], py + 2)
                x_min = max(0, px - 1)
                x_max = min(gray_frame.shape[1], px + 2)

                neighborhood = gray_frame[y_min:y_max, x_min:x_max]
                if neighborhood.size > 0:
                    brightness_samples.append(np.max(neighborhood))
                    sample_points.append((px, py))

        if len(brightness_samples) < 10:
            if return_debug_info:
                return 0, {'sample_points': sample_points, 'peak_indices': [], 'brightness_samples': brightness_samples}
            return 0

        # Detect local maxima (bright points)
        brightness_array = np.array(brightness_samples)
        mean_brightness = np.mean(brightness_array)
        std_brightness = np.std(brightness_array)

        # A point is considered a "peak" if it's significantly brighter than the mean
        # Tightened to require more prominent peaks
        threshold = mean_brightness + std_brightness * 1.0
        peaks = brightness_array > threshold

        # Count distinct peaks.  Consecutive above-threshold samples are
        # merged into one peak group, and peak groups closer than
        # min_peak_separation samples apart are also merged — a single
        # star can produce nearby brightness bumps that should not count
        # as separate airplane navigation-light features.
        min_peak_separation = max(5, num_samples // 10)  # ≥10 % of trail apart

        num_peaks = 0
        in_peak = False
        peak_indices = []  # Store indices of peak starts for debug visualization
        last_peak_end = -min_peak_separation  # Allow the first peak unconditionally

        for i, is_peak in enumerate(peaks):
            if is_peak and not in_peak:
                if i - last_peak_end >= min_peak_separation:
                    num_peaks += 1
                    peak_indices.append(i)
                else:
                    # Too close to previous peak — merge (don't increment)
                    pass
                in_peak = True
            elif not is_peak:
                if in_peak:
                    last_peak_end = i  # Record where this peak group ended
                in_peak = False

        if return_debug_info:
            debug_info = {
                'sample_points': sample_points,
                'peak_indices': peak_indices,
                'brightness_samples': brightness_samples,
                'threshold': threshold,
                'mean_brightness': mean_brightness
            }
            return num_peaks, debug_info

        return num_peaks

    def classify_trail(self, line, gray_frame, color_frame, hsv_frame=None, reusable_mask=None, supplementary=False, ledger=None):
        """
        Classify a detected line as either a satellite, airplane, or anomalous trail.

        Key distinction:
        - Airplanes: DOTTED features - bright point-like lights along the trail (navigation lights)
                    Sometimes colorful dots (red, green, white). Can be any length including 180-300px.
        - Satellites: SMOOTH, consistent brightness along trail. No bright point features.
                     Dim, monochromatic, uniform appearance. Typically 180-300 pixels for 1920x1080.
        - Anomalous: Linear features that pass basic validity checks but match neither
                     satellite nor airplane patterns.  Captures meteors, tumbling debris,
                     drones, ISS with solar flare, and other residual phenomena (Bowker & Star).

        Args:
            line: Detected line from HoughLinesP
            gray_frame: Grayscale frame
            color_frame: BGR color frame
            hsv_frame: Pre-computed HSV frame (optional, for performance)
            reusable_mask: Pre-allocated mask array (optional, for performance)
            supplementary: If True, this candidate came from the matched-filter
                stage and has already passed an SNR gate.  Contrast thresholds
                are relaxed and an additional SNR-based detection path is enabled.
            ledger: Optional TranslationLedger for tracking rejection statistics.

        Returns:
            trail_type: 'satellite', 'airplane', 'anomalous', or None
            detection_info: dict with 'bbox' and metadata if trail detected, None otherwise.
                Keys: 'bbox' (x_min, y_min, x_max, y_max), 'angle' (degrees 0-180),
                'center' (x, y), 'length' (pixels), 'avg_brightness' (float),
                'max_brightness' (int), 'line' (original line endpoints),
                'epistemic_profile' (dict — detection provenance, margins, counterfactual)
        """
        x1, y1, x2, y2 = line[0]

        # Calculate line properties
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Angle in degrees (0-180 range, normalized so direction doesn't matter)
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        # Epistemic profile — tracks how this detection was constructed (Latour).
        # Every threshold test records its value, threshold, and margin.
        ep = {
            'detection_path': None,
            'criteria_met': [],
            'criteria_failed': [],
            'rejection_reason': None,
            'margin_analysis': {},
            'stage': 'supplementary' if supplementary else 'primary',
        }

        if length < self.params['min_line_length']:
            if ledger:
                ledger.record_rejection('too_short', line)
            return None, None

        # Reject full-frame-width artifacts early — real satellite trails
        # rarely span > 80% of the frame.  Lines this long are typically
        # edge artifacts from morphological operations or sky gradients.
        frame_w = color_frame.shape[1] if color_frame is not None else 1920
        if length > frame_w * 0.80:
            if ledger:
                ledger.record_rejection('full_frame', line)
            return None, None

        # Use reusable mask if provided, otherwise allocate new one
        if reusable_mask is not None:
            mask = reusable_mask
            mask.fill(0)  # Clear the mask
        else:
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        # Use narrower mask (3px) for supplementary dim trails to avoid
        # background contamination of brightness/smoothness measurements.
        # Wider mask (5px) for primary detections where trails are brighter.
        mask_thickness = 3 if supplementary else 5
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=mask_thickness)

        # Check brightness along the trail
        trail_pixels_gray = gray_frame[mask > 0]
        if len(trail_pixels_gray) == 0:
            return None, None

        # Require minimum number of pixels - ensures we're detecting actual trails
        if len(trail_pixels_gray) < 15:
            if ledger:
                ledger.record_rejection('too_few_pixels', line)
            return None, None

        avg_brightness = np.mean(trail_pixels_gray)
        max_brightness = np.max(trail_pixels_gray)
        brightness_std = np.std(trail_pixels_gray)

        # Too dark (likely noise) - trails should have some minimum brightness
        if avg_brightness < 5:
            if ledger:
                ledger.record_rejection('too_dark', line)
            return None, None

        # Check minimum contrast - trail should stand out from background
        # Use per-sensitivity threshold so dim satellite trails aren't rejected
        surround_sample_size = 50
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Get background brightness from area around the trail, excluding trail
        # pixels to avoid inflating the background estimate for dim trails
        bg_x_min = max(0, x_center - surround_sample_size)
        bg_y_min = max(0, y_center - surround_sample_size)
        bg_x_max = min(gray_frame.shape[1], x_center + surround_sample_size)
        bg_y_max = min(gray_frame.shape[0], y_center + surround_sample_size)

        background_region = gray_frame[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
        contrast_ratio = None
        if background_region.size > 0:
            # Exclude trail pixels from background estimate for cleaner contrast
            bg_mask_crop = mask[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
            bg_only = background_region[bg_mask_crop == 0]
            if len(bg_only) > 20:
                background_brightness = np.median(bg_only)
            else:
                background_brightness = np.median(background_region)
            contrast_ratio = avg_brightness / (background_brightness + 1e-5)
            min_contrast = self.params.get('satellite_contrast_min', 1.08)
            # Supplementary candidates already passed the matched-filter SNR
            # gate, so the contrast criterion can be relaxed to avoid rejecting
            # dim trails that the primary pipeline would never have seen.
            if supplementary:
                min_contrast = max(1.03, min_contrast * 0.7)
            if contrast_ratio < min_contrast:
                if ledger:
                    ledger.record_rejection('low_contrast', line)
                return None, None

        # Calculate bounding box
        padding = 10
        x_min = max(0, min(x1, x2) - padding)
        y_min = max(0, min(y1, y2) - padding)
        x_max = min(color_frame.shape[1], max(x1, x2) + padding)
        y_max = min(color_frame.shape[0], max(y1, y2) + padding)

        width = x_max - x_min
        height = y_max - y_min

        if width == 0 or height == 0:
            return None, None

        # Check aspect ratio (trails are long and thin)
        aspect_ratio = max(width, height) / min(width, height)

        if not self.skip_aspect_ratio_check and aspect_ratio < self.params['min_aspect_ratio']:
            if ledger:
                ledger.record_rejection('aspect_ratio', line)
            return None, None

        # CLOUD AND FALSE POSITIVE FILTERING
        # Reject false positives from clouds, buildings, power lines, etc.

        # 1. Check surrounding texture (clouds have high local variance)
        surround_padding = 25
        surround_x_min = max(0, x_min - surround_padding)
        surround_y_min = max(0, y_min - surround_padding)
        surround_x_max = min(gray_frame.shape[1], x_max + surround_padding)
        surround_y_max = min(gray_frame.shape[0], y_max + surround_padding)

        surrounding_region = gray_frame[surround_y_min:surround_y_max, surround_x_min:surround_x_max]

        if surrounding_region.size > 0:
            # Calculate texture complexity using local standard deviation
            surrounding_std = np.std(surrounding_region)
            surrounding_mean = np.mean(surrounding_region)

            # Clouds have high texture variation and moderate brightness
            # Tightened thresholds to be more aggressive
            is_likely_cloud = (surrounding_std > 25 and surrounding_mean > 35)

            if is_likely_cloud:
                if ledger:
                    ledger.record_rejection('cloud_texture', line)
                return None, None

            # Reject very bright uniform areas (likely daytime sky or lit structures)
            if surrounding_mean > 80:
                if ledger:
                    ledger.record_rejection('too_bright', line)
                return None, None

        # 2. Check for gradient uniformity along the line
        # Real trails should have relatively consistent brightness
        if len(trail_pixels_gray) > 10:
            # Split trail into segments and check variance
            segment_size = len(trail_pixels_gray) // 3
            if segment_size > 0:
                seg1 = trail_pixels_gray[:segment_size]
                seg2 = trail_pixels_gray[segment_size:2*segment_size]
                seg3 = trail_pixels_gray[2*segment_size:]

                seg_means = [np.mean(seg1), np.mean(seg2), np.mean(seg3)]
                segment_variation = np.std(seg_means)

                # If segments have very different brightnesses, likely not a trail
                # Tightened from 25 to 20 for stricter filtering
                if segment_variation > 20:
                    if ledger:
                        ledger.record_rejection('segment_variation', line)
                    return None, None

        # 3. Check maximum brightness - extremely bright lines are likely not trails
        # Real airplane/satellite trails in night sky shouldn't be extremely bright
        if max_brightness > 240:
            if ledger:
                ledger.record_rejection('too_bright', line)
            return None, None

        # Analyze color information for airplane detection
        trail_pixels_color = color_frame[mask > 0]

        # Use pre-computed HSV frame if available, otherwise convert (for backward compatibility)
        if hsv_frame is not None:
            hsv_region = hsv_frame
        else:
            hsv_region = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        trail_pixels_hsv = hsv_region[mask > 0]

        avg_saturation = np.mean(trail_pixels_hsv[:, 1]) if len(trail_pixels_hsv) > 0 else 0
        max_saturation = np.max(trail_pixels_hsv[:, 1]) if len(trail_pixels_hsv) > 0 else 0

        # Calculate color variation (airplanes show more color due to navigation lights)
        b_channel = trail_pixels_color[:, 0]
        g_channel = trail_pixels_color[:, 1]
        r_channel = trail_pixels_color[:, 2]

        color_variation = np.std([np.mean(b_channel), np.mean(g_channel), np.mean(r_channel)])

        # ENHANCED DOTTED FEATURE DETECTION (primary airplane identifier)
        # Check for bright point-like features along the trail (characteristic of airplanes)
        brightness_variation = brightness_std / (avg_brightness + 1e-5)

        # Detect local maxima (bright spots) along the trail
        # Find brightest spots using partition (faster than full sort)
        if len(trail_pixels_gray) > 20:
            top_10_percent_count = max(1, len(trail_pixels_gray) // 10)
            # np.partition is O(n) vs O(n log n) for sort - only need top k elements
            partition_idx = len(trail_pixels_gray) - top_10_percent_count
            partitioned = np.partition(trail_pixels_gray, partition_idx)
            top_10_percent_mean = np.mean(partitioned[partition_idx:])

            # If the brightest 10% of pixels are significantly brighter than average, it's dotted
            brightness_peak_ratio = top_10_percent_mean / (avg_brightness + 1e-5)
            has_bright_spots = brightness_peak_ratio > 1.5  # Bright spots significantly brighter
        else:
            has_bright_spots = False
            brightness_peak_ratio = 1.0

        # Check for high brightness variance (indicates non-uniform, dotted pattern)
        has_high_variance = brightness_variation > 0.30  # Balanced threshold

        # STAR FALSE-POSITIVE SUPPRESSION
        # A single bright star on an otherwise dim trail can inflate
        # brightness_peak_ratio and brightness_variation, mimicking a
        # dotted airplane pattern.  Real airplane navigation-light dots
        # are distributed along the trail; a star is localised.  Sample
        # brightness along the line and check what fraction of the trail
        # length the bright pixels span.
        if has_bright_spots or has_high_variance:
            n_spread = max(20, min(100, int(length / 5)))
            ts = np.linspace(0, 1, n_spread)
            spx = (x1 + ts * (x2 - x1)).astype(int)
            spy = (y1 + ts * (y2 - y1)).astype(int)
            valid = (spy >= 0) & (spy < gray_frame.shape[0]) & (spx >= 0) & (spx < gray_frame.shape[1])
            spread_arr = gray_frame[spy[valid], spx[valid]]
            if len(spread_arr) >= 10:
                spread_mean = np.mean(spread_arr)
                spread_std = np.std(spread_arr)
                bright_mask = spread_arr > (spread_mean + spread_std)
                bright_fraction = np.sum(bright_mask) / len(spread_arr)
                # Real airplane dots span >20 % of the trail length;
                # a single star typically illuminates <15 %.
                if bright_fraction < 0.15:
                    has_bright_spots = False
                    has_high_variance = False

        # SPATIAL ANALYSIS: Detect distinct point-like features along the trail
        num_point_features = self.detect_point_features(line, gray_frame)
        has_multiple_points = num_point_features >= 2  # At least 2 distinct bright points

        bbox = (x_min, y_min, x_max, y_max)

        # Build detection metadata (shared by airplane, satellite, and anomalous results)
        def _make_detection_info(detection_path=None, extra_criteria_met=None, extra_criteria_failed=None):
            ep['detection_path'] = detection_path
            if extra_criteria_met:
                ep['criteria_met'].extend(extra_criteria_met)
            if extra_criteria_failed:
                ep['criteria_failed'].extend(extra_criteria_failed)
            # Margin analysis — how close was the decision to going the other way?
            if contrast_ratio is not None:
                ep['margin_analysis']['contrast_ratio'] = {
                    'value': round(float(contrast_ratio), 4),
                    'threshold': round(float(min_contrast), 4),
                    'margin': round(float(contrast_ratio - min_contrast), 4),
                }
            ep['margin_analysis']['brightness_variation'] = {
                'value': round(float(brightness_variation), 4),
                'threshold': 0.40,
                'margin': round(float(0.40 - brightness_variation), 4),
            }
            return {
                'bbox': bbox,
                'angle': angle,
                'center': center,
                'length': length,
                'avg_brightness': float(avg_brightness),
                'max_brightness': int(max_brightness),
                'line': (x1, y1, x2, y2),
                'contrast_ratio': float(contrast_ratio) if contrast_ratio is not None else None,
                'brightness_std': float(brightness_std),
                'avg_saturation': float(avg_saturation),
                'has_dotted_pattern': bool(has_bright_spots or has_high_variance or has_multiple_points),
                'epistemic_profile': dict(ep),
            }

        # AIRPLANE DETECTION CRITERIA (check first - dotted features are distinctive)
        # PRIMARY: Dotted/point-like bright features (most important!)
        # SECONDARY: Colorful features, overall brightness
        # Length can be similar to satellites (180-300px) or longer
        is_bright = avg_brightness > self.params['airplane_brightness_min'] or max_brightness > 170
        is_colorful = avg_saturation > self.params['airplane_saturation_min'] or max_saturation > 30
        has_color_variation = color_variation > 5
        has_dotted_pattern = has_bright_spots or has_high_variance or has_multiple_points

        # Strong airplane indicators - tightened thresholds
        has_strong_dots = has_bright_spots and brightness_peak_ratio > 1.8
        has_colored_dots = is_colorful and has_dotted_pattern and max_brightness > 90
        has_moderate_dots = has_bright_spots and brightness_peak_ratio > 1.5 and max_brightness > 100
        has_spatial_dots = has_multiple_points and is_bright and max_brightness > 100

        # If clear dotted pattern detected, it's definitely an airplane
        if has_strong_dots or has_colored_dots or has_moderate_dots or has_spatial_dots:
            path = 'airplane_strong_dots'
            if has_strong_dots: path = 'airplane_strong_dots'
            elif has_colored_dots: path = 'airplane_colored_dots'
            elif has_moderate_dots: path = 'airplane_moderate_dots'
            else: path = 'airplane_spatial_dots'
            if ledger:
                ledger.record_classification('airplane')
            return 'airplane', _make_detection_info(detection_path=path,
                extra_criteria_met=['has_dotted_pattern', 'is_bright'])

        # If multiple distinct point features detected (navigation lights pattern)
        # Require higher brightness to avoid false positives
        if has_multiple_points and max_brightness > 120 and is_bright:
            if ledger:
                ledger.record_classification('airplane')
            return 'airplane', _make_detection_info(detection_path='airplane_multiple_points',
                extra_criteria_met=['has_multiple_points', 'is_bright'])

        # Calculate airplane score - require more evidence
        airplane_score = sum([is_bright, is_colorful, has_color_variation, has_dotted_pattern])

        # Require dotted pattern AND at least 2 other characteristics
        if has_dotted_pattern and airplane_score >= 3:
            if ledger:
                ledger.record_classification('airplane')
            return 'airplane', _make_detection_info(detection_path='airplane_multi_evidence',
                extra_criteria_met=[k for k, v in [('is_bright', is_bright), ('is_colorful', is_colorful),
                    ('has_color_variation', has_color_variation)] if v])

        # Very strong dotted pattern with high brightness
        if has_dotted_pattern and brightness_peak_ratio > 2.0 and max_brightness > 120:
            if ledger:
                ledger.record_classification('airplane')
            return 'airplane', _make_detection_info(detection_path='airplane_very_strong_dots',
                extra_criteria_met=['has_dotted_pattern', 'very_high_peak_ratio'])

        # SATELLITE DETECTION CRITERIA
        # Satellites have SMOOTH, consistent brightness (no dotted features)
        # They are dim, monochromatic, and can range from short segments to
        # very long trails spanning much of the frame.
        is_dim = avg_brightness < self.params['airplane_brightness_min']
        is_monochrome = avg_saturation < self.params['airplane_saturation_min']

        # Smoothness check: use adaptive threshold for very dim trails.
        # When avg_brightness is very low (e.g. 8), even small noise in pixel
        # values causes brightness_std / avg to spike, falsely failing the
        # smoothness test. Use absolute std as a fallback for dim trails.
        smooth_threshold = 0.40
        is_smooth_relative = brightness_variation < smooth_threshold and not has_bright_spots
        is_smooth_absolute = brightness_std < 8.0 and not has_bright_spots  # Low absolute variation
        is_smooth = is_smooth_relative or (is_dim and is_smooth_absolute)

        is_satellite_length = self.params['satellite_min_length'] <= length <= self.params['satellite_max_length']

        # Check if trail has good contrast with background (useful for dim trails)
        has_contrast = contrast_ratio is not None and contrast_ratio >= min_contrast

        satellite_score = sum([is_dim, is_monochrome, is_smooth, is_satellite_length])

        # --- Primary paths (strongest confidence) ---

        sat_criteria_met = [k for k, v in [
            ('is_dim', is_dim), ('is_monochrome', is_monochrome),
            ('is_smooth', is_smooth), ('is_satellite_length', is_satellite_length),
            ('has_contrast', has_contrast)] if v]
        sat_criteria_failed = [k for k, v in [
            ('is_dim', is_dim), ('is_monochrome', is_monochrome),
            ('is_smooth', is_smooth), ('is_satellite_length', is_satellite_length)] if not v]

        # All 4 characteristics met
        if satellite_score >= 4 and not has_dotted_pattern:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_primary_4of4',
                extra_criteria_met=sat_criteria_met, extra_criteria_failed=sat_criteria_failed)

        # 3 characteristics including both smoothness and length
        if satellite_score >= 3 and is_smooth and is_satellite_length and not has_dotted_pattern:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_primary_3of4',
                extra_criteria_met=sat_criteria_met, extra_criteria_failed=sat_criteria_failed)

        # Very dim, smooth trails in correct length range
        if is_smooth and avg_brightness <= self.params['brightness_threshold'] * 1.5 and is_satellite_length and not has_dotted_pattern:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_very_dim',
                extra_criteria_met=['is_smooth', 'very_dim', 'is_satellite_length'],
                extra_criteria_failed=sat_criteria_failed)

        # --- Extended paths for dim/long trails that miss primary criteria ---
        # These paths allow slightly beyond satellite_max_length (1.3×) but
        # NOT unlimited — full-frame artifacts are already rejected above.
        extended_max = self.params['satellite_max_length'] * 1.3

        # Long smooth dim trail outside the "typical" length range but clearly
        # not an airplane: no dotted pattern, dim, monochrome, smooth
        if is_smooth and is_dim and is_monochrome and not has_dotted_pattern and length >= self.params['satellite_min_length'] and length <= extended_max:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_extended_dim_smooth_mono',
                extra_criteria_met=sat_criteria_met, extra_criteria_failed=sat_criteria_failed)

        # Dim smooth trail with confirmed background contrast — even if
        # length or monochrome criteria aren't perfectly met
        if is_smooth and is_dim and has_contrast and not has_dotted_pattern and length >= self.params['satellite_min_length'] and length <= extended_max:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_extended_dim_smooth_contrast',
                extra_criteria_met=sat_criteria_met, extra_criteria_failed=sat_criteria_failed)

        # Very dim trail (below brightness_threshold) that is smooth and long
        # enough — relaxed monochrome requirement since very dim trails have
        # negligible color information anyway
        if is_smooth and avg_brightness <= self.params['brightness_threshold'] and not has_dotted_pattern and length >= self.params['satellite_min_length'] and length <= extended_max:
            if ledger:
                ledger.record_classification('satellite')
            return 'satellite', _make_detection_info(
                detection_path='satellite_extended_very_dim_smooth',
                extra_criteria_met=['is_smooth', 'very_dim'],
                extra_criteria_failed=sat_criteria_failed)

        # --- SNR-based path for matched-filter candidates ---
        # Trails found by the supplementary matched filter have already
        # passed a global SNR gate, but may fail the absolute-brightness or
        # contrast thresholds used above.  Compute a per-trail SNR using
        # perpendicular flank sampling — a statistically rigorous measure
        # that is independent of absolute brightness.  This rescues very dim
        # trails that are clearly above the local noise floor.
        if supplementary and not has_dotted_pattern and length >= self.params['satellite_min_length'] and length <= extended_max:
            trail_snr = self._compute_trail_snr(gray_frame, line)
            if trail_snr >= 2.5 and is_smooth:
                if ledger:
                    ledger.record_classification('satellite')
                return 'satellite', _make_detection_info(
                    detection_path='satellite_snr_based',
                    extra_criteria_met=['is_smooth', f'snr={trail_snr:.1f}'],
                    extra_criteria_failed=sat_criteria_failed)

        # --- Anomalous category (Bowker & Star — the residual) ---
        # The trail passed basic validity checks (length, contrast, aspect ratio,
        # not a cloud/texture artifact) but matches neither satellite nor airplane
        # patterns.  Instead of silently discarding it, classify as "anomalous" —
        # making the residual category visible.  These are often the most
        # interesting objects: meteors, tumbling debris, drones, ISS with solar
        # panel flare, or atmospheric phenomena the binary classification misses.
        if length >= self.params['satellite_min_length'] * 0.7:
            if ledger:
                ledger.record_classification('anomalous')
            ep['rejection_reason'] = 'no_satellite_or_airplane_match'
            return 'anomalous', _make_detection_info(
                detection_path='anomalous_residual',
                extra_criteria_met=[k for k, v in [
                    ('is_dim', is_dim), ('is_smooth', is_smooth),
                    ('has_dotted_pattern', has_dotted_pattern),
                    ('is_bright', is_bright)] if v],
                extra_criteria_failed=[k for k, v in [
                    ('is_dim', is_dim), ('is_smooth', is_smooth),
                    ('satellite_length', is_satellite_length)] if not v])

        if ledger:
            ledger.record_rejection('unclassifiable', line)
        return None, None
    
    def merge_overlapping_boxes(self, boxes, overlap_threshold=0.3):
        """Merge overlapping bounding boxes.

        Args:
            boxes: List of (x_min, y_min, x_max, y_max) tuples
            overlap_threshold: Minimum overlap ratio to trigger merge

        Returns:
            List of merged (x_min, y_min, x_max, y_max) tuples
        """
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []

        for box in boxes:
            if not merged:
                merged.append(list(box))
                continue

            # Check if current box overlaps with any merged box
            found_overlap = False
            for i, mbox in enumerate(merged):
                # Calculate intersection
                x1 = max(box[0], mbox[0])
                y1 = max(box[1], mbox[1])
                x2 = min(box[2], mbox[2])
                y2 = min(box[3], mbox[3])

                if x1 < x2 and y1 < y2:
                    # Calculate overlap ratio
                    intersection = (x2 - x1) * (y2 - y1)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    mbox_area = (mbox[2] - mbox[0]) * (mbox[3] - mbox[1])
                    min_area = min(box_area, mbox_area)

                    if min_area > 0 and intersection / min_area > overlap_threshold:
                        # Merge boxes
                        merged[i] = [
                            min(box[0], mbox[0]),
                            min(box[1], mbox[1]),
                            max(box[2], mbox[2]),
                            max(box[3], mbox[3])
                        ]
                        found_overlap = True
                        break

            if not found_overlap:
                merged.append(list(box))

        return [tuple(b) for b in merged]

    def merge_airplane_detections(self, detection_infos, overlap_threshold=0.3, angle_threshold=20.0):
        """Merge overlapping airplane detections, keeping distinct airplanes separate.

        Unlike generic box merging, this considers trail angle to avoid merging
        two different airplanes whose bounding boxes happen to overlap (e.g. crossing
        paths). Two detections are only merged if their boxes overlap AND their
        trail angles are similar (within angle_threshold degrees).

        Args:
            detection_infos: List of detection_info dicts from classify_trail,
                each containing 'bbox', 'angle', 'center', 'length', etc.
            overlap_threshold: Minimum overlap ratio to consider merging (0-1)
            angle_threshold: Maximum angle difference (degrees) to allow merging

        Returns:
            List of merged detection_info dicts. Merged entries combine bounding
            boxes and average the metadata from their constituent detections.
        """
        if not detection_infos:
            return []

        # Sort by x_min of bbox for consistent processing
        infos = sorted(detection_infos, key=lambda d: d['bbox'][0])
        merged = []

        for info in infos:
            if not merged:
                # Wrap in a list to track constituent detections for averaging
                merged.append({
                    'bbox': list(info['bbox']),
                    'angle': info['angle'],
                    'center': info['center'],
                    'length': info['length'],
                    'avg_brightness': info['avg_brightness'],
                    'max_brightness': info['max_brightness'],
                    'line': info['line'],
                    '_count': 1,
                })
                continue

            box = info['bbox']
            found_overlap = False

            for i, minfo in enumerate(merged):
                mbox = minfo['bbox']

                # Calculate intersection
                x1 = max(box[0], mbox[0])
                y1 = max(box[1], mbox[1])
                x2 = min(box[2], mbox[2])
                y2 = min(box[3], mbox[3])

                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    mbox_area = (mbox[2] - mbox[0]) * (mbox[3] - mbox[1])
                    min_area = min(box_area, mbox_area)

                    if min_area > 0 and intersection / min_area > overlap_threshold:
                        # Check angle similarity before merging
                        angle_diff = abs(info['angle'] - minfo['angle'])
                        # Angles wrap around (0 and 180 are similar for lines)
                        angle_diff = min(angle_diff, 180 - angle_diff)

                        if angle_diff <= angle_threshold:
                            # Same airplane - merge bounding boxes and average metadata
                            n = minfo['_count']
                            merged[i]['bbox'] = [
                                min(box[0], mbox[0]),
                                min(box[1], mbox[1]),
                                max(box[2], mbox[2]),
                                max(box[3], mbox[3])
                            ]
                            # Running average of metadata
                            merged[i]['angle'] = (minfo['angle'] * n + info['angle']) / (n + 1)
                            merged[i]['center'] = (
                                (minfo['center'][0] * n + info['center'][0]) / (n + 1),
                                (minfo['center'][1] * n + info['center'][1]) / (n + 1),
                            )
                            merged[i]['length'] = max(minfo['length'], info['length'])
                            merged[i]['avg_brightness'] = (minfo['avg_brightness'] * n + info['avg_brightness']) / (n + 1)
                            merged[i]['max_brightness'] = max(minfo['max_brightness'], info['max_brightness'])
                            merged[i]['_count'] = n + 1
                            found_overlap = True
                            break
                        # else: angles differ too much - treat as separate airplanes

            if not found_overlap:
                merged.append({
                    'bbox': list(info['bbox']),
                    'angle': info['angle'],
                    'center': info['center'],
                    'length': info['length'],
                    'avg_brightness': info['avg_brightness'],
                    'max_brightness': info['max_brightness'],
                    'line': info['line'],
                    '_count': 1,
                })

        # Convert bbox lists back to tuples and remove internal _count
        results = []
        for m in merged:
            results.append({
                'bbox': tuple(m['bbox']),
                'angle': m['angle'],
                'center': m['center'],
                'length': m['length'],
                'avg_brightness': m['avg_brightness'],
                'max_brightness': m['max_brightness'],
                'line': m['line'],
            })
        return results

    def detect_trails(self, frame, debug_info=None, temporal_context=None,
                       exposure_time=13.0, fov_degrees=None):
        """
        Detect and classify trails in a frame as satellites or airplanes.

        Supports multiple simultaneous detections of each type. Airplane
        detections use angle-aware merging so that two airplanes with
        crossing or nearby paths are kept as separate detections.

        Uses a two-stage pipeline:
          1. Primary detection via Canny edge detection + Hough line transform.
          2. Supplementary detection via directional matched filtering for very
             dim trails that fall below the Canny edge threshold.

        When a temporal_context is provided (from TemporalFrameBuffer), the
        supplementary stage uses the temporal difference image instead of
        spatial median subtraction — dramatically improving SNR for dim trails.

        After detection and merging, each trail is enriched with:
          - Streak photometry (lightcurve classification)
          - Trail curvature (orbit altitude proxy)
          - Angular velocity estimate (orbit class)

        Args:
            frame: Input frame
            debug_info: Optional dict to collect debug information
            temporal_context: Optional dict from TemporalFrameBuffer with
                'diff_image', 'noise_map', 'reference', 'buffer_depth'.
            exposure_time: Exposure time per frame in seconds (default 13.0).
                Used for angular velocity estimation.
            fov_degrees: Horizontal field of view in degrees (optional).
                Enables angular velocity in degrees/second.

        Returns:
            List of tuples: [('satellite', detection_info), ('airplane', detection_info), ...]
            where detection_info is a dict with keys:
                'bbox': (x_min, y_min, x_max, y_max)
                'angle': trail angle in degrees (0-180)
                'center': (x, y) center point of the trail
                'length': trail length in pixels
                'avg_brightness': mean brightness along trail
                'max_brightness': peak brightness along trail
                'line': (x1, y1, x2, y2) original line endpoints
                'photometry': dict or None (streak photometry analysis)
                'curvature': dict or None (trail curvature fit)
                'velocity': dict or None (angular velocity estimate)
        """
        gray, preprocessed = self.preprocess_frame(frame)
        lines, edges = self.detect_lines(preprocessed)

        # Pre-compute HSV frame once for all line classifications (performance optimization)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pre-allocate reusable mask array (performance optimization)
        reusable_mask = np.zeros(gray.shape, dtype=np.uint8)

        classified_trails = []
        all_classifications = []  # For debug: store all attempted classifications

        # Translation ledger: count primary lines detected
        _ledger = self.ledger
        if _ledger and lines is not None:
            _ledger.total_lines_detected += len(lines)

        # --- Stage 1: Primary detection (Canny + Hough) ---
        if lines is not None:
            for line in lines:
                trail_type, detection_info = self.classify_trail(
                    line, gray, frame, hsv_frame, reusable_mask, ledger=_ledger)

                # Store for debug (even if filtered out)
                if debug_info is not None:
                    all_classifications.append({
                        'line': line,
                        'type': trail_type,
                        'detection_info': detection_info,
                        # Keep 'bbox' for backward compat with debug panel lookup
                        'bbox': detection_info['bbox'] if detection_info is not None else None,
                    })

                if trail_type and detection_info:
                    classified_trails.append((trail_type, detection_info))

        # --- Stage 2: Supplementary dim-trail detection (Matched Filter) ---
        # Directional matched filtering catches dim linear features that fall
        # below Canny's edge threshold.  Only new (non-duplicate) candidates
        # are returned, so existing primary detections are not affected.
        # When temporal_context is available, the matched filter uses the
        # temporal difference image for dramatically improved SNR.
        supplementary_lines = self._detect_dim_lines_matched_filter(
            gray, lines, temporal_context=temporal_context)
        if supplementary_lines is not None:
            if _ledger:
                _ledger.supplementary_lines += len(supplementary_lines)
            for line in supplementary_lines:
                trail_type, detection_info = self.classify_trail(
                    line, gray, frame, hsv_frame, reusable_mask,
                    supplementary=True, ledger=_ledger)

                if debug_info is not None:
                    all_classifications.append({
                        'line': line,
                        'type': trail_type,
                        'detection_info': detection_info,
                        'bbox': detection_info['bbox'] if detection_info is not None else None,
                    })

                if trail_type and detection_info:
                    classified_trails.append((trail_type, detection_info))

        # Store debug info
        if debug_info is not None:
            all_lines_combined = []
            if lines is not None:
                all_lines_combined.extend(lines)
            if supplementary_lines is not None:
                all_lines_combined.extend(supplementary_lines)
            debug_info['all_lines'] = all_lines_combined if all_lines_combined else []
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray
            if temporal_context is not None:
                debug_info['temporal_context'] = temporal_context

        # Separate by type for merging
        satellite_infos, airplane_infos, anomalous_infos = [], [], []
        for t, info in classified_trails:
            if t == 'satellite':
                satellite_infos.append(info)
            elif t == 'airplane':
                airplane_infos.append(info)
            elif t == 'anomalous':
                anomalous_infos.append(info)

        # Merge overlapping satellite detections (simple box merge)
        satellite_boxes = [info['bbox'] for info in satellite_infos]
        merged_satellite_boxes = self.merge_overlapping_boxes(satellite_boxes)

        # Rebuild satellite detection_info from merged boxes (use nearest original info)
        merged_satellite_infos = []
        for mbox in merged_satellite_boxes:
            # Find the original detection whose center is closest to the merged box center
            mx = (mbox[0] + mbox[2]) / 2
            my = (mbox[1] + mbox[3]) / 2
            best = None
            best_dist = float('inf')
            for info in satellite_infos:
                dx = info['center'][0] - mx
                dy = info['center'][1] - my
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best = info
            if best:
                merged_info = dict(best)
                merged_info['bbox'] = mbox
                merged_satellite_infos.append(merged_info)

        # Merge airplane detections with angle awareness (keeps distinct airplanes separate)
        merged_airplane_infos = self.merge_airplane_detections(airplane_infos)

        # Merge anomalous detections (simple box merge like satellites)
        anomalous_boxes = [info['bbox'] for info in anomalous_infos]
        merged_anomalous_boxes = self.merge_overlapping_boxes(anomalous_boxes)
        merged_anomalous_infos = []
        for mbox in merged_anomalous_boxes:
            mx = (mbox[0] + mbox[2]) / 2
            my = (mbox[1] + mbox[3]) / 2
            best, best_dist = None, float('inf')
            for info in anomalous_infos:
                dx = info['center'][0] - mx
                dy = info['center'][1] - my
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist, best = dist, info
            if best:
                merged_info = dict(best)
                merged_info['bbox'] = mbox
                merged_anomalous_infos.append(merged_info)

        # --- Enrich detections with photometry, curvature, velocity ---
        frame_width = frame.shape[1]
        diff_img = temporal_context['diff_image'] if temporal_context else None

        all_merged = (
            [('satellite', info) for info in merged_satellite_infos] +
            [('airplane', info) for info in merged_airplane_infos] +
            [('anomalous', info) for info in merged_anomalous_infos]
        )
        # Skip expensive photometry/curvature when detection count is high
        # (likely false-positive-heavy frames). Velocity is cheap and always runs.
        do_full_enrichment = len(all_merged) <= 20
        for trail_type, info in all_merged:
            line_arr = np.array([[info['line'][0], info['line'][1],
                                  info['line'][2], info['line'][3]]])

            if do_full_enrichment:
                # Streak photometry
                info['photometry'] = self._analyze_streak_photometry(gray, line_arr)

                # Trail curvature
                info['curvature'] = self._fit_trail_curvature(
                    gray, line_arr, diff_image=diff_img)
            else:
                info['photometry'] = None
                info['curvature'] = None

            # Angular velocity (cheap, always computed)
            info['velocity'] = self._estimate_angular_velocity(
                info['length'], frame_width,
                exposure_time=exposure_time,
                fov_degrees=fov_degrees)

            # Inscription metadata (Latour) — the detection as a constructed artifact.
            # Every detection carries the full trace of its construction so it is
            # self-documenting and scientifically auditable.
            info['inscription'] = {
                'software_version': __version__,
                'algorithm': 'default',
                'sensitivity': getattr(self, 'sensitivity', 'medium'),
                'parameters_at_detection': {
                    'satellite_contrast_min': self.params.get('satellite_contrast_min'),
                    'canny_low': self.params.get('canny_low'),
                    'canny_high': self.params.get('canny_high'),
                    'min_line_length': self.params.get('min_line_length'),
                    'satellite_min_length': self.params.get('satellite_min_length'),
                    'satellite_max_length': self.params.get('satellite_max_length'),
                },
                'observer_context': getattr(self, '_observer_context', None),
            }

        return all_merged

    def draw_dotted_line(self, frame, pt1, pt2, color, thickness):
        """Draw a dotted line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Calculate line length and direction
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        if distance == 0:
            return

        # Normalize direction
        dx_norm = dx / distance
        dy_norm = dy / distance

        # Draw dashes along the line
        current_pos = 0
        segment_length = self.dot_length + self.gap_length

        while current_pos < distance:
            # Calculate start and end of current dash
            dash_start = current_pos
            dash_end = min(current_pos + self.dot_length, distance)

            # Calculate pixel coordinates
            start_x = int(x1 + dx_norm * dash_start)
            start_y = int(y1 + dy_norm * dash_start)
            end_x = int(x1 + dx_norm * dash_end)
            end_y = int(y1 + dy_norm * dash_end)

            # Draw the dash
            cv2.line(frame, (start_x, start_y), (end_x, end_y), color, thickness)

            current_pos += segment_length

    def draw_dotted_rectangle(self, frame, pt1, pt2, color, thickness):
        """Draw a dotted rectangle."""
        x_min, y_min = pt1
        x_max, y_max = pt2

        # Draw four dotted sides
        self.draw_dotted_line(frame, (x_min, y_min), (x_max, y_min), color, thickness)  # Top
        self.draw_dotted_line(frame, (x_max, y_min), (x_max, y_max), color, thickness)  # Right
        self.draw_dotted_line(frame, (x_max, y_max), (x_min, y_max), color, thickness)  # Bottom
        self.draw_dotted_line(frame, (x_min, y_max), (x_min, y_min), color, thickness)  # Left

    def draw_highlight(self, frame, trail_type, bbox, show_label=True):
        """Draw a dotted bounding box around the detected trail with optional label and color."""
        x_min, y_min, x_max, y_max = bbox

        # Select color and label based on trail type
        if trail_type == 'airplane':
            color = self.airplane_color
            label = "AIRPLANE"
        elif trail_type == 'anomalous':
            color = self.anomalous_color
            label = "Anomalous"
        else:  # satellite
            color = self.satellite_color
            label = "Satellite"

        # Draw dotted rectangle
        self.draw_dotted_rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            color,
            self.box_thickness
        )

        # Add label with semi-transparent background (if enabled)
        if show_label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            label_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

            # Position label above the box
            label_x = x_min
            label_y = y_min - 10 if y_min > 30 else y_max + 20

            # Create semi-transparent background for label (ROI-based, not full frame copy)
            alpha = 0.7
            rx1 = max(0, label_x - 2)
            ry1 = max(0, label_y - label_size[1] - 6)
            rx2 = min(frame.shape[1], label_x + label_size[0] + 4)
            ry2 = min(frame.shape[0], label_y + 4)
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2].copy()
                cv2.rectangle(roi, (0, 0), (rx2 - rx1, ry2 - ry1), color, -1)
                cv2.addWeighted(roi, alpha, frame[ry1:ry2, rx1:rx2], 1 - alpha, 0,
                                frame[ry1:ry2, rx1:rx2])

            # Draw label text
            cv2.putText(
                frame, label,
                (label_x, label_y),
                font, font_scale,
                (0, 0, 0),  # Black text for better contrast
                font_thickness,
                cv2.LINE_AA
            )

        return frame

    def create_debug_frame(self, frame, debug_info):
        """
        Create a debug visualization showing all detected lines and their classifications.

        Args:
            frame: Original frame
            debug_info: Dict with 'all_lines', 'all_classifications', 'edges'

        Returns:
            Debug visualization frame
        """
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]

        # Draw edge detection result as background (dimmed)
        if 'edges' in debug_info and debug_info['edges'] is not None:
            edges_colored = cv2.cvtColor(debug_info['edges'], cv2.COLOR_GRAY2BGR)
            # Dim the edges
            edges_colored = (edges_colored * 0.3).astype(np.uint8)
            debug_frame = cv2.addWeighted(debug_frame, 0.7, edges_colored, 0.3, 0)

        # Color coding:
        # - Green: Detected as airplane
        # - Cyan: Detected as satellite
        # - Red: Filtered out (too dim, wrong aspect ratio, etc.)
        # - Yellow: Filtered out but had some airplane characteristics
        # - Magenta: Filtered out but had some satellite characteristics

        if 'all_classifications' in debug_info:
            for classification in debug_info['all_classifications']:
                line = classification['line']
                trail_type = classification['type']
                bbox = classification['bbox']

                x1, y1, x2, y2 = line[0]

                # Determine color based on classification
                if trail_type == 'airplane':
                    color = (0, 255, 0)  # Green - detected airplane
                    thickness = 1
                    label = "A"
                elif trail_type == 'satellite':
                    color = (255, 255, 0)  # Cyan - detected satellite
                    thickness = 1
                    label = "S"
                elif trail_type == 'anomalous':
                    color = (200, 50, 200)  # Magenta - anomalous (residual)
                    thickness = 1
                    label = "?"
                else:
                    # Filtered out - use red but thinner
                    color = (0, 0, 255)  # Red - filtered
                    thickness = 1
                    label = "X"

                # Draw the line
                cv2.line(debug_frame, (x1, y1), (x2, y2), color, thickness)

                # Draw small circles at endpoints
                cv2.circle(debug_frame, (x1, y1), 3, color, -1)
                cv2.circle(debug_frame, (x2, y2), 3, color, -1)

                # Draw label at midpoint
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                cv2.putText(
                    debug_frame,
                    label,
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # Add legend in top-left corner
        legend_y = 20
        legend_x = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Semi-transparent background for legend
        overlay = debug_frame.copy()
        cv2.rectangle(overlay, (5, 5), (180, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug_frame, 0.4, 0, debug_frame)

        # Legend text
        cv2.putText(debug_frame, "DEBUG VIEW", (legend_x, legend_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        legend_y += 20
        cv2.putText(debug_frame, "Green (A): Airplane", (legend_x, legend_y), font, font_scale - 0.1, (0, 255, 0), font_thickness, cv2.LINE_AA)
        legend_y += 15
        cv2.putText(debug_frame, "Cyan (S): Satellite", (legend_x, legend_y), font, font_scale - 0.1, (255, 255, 0), font_thickness, cv2.LINE_AA)
        legend_y += 15
        cv2.putText(debug_frame, "Magenta (?): Anomalous", (legend_x, legend_y), font, font_scale - 0.1, (200, 50, 200), font_thickness, cv2.LINE_AA)
        legend_y += 15
        cv2.putText(debug_frame, "Red (X): Filtered", (legend_x, legend_y), font, font_scale - 0.1, (0, 0, 255), font_thickness, cv2.LINE_AA)

        # Add count at bottom
        if 'all_classifications' in debug_info:
            total_lines = len(debug_info['all_classifications'])
            detected = sum(1 for c in debug_info['all_classifications'] if c['type'] is not None)
            filtered = total_lines - detected

            count_text = f"Lines: {total_lines} | Detected: {detected} | Filtered: {filtered}"
            cv2.putText(
                debug_frame,
                count_text,
                (10, height - 10),
                font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return debug_frame

    def create_detection_debug_panel(self, frame, line, trail_type, edges, gray_frame):
        """
        Create a small debug panel showing edge detection and point features for a detection.

        Args:
            frame: Original color frame
            line: Detected line
            trail_type: 'airplane' or 'satellite'
            edges: Edge detection output
            gray_frame: Grayscale frame

        Returns:
            Small debug visualization panel (width x height)
        """
        x1, y1, x2, y2 = line[0]

        # Create bounding box around the trail with padding
        padding = 40
        x_min = max(0, min(x1, x2) - padding)
        y_min = max(0, min(y1, y2) - padding)
        x_max = min(frame.shape[1], max(x1, x2) + padding)
        y_max = min(frame.shape[0], max(y1, y2) + padding)

        # Crop regions
        frame_crop = frame[y_min:y_max, x_min:x_max].copy()
        edges_crop = edges[y_min:y_max, x_min:x_max]
        gray_crop = gray_frame[y_min:y_max, x_min:x_max]

        # Get point features debug info for airplanes
        if trail_type == 'airplane':
            # Adjust line coordinates to cropped region
            line_crop = [[x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min]]
            num_peaks, point_debug = self.detect_point_features(line_crop, gray_crop, return_debug_info=True)

            # Draw point features on the crop
            points_viz = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)

            # Draw the line
            cv2.line(points_viz, (line_crop[0][0], line_crop[0][1]),
                    (line_crop[0][2], line_crop[0][3]), (0, 255, 0), 1)

            # Mark detected peaks with circles
            for peak_idx in point_debug['peak_indices']:
                if peak_idx < len(point_debug['sample_points']):
                    px, py = point_debug['sample_points'][peak_idx]
                    cv2.circle(points_viz, (px, py), 3, (0, 0, 255), -1)

            # Draw all sample points as small dots
            for px, py in point_debug['sample_points']:
                cv2.circle(points_viz, (px, py), 1, (255, 255, 0), -1)
        else:
            # For satellites, just show the trail on grayscale
            points_viz = cv2.cvtColor(gray_crop, cv2.COLOR_GRAY2BGR)
            line_crop = [[x1 - x_min, y1 - y_min, x2 - x_min, y2 - y_min]]
            cv2.line(points_viz, (line_crop[0][0], line_crop[0][1]),
                    (line_crop[0][2], line_crop[0][3]), (0, 255, 255), 1)

        # Convert edges to BGR for visualization
        edges_viz = cv2.cvtColor(edges_crop, cv2.COLOR_GRAY2BGR)

        # Resize both to a standard small size (150 pixels wide)
        target_width = 150
        scale = target_width / max(edges_viz.shape[1], 1)
        target_height = int(edges_viz.shape[0] * scale)

        if target_height > 0:
            edges_viz = cv2.resize(edges_viz, (target_width, target_height))
            points_viz = cv2.resize(points_viz, (target_width, target_height))
            frame_crop = cv2.resize(frame_crop, (target_width, target_height))

        # Stack vertically: original crop, edges, point features
        # Add labels
        label_height = 15
        label_bg = np.zeros((label_height, target_width, 3), dtype=np.uint8)

        # Label 1: "Original"
        label1 = label_bg.copy()
        cv2.putText(label1, "Original", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Label 2: "Edges"
        label2 = label_bg.copy()
        cv2.putText(label2, "Edges", (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Label 3: "Points" or "Trail"
        label3 = label_bg.copy()
        text = "Points" if trail_type == 'airplane' else "Trail"
        cv2.putText(label3, text, (5, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        # Stack all together
        panel = np.vstack([label1, frame_crop, label2, edges_viz, label3, points_viz])

        # Add border
        panel = cv2.copyMakeBorder(panel, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        return panel


class RadonStreakDetector(SatelliteTrailDetector):
    """
    Advanced satellite trail detector using Radon transform, LSD (Line Segment
    Detector) with NFA significance, and perpendicular cross filtering.

    Calibrates detection thresholds from ground truth example images in a
    reference directory.  Combines three cutting-edge detection stages:

      1. **LSD + NFA**: OpenCV's a-contrario Line Segment Detector provides
         subpixel-accurate line detection with statistically rigorous false
         alarm control (Number of False Alarms framework).

      2. **Radon Transform streak detection**: The Radon transform projects
         the image along all orientations, turning linear streaks into bright
         peaks in sinogram space.  SNR-normalised peak detection finds streaks
         down to ~SNR 1.5 — far below what edge-based methods can achieve.

      3. **Perpendicular Cross Filtering (PCF)**: For each candidate streak,
         a matched filter is applied both parallel and perpendicular to the
         streak direction.  Real linear features produce a high parallel /
         perpendicular response ratio; stars and noise do not.

    Ground truth calibration measures PSF width, brightness profiles, and
    angular distributions from example trail patches, dynamically adapting
    all detection thresholds.

    Inherits drawing, debug visualisation, and classification methods from
    SatelliteTrailDetector.

    Usage:
        python satellite_trail_detector.py input.mp4 output.mp4 --algorithm radon
        python satellite_trail_detector.py input.mp4 output.mp4 --algorithm radon --groundtruth ./groundtruth
    """

    def __init__(self, sensitivity='medium', preprocessing_params=None,
                 skip_aspect_ratio_check=False, signal_envelope=None,
                 groundtruth_dir=None):
        super().__init__(sensitivity, preprocessing_params,
                         skip_aspect_ratio_check, signal_envelope)

        # LSD availability flag (removed in some OpenCV builds)
        self._lsd_available = hasattr(cv2, 'createLineSegmentDetector')

        # Cached CLAHE for LSD stage (clipLimit=8.0, different from parent's 6.0)
        self._clahe_lsd = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(6, 6))

        # Radon detection parameters (tuned for performance)
        self.radon_num_angles = 90        # 2-degree angular resolution (fast)
        self.radon_max_dim = 480          # Max short-side pixels for Radon input
        self.radon_snr_threshold = 3.0    # SNR threshold in sinogram space
        self.pcf_ratio_threshold = 2.5    # Parallel / perpendicular ratio (raised from 2.0 to reject star residuals)
        self.pcf_kernel_length = 31       # Matched filter kernel length for PCF

        # Overridable from Radon preview (default None = use hardcoded values)
        self._star_mask_sigma = None      # Star mask threshold (default 5.0σ)
        self._lsd_log_eps = None          # LSD significance (default 1.0)

        # Multi-frame residual accumulator for Radon SNR boost.
        # In sequential mode, accumulates star-subtracted residuals across
        # frames so the Radon transform sees sqrt(N) better SNR.
        # In parallel mode (workers > 0), each worker has its own instance
        # so accumulation doesn't work — Radon runs on single frames.
        self._residual_buffer = []        # ring buffer of cleaned residuals
        self._residual_buffer_depth = 4   # how many frames to stack
        self._residual_star_masks = []    # corresponding star masks

        # Ground truth calibration profiles (must come after parameter defaults)
        self.gt_profiles = None
        if groundtruth_dir is not None:
            self.gt_profiles = self._calibrate_from_groundtruth(groundtruth_dir)
            if self.gt_profiles:
                self._apply_gt_calibration(self.gt_profiles)

    # ── Ground truth calibration ────────────────────────────────────

    def _calibrate_from_groundtruth(self, gt_dir):
        """Load ground truth trail patches and extract detection parameters.

        Analyzes each image to measure:
          - PSF width (Gaussian fit to perpendicular cross-section)
          - Trail brightness profile (mean, std, peak)
          - Trail angle
          - Trail-to-background contrast ratio
          - Trail length

        Returns:
            Dict with calibrated parameters, or None if no valid images found.
        """
        gt_path = Path(gt_dir)
        if not gt_path.exists():
            print(f"Warning: groundtruth directory not found: {gt_dir}")
            return None

        images = sorted(gt_path.glob('*.png'))
        if not images:
            print(f"Warning: no PNG images in {gt_dir}")
            return None

        psf_widths = []
        brightnesses = []
        contrasts = []
        angles = []
        lengths = []

        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Skip very small images
            h, w = img.shape
            if h < 20 or w < 20:
                continue

            profile = self._analyze_gt_patch(img)
            if profile is None:
                continue

            psf_widths.append(profile['psf_width'])
            brightnesses.append(profile['brightness'])
            contrasts.append(profile['contrast'])
            angles.append(profile['angle'])
            lengths.append(profile['length'])

        if not brightnesses:
            print("Warning: could not extract profiles from groundtruth images")
            return None

        calibration = {
            'psf_width_median': float(np.median(psf_widths)),
            'psf_width_range': (float(np.min(psf_widths)), float(np.max(psf_widths))),
            'brightness_median': float(np.median(brightnesses)),
            'brightness_range': (float(np.min(brightnesses)), float(np.max(brightnesses))),
            'contrast_median': float(np.median(contrasts)),
            'contrast_range': (float(np.min(contrasts)), float(np.max(contrasts))),
            'angles': angles,
            'length_range': (float(np.min(lengths)), float(np.max(lengths))),
            'num_examples': len(brightnesses),
        }

        print(f"Ground truth calibration: {calibration['num_examples']} examples")
        print(f"  PSF width: {calibration['psf_width_median']:.1f}px "
              f"(range {calibration['psf_width_range'][0]:.1f}-{calibration['psf_width_range'][1]:.1f})")
        print(f"  Brightness: {calibration['brightness_median']:.1f} "
              f"(range {calibration['brightness_range'][0]:.1f}-{calibration['brightness_range'][1]:.1f})")
        print(f"  Contrast: {calibration['contrast_median']:.3f} "
              f"(range {calibration['contrast_range'][0]:.3f}-{calibration['contrast_range'][1]:.3f})")

        return calibration

    def _analyze_gt_patch(self, gray_patch):
        """Analyze a single ground truth trail patch.

        Uses Hough transform to find the dominant line, then measures
        cross-sectional PSF width, brightness, and contrast.

        Returns:
            Dict with 'psf_width', 'brightness', 'contrast', 'angle', 'length'
            or None if no trail detected.
        """
        h, w = gray_patch.shape

        # Background estimation
        bg = cv2.medianBlur(gray_patch, min(31, max(3, (min(h, w) // 4) | 1)))
        residual = gray_patch.astype(np.float64) - bg.astype(np.float64)

        # Noise estimation via MAD
        flat = residual.ravel()
        mad = np.median(np.abs(flat - np.median(flat)))
        noise_sigma = max(0.5, mad * 1.4826)

        # Edge detection for Hough
        residual_u8 = np.clip(residual * (255.0 / max(1, residual.max())), 0, 255).astype(np.uint8)
        edges = cv2.Canny(residual_u8, 10, 50)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15,
                                minLineLength=max(15, min(h, w) // 3),
                                maxLineGap=10)

        if lines is None or len(lines) == 0:
            # Fallback: try LSD if available
            if self._lsd_available:
                lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
                lsd_lines, _, _, _ = lsd.detect(gray_patch)
                if lsd_lines is not None and len(lsd_lines) > 0:
                    # Pick longest
                    best_len = 0
                    best_line = None
                    for l in lsd_lines:
                        lx1, ly1, lx2, ly2 = l[0]
                        ll = np.sqrt((lx2 - lx1) ** 2 + (ly2 - ly1) ** 2)
                        if ll > best_len:
                            best_len = ll
                            best_line = l
                    if best_line is not None:
                        lines = np.array([[[int(best_line[0][0]), int(best_line[0][1]),
                                            int(best_line[0][2]), int(best_line[0][3])]]])
            if lines is None or len(lines) == 0:
                return None

        # Pick the longest line
        best_len = 0
        best_line = None
        for l in lines:
            x1, y1, x2, y2 = l[0]
            ll = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if ll > best_len:
                best_len = ll
                best_line = (x1, y1, x2, y2)

        if best_line is None or best_len < 15:
            return None

        x1, y1, x2, y2 = best_line
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

        # Measure perpendicular cross-section (PSF width)
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx * dx + dy * dy)
        nx = -dy / length  # perpendicular unit vector
        ny = dx / length

        # Sample perpendicular profiles at several points along the trail
        num_samples = min(20, max(5, int(length / 10)))
        cross_profiles = []
        half_width = 8  # sample +/- 8 pixels perpendicular

        for i in range(num_samples):
            t = (i + 1) / (num_samples + 1)
            cx = x1 + t * dx
            cy = y1 + t * dy

            profile = []
            for d in range(-half_width, half_width + 1):
                px = int(round(cx + d * nx))
                py = int(round(cy + d * ny))
                if 0 <= py < h and 0 <= px < w:
                    profile.append(float(residual[py, px]))
                else:
                    profile.append(0.0)
            cross_profiles.append(profile)

        if not cross_profiles:
            return None

        avg_profile = np.mean(cross_profiles, axis=0)

        # Fit Gaussian to cross-section to get PSF width
        peak_idx = np.argmax(avg_profile)
        peak_val = avg_profile[peak_idx]
        if peak_val <= noise_sigma:
            psf_width = 2.0  # default
        else:
            half_max = peak_val / 2.0
            above = avg_profile >= half_max
            fwhm_pixels = float(np.sum(above))
            psf_width = max(1.0, fwhm_pixels / 2.355)  # sigma from FWHM

        # Trail brightness (mean along the line)
        trail_brightness_vals = []
        for i in range(int(length)):
            t = i / max(1, length - 1)
            px = int(round(x1 + t * dx))
            py = int(round(y1 + t * dy))
            if 0 <= py < h and 0 <= px < w:
                trail_brightness_vals.append(float(gray_patch[py, px]))

        if not trail_brightness_vals:
            return None

        trail_brightness = np.mean(trail_brightness_vals)

        # Background brightness (median of the patch)
        bg_brightness = float(np.median(gray_patch))
        contrast = trail_brightness / max(1.0, bg_brightness)

        return {
            'psf_width': psf_width,
            'brightness': trail_brightness,
            'contrast': contrast,
            'angle': angle,
            'length': best_len,
        }

    def _apply_gt_calibration(self, cal):
        """Adapt detection parameters from ground truth calibration."""
        # Widen length range to cover all GT examples
        self.params['satellite_min_length'] = min(
            self.params['satellite_min_length'],
            max(20, int(cal['length_range'][0] * 0.7)))
        self.params['satellite_max_length'] = max(
            self.params['satellite_max_length'],
            int(cal['length_range'][1] * 1.5))

        # Lower contrast threshold to match dimmest GT trail
        self.params['satellite_contrast_min'] = min(
            self.params['satellite_contrast_min'],
            max(1.01, cal['contrast_range'][0] * 0.9))

        # Calibrate Radon SNR threshold based on GT brightness
        if cal['brightness_median'] < 15:
            self.radon_snr_threshold = 2.0  # very dim trails — be more sensitive
        elif cal['brightness_median'] < 25:
            self.radon_snr_threshold = 2.5

        # Set PCF kernel length based on PSF width
        self.pcf_kernel_length = max(21, int(cal['psf_width_median'] * 15))
        if self.pcf_kernel_length % 2 == 0:
            self.pcf_kernel_length += 1

        print(f"  Radon SNR threshold: {self.radon_snr_threshold}")
        print(f"  PCF kernel length: {self.pcf_kernel_length}")

    # ── LSD detection with NFA significance ─────────────────────────

    def _detect_lines_lsd(self, gray_frame, min_length=50, log_eps=0.0):
        """Detect lines using OpenCV's Line Segment Detector with NFA filtering.

        The a-contrario framework detects line segments whose number of aligned
        gradient pixels is too large to occur by chance.  Each segment gets an
        NFA (Number of False Alarms) score — lower is more significant.

        Args:
            gray_frame: Grayscale input (uint8)
            min_length: Minimum segment length in pixels
            log_eps: Detection threshold (-log10(NFA) > log_eps).
                     Higher = stricter (fewer detections).

        Returns:
            List of (x1, y1, x2, y2, nfa) tuples, sorted by significance.
        """
        if not self._lsd_available:
            return []

        lsd = cv2.createLineSegmentDetector(
            refine=cv2.LSD_REFINE_ADV,
            scale=0.8,
            sigma_scale=0.6,
            quant=2.0,
            ang_th=22.5,
            log_eps=log_eps,
            density_th=0.5,
            n_bins=1024
        )

        lines, widths, precisions, nfas = lsd.detect(gray_frame)
        if lines is None:
            return []

        results = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length >= min_length:
                nfa_val = nfas[i][0] if nfas is not None else 0.0
                results.append((float(x1), float(y1), float(x2), float(y2), nfa_val))

        # Sort by NFA (most significant first — most negative)
        results.sort(key=lambda r: r[4])
        return results

    # ── Radon transform streak detection ────────────────────────────

    def _radon_transform(self, image, num_angles=90):
        """Compute the Radon transform (sinogram) of an image.

        Uses float32 arithmetic and INTER_NEAREST for maximum speed.

        Args:
            image: 2D float array (background-subtracted, star-cleaned)
            num_angles: Number of projection angles in [0, 180)

        Returns:
            sinogram: 2D array of shape (num_offsets, num_angles)
            angles: 1D array of angles in degrees
        """
        h, w = image.shape
        diag = int(np.ceil(np.sqrt(h * h + w * w)))
        pad_h = (diag - h) // 2
        pad_w = (diag - w) // 2
        padded = np.zeros((diag, diag), dtype=np.float32)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image.astype(np.float32)

        angles = np.linspace(0, 180, num_angles, endpoint=False)
        sinogram = np.zeros((diag, num_angles), dtype=np.float32)
        center = (diag / 2.0, diag / 2.0)

        # GPU-accelerated path: upload padded image once, reduce on GPU
        # to avoid downloading the full diag×diag rotated image per angle.
        # cv2.cuda.reduce sums columns on-device, so we only download a
        # 1×diag row per angle (~700× less data than the full image).
        if self._use_gpu_warp:
            try:
                gpu_padded = cv2.cuda_GpuMat()
                gpu_padded.upload(padded)
                for i, theta in enumerate(angles):
                    M = cv2.getRotationMatrix2D(center, -theta, 1.0)
                    gpu_rotated = cv2.cuda.warpAffine(
                        gpu_padded, M, (diag, diag),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0,))
                    # Sum columns on GPU (axis 0 = reduce rows → 1×diag result)
                    gpu_col_sum = cv2.cuda.reduce(
                        gpu_rotated, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
                    sinogram[:, i] = gpu_col_sum.download().ravel()
                return sinogram, angles
            except Exception:
                self._use_gpu_warp = False
                # Fall through to CPU path

        # CPU path
        for i, theta in enumerate(angles):
            M = cv2.getRotationMatrix2D(center, -theta, 1.0)
            rotated = cv2.warpAffine(padded, M, (diag, diag),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            sinogram[:, i] = np.sum(rotated, axis=0)

        return sinogram, angles

    def _detect_streaks_radon(self, residual, noise_sigma, snr_threshold=3.0,
                              min_length=40):
        """Detect linear streaks via Radon transform peak finding.

        Pipeline:
          1. Compute sinogram of the background-subtracted image
          2. Normalise each sinogram column by the expected noise
          3. Find peaks above snr_threshold in the SNR sinogram
          4. Convert (angle, offset) peaks back to image-space line segments
          5. Estimate streak length from the sinogram peak profile

        Args:
            residual: Background-subtracted float64 image
            noise_sigma: Estimated noise standard deviation
            snr_threshold: Minimum SNR for a valid detection
            min_length: Minimum streak length in pixels

        Returns:
            List of (x1, y1, x2, y2, snr) tuples.
        """
        h, w = residual.shape
        diag = int(np.ceil(np.sqrt(h * h + w * w)))
        pad_h = (diag - h) // 2
        pad_w = (diag - w) // 2

        sinogram, angles = self._radon_transform(residual, self.radon_num_angles)

        # Noise normalisation: for a projection of N pixels, noise std scales
        # as noise_sigma * sqrt(N).  Use the image height as a proxy for N
        # (varies slightly with angle but this is a good first approximation).
        n_pixels = max(1.0, float(min(h, w)))
        noise_per_projection = noise_sigma * np.sqrt(n_pixels)

        snr_sinogram = sinogram / max(1e-10, noise_per_projection)

        # Suppress low-frequency background in sinogram (remove broad trends)
        # Use 2D Gaussian blur on the sinogram as a fast baseline estimator
        blur_k = min(51, max(3, diag // 10))
        if blur_k % 2 == 0:
            blur_k += 1
        baseline = cv2.GaussianBlur(
            snr_sinogram, (blur_k, 1), 0)
        snr_sinogram = snr_sinogram - baseline

        # Find peaks above threshold
        peak_mask = snr_sinogram > snr_threshold
        if not np.any(peak_mask):
            return []

        # Non-maximum suppression: dilate and compare
        # NMS window (11, 7): 11 pixels in offset covers ~44px full-res
        # (well within a single trail's sinogram footprint), 7 in angle
        # covers ~14 degrees (suppresses near-parallel ghost peaks).
        nms_size = (11, 7)
        if _HAS_SCIPY:
            local_max = _scipy_maximum_filter(snr_sinogram, size=nms_size)
        else:
            # Fallback: use cv2.dilate for NMS
            kernel_nms = np.ones(nms_size, dtype=np.uint8)
            local_max = cv2.dilate(snr_sinogram, kernel_nms)
        peaks = peak_mask & (snr_sinogram == local_max)

        peak_coords = np.argwhere(peaks)  # (offset_idx, angle_idx)
        if len(peak_coords) == 0:
            return []

        # Sort by SNR descending (vectorized fancy indexing)
        peak_snrs = snr_sinogram[peak_coords[:, 0], peak_coords[:, 1]]
        order = np.argsort(peak_snrs)[::-1]
        peak_coords = peak_coords[order]
        peak_snrs = peak_snrs[order]

        # Limit to top 20 candidates to avoid processing too many
        peak_coords = peak_coords[:20]
        peak_snrs = peak_snrs[:20]

        results = []
        center_offset = diag / 2.0

        for idx, (offset_idx, angle_idx) in enumerate(peak_coords):
            theta = angles[angle_idx]
            offset = float(offset_idx) - center_offset
            snr_val = peak_snrs[idx]

            theta_rad = np.radians(theta)
            cos_t = np.cos(theta_rad)
            sin_t = np.sin(theta_rad)

            # The line in image space: x*cos(theta) + y*sin(theta) = offset
            # Closest point on line to origin (foot of perpendicular)
            # Direction along the line: (-sin(theta), cos(theta))
            cx = offset * cos_t
            cy = offset * sin_t

            # Estimate streak length from sinogram peak width (FWHM along offset)
            col_profile = snr_sinogram[:, angle_idx]
            half_max = snr_val / 2.0
            above = col_profile >= half_max
            streak_length_est = float(np.sum(above))

            if streak_length_est < min_length:
                continue

            # Compute line endpoints along (-sin_t, cos_t) direction
            half_len = streak_length_est / 2.0
            x1 = cx - half_len * (-sin_t)
            y1 = cy - half_len * cos_t
            x2 = cx + half_len * (-sin_t)
            y2 = cy + half_len * cos_t

            # Convert from padded to original image coordinates
            x1 -= pad_w
            y1 -= pad_h
            x2 -= pad_w
            y2 -= pad_h

            # Clip to image bounds
            x1 = np.clip(x1, 0, w - 1)
            y1 = np.clip(y1, 0, h - 1)
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)

            seg_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if seg_len >= min_length:
                results.append((float(x1), float(y1), float(x2), float(y2), snr_val))

        return results

    # ── Perpendicular cross filtering ───────────────────────────────

    def _perpendicular_cross_filter(self, residual, candidates, kernel_length=None):
        """Apply perpendicular cross filtering to reject false positives.

        Uses vectorized NumPy sampling for speed.  For each candidate,
        samples brightness along the line direction and perpendicular to
        it, then compares the mean responses.

        Args:
            residual: Background-subtracted float32 image
            candidates: List of (x1, y1, x2, y2, score) tuples
            kernel_length: Half-width for perpendicular sampling

        Returns:
            Filtered list of (x1, y1, x2, y2, score) tuples.
        """
        if kernel_length is None:
            kernel_length = self.pcf_kernel_length
        half_w = kernel_length // 2

        if not candidates:
            return []

        confirmed = []
        h, w = residual.shape

        # Pre-compute the kernel offset array once (shared across candidates)
        d_offsets = np.arange(-half_w, half_w + 1, dtype=np.float32)

        for x1, y1, x2, y2, score in candidates:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            if length < 5:
                continue

            # Unit vectors: along trail and perpendicular
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux  # perpendicular

            # Sample at several points along the trail
            num_samples = max(5, min(20, int(length / 10)))

            # Vectorized: generate all sample center points at once
            t_vals = np.arange(1, num_samples + 1, dtype=np.float32) / (num_samples + 1)
            cx_arr = x1 + t_vals * dx  # (num_samples,)
            cy_arr = y1 + t_vals * dy

            # Parallel sampling coordinates: center + d * (ux, uy)
            # Shape: (num_samples, kernel_size) via broadcasting
            par_px = np.round(cx_arr[:, None] + d_offsets[None, :] * ux).astype(np.intp)
            par_py = np.round(cy_arr[:, None] + d_offsets[None, :] * uy).astype(np.intp)

            # Perpendicular sampling coordinates: center + d * (nx, ny)
            perp_px = np.round(cx_arr[:, None] + d_offsets[None, :] * nx).astype(np.intp)
            perp_py = np.round(cy_arr[:, None] + d_offsets[None, :] * ny).astype(np.intp)

            # Bounds masks
            par_valid = (par_py >= 0) & (par_py < h) & (par_px >= 0) & (par_px < w)
            perp_valid = (perp_py >= 0) & (perp_py < h) & (perp_px >= 0) & (perp_px < w)

            # Clamp coordinates for safe indexing (invalid positions will be
            # zeroed out via the mask)
            par_py_c = np.clip(par_py, 0, h - 1)
            par_px_c = np.clip(par_px, 0, w - 1)
            perp_py_c = np.clip(perp_py, 0, h - 1)
            perp_px_c = np.clip(perp_px, 0, w - 1)

            # Fancy-index all values at once
            par_vals = residual[par_py_c, par_px_c]   # (num_samples, kernel_size)
            perp_vals = residual[perp_py_c, perp_px_c]
            par_vals[~par_valid] = 0.0
            perp_vals[~perp_valid] = 0.0

            # Per-sample means (only counting valid pixels)
            par_counts = par_valid.sum(axis=1)    # (num_samples,)
            perp_counts = perp_valid.sum(axis=1)
            both_valid = (par_counts > 0) & (perp_counts > 0)

            if not np.any(both_valid):
                continue

            par_means = np.zeros(num_samples, dtype=np.float32)
            perp_means = np.zeros(num_samples, dtype=np.float32)
            par_means[both_valid] = par_vals[both_valid].sum(axis=1) / par_counts[both_valid]
            perp_means[both_valid] = perp_vals[both_valid].sum(axis=1) / perp_counts[both_valid]

            mean_par = par_means[both_valid].mean()
            mean_perp = perp_means[both_valid].mean()

            ratio = mean_par / (abs(mean_perp) + 1e-10)
            if ratio >= self.pcf_ratio_threshold:
                confirmed.append((x1, y1, x2, y2, score))

        return confirmed

    # ── Line-proximity utilities for dedup and merging ────────────

    @staticmethod
    def _line_angle(x1, y1, x2, y2):
        """Return line angle in [0, 180) degrees."""
        import math
        return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180

    @staticmethod
    def _angle_diff(a1, a2):
        """Absolute angular difference in [0, 90] degrees, handling 180-wrap."""
        d = abs(a1 - a2) % 180
        return min(d, 180 - d)

    @staticmethod
    def _perp_distance_between_lines(line1, line2):
        """Perpendicular distance from midpoint of line2 to the infinite line through line1."""
        import math
        x1, y1, x2, y2 = line1
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            mx, my = (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2
            return math.sqrt((mx - x1) ** 2 + (my - y1) ** 2)
        nx, ny = -dy / length, dx / length  # unit perpendicular
        mx, my = (line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2
        return abs((mx - x1) * nx + (my - y1) * ny)

    @classmethod
    def _is_duplicate_line(cls, line1, line2, dist_thresh=20, angle_thresh=15):
        """Check if two lines describe the same physical trail.

        Uses angle similarity + perpendicular distance between midpoints,
        far more robust than the old midpoint-in-AABB check.

        Args:
            line1: (x1, y1, x2, y2)
            line2: (x1, y1, x2, y2)
            dist_thresh: Max perpendicular distance in pixels
            angle_thresh: Max angle difference in degrees

        Returns:
            True if both lines likely describe the same physical trail.
        """
        a1 = cls._line_angle(*line1)
        a2 = cls._line_angle(*line2)
        if cls._angle_diff(a1, a2) > angle_thresh:
            return False
        # Check perpendicular distance in both directions (symmetric)
        d1 = cls._perp_distance_between_lines(line1, line2)
        d2 = cls._perp_distance_between_lines(line2, line1)
        return min(d1, d2) < dist_thresh

    @classmethod
    def _merge_collinear_segments(cls, segments, angle_thresh=5, perp_thresh=10, gap_thresh=50):
        """Merge collinear LSD segments that belong to the same physical trail.

        LSD often splits a long dim trail into 2-5 fragments.  This merges
        them before classify_trail() so each trail produces one detection.

        Args:
            segments: List of (x1, y1, x2, y2, nfa) tuples from LSD
            angle_thresh: Max angle difference (degrees) to consider collinear
            perp_thresh: Max perpendicular distance (pixels) between segments
            gap_thresh: Max gap along the line direction (pixels)

        Returns:
            List of (x1, y1, x2, y2, nfa) tuples with collinear segments merged.
        """
        import math
        if len(segments) <= 1:
            return segments

        # Build merge groups via union-find
        n = len(segments)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        # Pre-compute angles and unit vectors
        angles = []
        for x1, y1, x2, y2, _ in segments:
            angles.append(cls._line_angle(x1, y1, x2, y2))

        for i in range(n):
            for j in range(i + 1, n):
                if cls._angle_diff(angles[i], angles[j]) > angle_thresh:
                    continue

                seg_i = segments[i][:4]
                seg_j = segments[j][:4]

                # Perpendicular distance
                d1 = cls._perp_distance_between_lines(seg_i, seg_j)
                d2 = cls._perp_distance_between_lines(seg_j, seg_i)
                if min(d1, d2) > perp_thresh:
                    continue

                # Gap along line direction: project all 4 endpoints onto
                # the average direction vector and check the gap
                avg_angle = math.radians(angles[i])
                ux, uy = math.cos(avg_angle), math.sin(avg_angle)
                projs_i = [seg_i[0] * ux + seg_i[1] * uy,
                           seg_i[2] * ux + seg_i[3] * uy]
                projs_j = [seg_j[0] * ux + seg_j[1] * uy,
                           seg_j[2] * ux + seg_j[3] * uy]
                gap = max(min(projs_j) - max(projs_i),
                          min(projs_i) - max(projs_j))
                if gap > gap_thresh:
                    continue

                union(i, j)

        # Collect groups and merge each group into a single segment
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merged = []
        for indices in groups.values():
            if len(indices) == 1:
                merged.append(segments[indices[0]])
                continue

            # Merge: find the two endpoints that are farthest apart
            # (the overall trail extent), keep the best NFA
            best_nfa = min(segments[idx][4] for idx in indices)
            all_points = []
            for idx in indices:
                s = segments[idx]
                all_points.append((s[0], s[1]))
                all_points.append((s[2], s[3]))

            # Find the pair of points with maximum distance
            max_dist = 0
            p_start, p_end = all_points[0], all_points[-1]
            for a_idx in range(len(all_points)):
                for b_idx in range(a_idx + 1, len(all_points)):
                    pa, pb = all_points[a_idx], all_points[b_idx]
                    d = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                    if d > max_dist:
                        max_dist = d
                        p_start, p_end = pa, pb

            merged.append((p_start[0], p_start[1], p_end[0], p_end[1], best_nfa))

        return merged

    @classmethod
    def _merge_lines_oriented(cls, detection_infos, angle_thresh=10, perp_thresh=15):
        """Merge satellite detections using line proximity instead of AABB overlap.

        Replaces merge_overlapping_boxes() for satellite trails in the Radon
        pipeline.  Two detections merge if their trail lines are nearly
        collinear (similar angle and close perpendicular distance).

        Args:
            detection_infos: List of detection_info dicts with 'line', 'angle', etc.
            angle_thresh: Max angle difference to merge (degrees)
            perp_thresh: Max perpendicular distance to merge (pixels)

        Returns:
            List of merged detection_info dicts.
        """
        if len(detection_infos) <= 1:
            return list(detection_infos)

        n = len(detection_infos)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        for i in range(n):
            for j in range(i + 1, n):
                info_i, info_j = detection_infos[i], detection_infos[j]
                if cls._angle_diff(info_i.get('angle', 0),
                                   info_j.get('angle', 0)) > angle_thresh:
                    continue
                line_i = info_i.get('line', info_i['bbox'])
                line_j = info_j.get('line', info_j['bbox'])
                d1 = cls._perp_distance_between_lines(line_i, line_j)
                d2 = cls._perp_distance_between_lines(line_j, line_i)
                if min(d1, d2) < perp_thresh:
                    union(i, j)

        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        merged = []
        for indices in groups.values():
            # Pick the detection with the longest trail as the representative
            best_idx = max(indices, key=lambda i: detection_infos[i].get('length', 0))
            best = dict(detection_infos[best_idx])

            # Expand bbox to encompass all members
            all_bboxes = [detection_infos[i]['bbox'] for i in indices]
            best['bbox'] = (
                min(b[0] for b in all_bboxes),
                min(b[1] for b in all_bboxes),
                max(b[2] for b in all_bboxes),
                max(b[3] for b in all_bboxes),
            )
            merged.append(best)

        return merged

    # ── GPU-accelerated median blur ─────────────────────────────────

    def _median_blur_gpu(self, gray_u8, kernel_size):
        """Run medianBlur on GPU if available, otherwise fall back to CPU.

        Args:
            gray_u8: Grayscale uint8 image.
            kernel_size: Odd kernel size for median filter.

        Returns:
            Median-blurred uint8 image.
        """
        if self._use_gpu_median and _HAS_CUDA_MEDIAN:
            try:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(gray_u8)
                median_filter = cv2.cuda.createMedianFilter(
                    gpu_img.type(), kernel_size)
                gpu_result = median_filter.apply(gpu_img)
                return gpu_result.download()
            except Exception:
                # Permanent fallback — disable median GPU only
                self._use_gpu_median = False
        return cv2.medianBlur(gray_u8, kernel_size)

    # ── Dual-threshold star masking ─────────────────────────────────

    def _dual_threshold_star_mask(self, gray_frame, noise_sigma,
                                   residual=None):
        """Remove bright stars using dual-threshold segmentation.

        High threshold identifies bright stars; dilation expands the mask
        to cover diffraction spikes and halos.  The masked pixels are
        replaced with local background estimates.

        Args:
            gray_frame: Grayscale frame (uint8 or float32)
            noise_sigma: Estimated noise standard deviation
            residual: Pre-computed background-subtracted image. When
                      provided, skips the expensive medianBlur bg
                      estimation (~100ms savings at 1080p).

        Returns:
            Star-cleaned image (float32), star mask (bool)
        """
        if residual is None:
            img = gray_frame.astype(np.float32)
            bg_kernel = min(51, max(3, min(img.shape) // 8))
            if bg_kernel % 2 == 0:
                bg_kernel += 1
            gray_u8 = (gray_frame.astype(np.uint8) if img.max() <= 255
                        else np.clip(img, 0, 255).astype(np.uint8))
            bg = self._median_blur_gpu(gray_u8, bg_kernel).astype(np.float32)
            residual = img - bg

        # High threshold: identify bright stars (configurable sigma)
        star_sigma = self._star_mask_sigma if self._star_mask_sigma is not None else 5.0
        high_thresh = star_sigma * noise_sigma
        star_mask = residual > high_thresh

        # Dilate to cover halos and spikes — cap at 15 to avoid eating
        # trail pixels near stars on high-noise frames
        dilate_size = max(5, min(15, int(noise_sigma * 2)))
        if dilate_size % 2 == 0:
            dilate_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (dilate_size, dilate_size))
        star_mask_dilated = cv2.dilate(star_mask.astype(np.uint8), kernel).astype(bool)

        # Replace star pixels with local background
        cleaned = residual.copy()
        cleaned[star_mask_dilated] = 0.0

        return cleaned, star_mask_dilated

    # ── Main detection pipeline ─────────────────────────────────────

    def detect_trails(self, frame, debug_info=None, temporal_context=None,
                      exposure_time=13.0, fov_degrees=None):
        """Advanced three-stage detection pipeline.

        Stage 1: LSD + NFA for subpixel line detection with significance
        Stage 2: Radon transform for ultra-dim streak detection
        Stage 3: Perpendicular cross filtering for false positive suppression

        Falls back to parent's Canny+Hough pipeline for airplane detection
        since airplanes have distinctive bright features easily caught by
        the classical approach.

        Args:
            frame: Input BGR frame
            debug_info: Optional dict to collect debug information
            temporal_context: Optional dict from TemporalFrameBuffer
            exposure_time: Exposure time per frame in seconds
            fov_degrees: Horizontal field of view in degrees

        Returns:
            List of (trail_type, detection_info) tuples.
        """
        gray, preprocessed = self.preprocess_frame(frame)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        reusable_mask = np.zeros(gray.shape, dtype=np.uint8)
        h, w = gray.shape

        classified_trails = []
        all_classifications = []

        # ── Background subtraction and noise estimation ─────────────
        if temporal_context is not None:
            diff_image = temporal_context['diff_image'].astype(np.float32)
            noise_map = temporal_context['noise_map'].astype(np.float32)
            noise_sigma = float(np.median(noise_map))
            residual = diff_image.copy()
        else:
            bg_kernel = min(51, max(3, min(h, w) // 8))
            if bg_kernel % 2 == 0:
                bg_kernel += 1
            bg = self._median_blur_gpu(gray, bg_kernel).astype(np.float32)
            residual = gray.astype(np.float32) - bg
            flat = residual.ravel()
            mad = np.median(np.abs(flat - np.median(flat)))
            noise_sigma = max(0.5, mad * 1.4826)

        # ── Star masking (pass pre-computed residual to skip redundant medianBlur) ─
        cleaned, star_mask = self._dual_threshold_star_mask(
            gray, noise_sigma, residual=residual)

        # ── Stage 1: LSD detection ──────────────────────────────────
        # LSD is the dominant bottleneck (~500ms at 1920px).  Downsample
        # to 960px wide — LSD detects line *segments* not individual
        # pixels, so half-res still finds all trails with negligible
        # accuracy loss while giving ~4x speedup.
        lsd_max_width = 960
        lsd_scale = 1.0
        if w > lsd_max_width:
            lsd_scale = float(lsd_max_width) / w
        lsd_gray = gray if lsd_scale >= 1.0 else cv2.resize(
            gray, (int(w * lsd_scale), int(h * lsd_scale)),
            interpolation=cv2.INTER_AREA)

        enhanced = self._clahe_lsd.apply(lsd_gray)

        lsd_segments = self._detect_lines_lsd(
            enhanced,
            min_length=self.params.get('satellite_min_length', 50) * 0.6 * lsd_scale,
            log_eps=self._lsd_log_eps if self._lsd_log_eps is not None else 0.0
        )

        # Cap LSD segments to prevent classify_trail explosion (4-8ms each).
        # Keep the most significant segments (already sorted by NFA).
        lsd_segments = lsd_segments[:50]

        # Merge collinear LSD fragments before classification — LSD often
        # splits a long dim trail into 2-5 pieces that would produce duplicates
        lsd_segments = self._merge_collinear_segments(lsd_segments)

        # Convert LSD results to HoughLinesP format, scaling back to full res
        lsd_lines = []
        for x1, y1, x2, y2, nfa in lsd_segments:
            lsd_lines.append(np.array([[int(round(x1 / lsd_scale)),
                                        int(round(y1 / lsd_scale)),
                                        int(round(x2 / lsd_scale)),
                                        int(round(y2 / lsd_scale))]]))

        for line in lsd_lines:
            trail_type, detection_info = self.classify_trail(
                line, gray, frame, hsv_frame, reusable_mask)

            if debug_info is not None:
                all_classifications.append({
                    'line': line, 'type': trail_type,
                    'detection_info': detection_info,
                    'bbox': detection_info['bbox'] if detection_info is not None else None,
                })

            if trail_type and detection_info:
                classified_trails.append((trail_type, detection_info))

        # ── Multi-frame residual accumulation (sequential mode only) ──
        # Accumulate star-subtracted residuals across frames to boost Radon
        # SNR by sqrt(N).  In parallel mode each worker has its own detector
        # so the buffer stays empty and we fall back to single-frame Radon.
        self._residual_buffer.append(cleaned)
        self._residual_star_masks.append(star_mask)
        if len(self._residual_buffer) > self._residual_buffer_depth:
            self._residual_buffer.pop(0)
            self._residual_star_masks.pop(0)

        if len(self._residual_buffer) >= 2:
            # Stack: mean of recent residuals with union star mask
            stacked = np.mean(self._residual_buffer, axis=0).astype(np.float32)
            combined_mask = self._residual_star_masks[0].copy()
            for m in self._residual_star_masks[1:]:
                combined_mask = combined_mask | m
            stacked[combined_mask] = 0.0
            radon_input = stacked
            # Noise reduces by sqrt(N) when averaging N frames
            radon_noise = noise_sigma / np.sqrt(len(self._residual_buffer))
        else:
            radon_input = cleaned
            radon_noise = noise_sigma

        # ── Stage 2: Radon transform detection ──────────────────────
        # Downsample: cap total pixel area to ~500k pixels (increased from
        # 250k to preserve 1-2px trails that were killed by 4x downsample)
        max_area = 500000  # ~700x700 for 1080p (was 250k / 500x500)
        current_area = h * w
        scale = min(1.0, np.sqrt(max_area / max(1, current_area)))
        small_h, small_w = max(64, int(h * scale)), max(64, int(w * scale))
        small_radon = cv2.resize(radon_input, (small_w, small_h),
                                 interpolation=cv2.INTER_AREA)
        small_noise = radon_noise * scale

        radon_candidates = self._detect_streaks_radon(
            small_radon, small_noise,
            snr_threshold=self.radon_snr_threshold,
            min_length=max(10, int(self.params.get('satellite_min_length', 50) * scale * 0.5))
        )

        # Scale candidates back to full resolution
        radon_candidates_full = []
        for x1, y1, x2, y2, snr in radon_candidates:
            radon_candidates_full.append(
                (x1 / scale, y1 / scale, x2 / scale, y2 / scale, snr))

        # ── Refine Radon endpoints on full-resolution residual ────
        # The sinogram FWHM gives a rough length estimate.  Walk along each
        # candidate line in the full-res cleaned image to find where the
        # signal actually drops to noise, giving accurate endpoints for
        # dedup, merge, and bounding box computation.
        refined_candidates = []
        for x1, y1, x2, y2, snr in radon_candidates_full:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx * dx + dy * dy)
            if length < 5:
                refined_candidates.append((x1, y1, x2, y2, snr))
                continue

            ux, uy = dx / length, dy / length
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Sample along the line at 1px steps from center outward
            max_half = int(length * 0.75)  # search up to 1.5x original estimate
            noise_floor = noise_sigma * 1.5  # signal threshold

            def _find_endpoint(direction):
                """Walk from center in +/- direction, return last above-noise point."""
                last_good = 0
                for step in range(1, max_half + 1):
                    px = int(round(cx + direction * step * ux))
                    py = int(round(cy + direction * step * uy))
                    if 0 <= py < h and 0 <= px < w:
                        if cleaned[py, px] > noise_floor:
                            last_good = step
                        elif step - last_good > 5:
                            break  # 5px gap → end of trail
                    else:
                        break
                return last_good

            half_fwd = _find_endpoint(1)
            half_bwd = _find_endpoint(-1)

            # Only use refined endpoints if they give a reasonable length
            ref_len = half_fwd + half_bwd
            min_len = self.params.get('satellite_min_length', 50) * scale * 0.4
            if ref_len >= min_len:
                rx1 = cx - half_bwd * ux
                ry1 = cy - half_bwd * uy
                rx2 = cx + half_fwd * ux
                ry2 = cy + half_fwd * uy
                rx1 = np.clip(rx1, 0, w - 1)
                ry1 = np.clip(ry1, 0, h - 1)
                rx2 = np.clip(rx2, 0, w - 1)
                ry2 = np.clip(ry2, 0, h - 1)
                refined_candidates.append((float(rx1), float(ry1),
                                           float(rx2), float(ry2), snr))
            else:
                refined_candidates.append((x1, y1, x2, y2, snr))

        radon_candidates_full = refined_candidates

        # ── Stage 3: Perpendicular cross filtering ──────────────────
        # Apply PCF to both LSD and Radon candidates (on the cleaned residual)
        # LSD candidates that already passed classify_trail are kept as-is;
        # apply PCF only to Radon candidates to suppress false positives.
        pcf_confirmed = self._perpendicular_cross_filter(
            cleaned, radon_candidates_full)

        # Deduplicate: remove Radon candidates that match existing LSD
        # detections using line proximity (angle + perpendicular distance),
        # far more robust than the old midpoint-in-AABB check.
        existing_lines = []
        for _, info in classified_trails:
            ln = info.get('line')
            if ln is not None:
                existing_lines.append(ln)
            else:
                # Fallback: use bbox corners as a pseudo-line
                b = info['bbox']
                existing_lines.append((b[0], b[1], b[2], b[3]))
        for x1, y1, x2, y2, snr in pcf_confirmed:
            radon_line = (x1, y1, x2, y2)
            is_duplicate = False
            for ex_line in existing_lines:
                if self._is_duplicate_line(radon_line, ex_line):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            # Convert to HoughLinesP format and classify
            line = np.array([[int(round(x1)), int(round(y1)),
                              int(round(x2)), int(round(y2))]])
            trail_type, detection_info = self.classify_trail(
                line, gray, frame, hsv_frame, reusable_mask,
                supplementary=True)

            if debug_info is not None:
                all_classifications.append({
                    'line': line, 'type': trail_type,
                    'detection_info': detection_info,
                    'bbox': detection_info['bbox'] if detection_info is not None else None,
                })

            if trail_type and detection_info:
                classified_trails.append((trail_type, detection_info))

        # Note: parent's matched filter is skipped — the Radon transform
        # subsumes it with better SNR sensitivity and the LSD covers the
        # bright-trail regime.  This saves ~2-4s per frame.

        # ── Store debug info ────────────────────────────────────────
        if debug_info is not None:
            _, edges = self.detect_lines(preprocessed)
            radon_lines = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in radon_candidates_full]
            debug_info['all_lines'] = [l for l in lsd_lines] + radon_lines
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray
            if temporal_context is not None:
                debug_info['temporal_context'] = temporal_context

        # ── Merge overlapping detections ────────────────────────────
        satellite_infos, airplane_infos = [], []
        for t, info in classified_trails:
            (satellite_infos if t == 'satellite' else airplane_infos).append(info)

        # Use oriented line-proximity merge for satellites (replaces AABB
        # merge which fails for diagonal trails — bloated AABBs cause false
        # merges of parallel trails and missed merges of collinear segments)
        merged_satellite_infos = self._merge_lines_oriented(satellite_infos)
        merged_airplane_infos = self.merge_airplane_detections(airplane_infos)

        # ── Cross-type dedup: same trail detected as both satellite & airplane
        # Keep the detection with the longer trail (more context = more reliable)
        if merged_satellite_infos and merged_airplane_infos:
            keep_airplanes = []
            for a_info in merged_airplane_infos:
                a_line = a_info.get('line', a_info['bbox'])
                is_dup = False
                for s_info in merged_satellite_infos:
                    s_line = s_info.get('line', s_info['bbox'])
                    if self._is_duplicate_line(a_line, s_line,
                                               dist_thresh=20, angle_thresh=10):
                        # Same trail — prefer the longer detection
                        if a_info.get('length', 0) > s_info.get('length', 0):
                            # Airplane detection is longer — keep airplane,
                            # remove the satellite (rare case)
                            merged_satellite_infos = [
                                s for s in merged_satellite_infos if s is not s_info]
                        # else: satellite is longer or equal — drop airplane
                        is_dup = True
                        break
                if not is_dup:
                    keep_airplanes.append(a_info)
            merged_airplane_infos = keep_airplanes

        # ── Enrich detections ───────────────────────────────────────
        frame_width = frame.shape[1]
        diff_img = temporal_context['diff_image'] if temporal_context else None

        all_merged = (
            [('satellite', info) for info in merged_satellite_infos] +
            [('airplane', info) for info in merged_airplane_infos]
        )
        # Skip expensive enrichment when many detections (same guard as parent)
        do_full_enrichment = len(all_merged) <= 20
        for trail_type, info in all_merged:
            line_arr = np.array([[info['line'][0], info['line'][1],
                                  info['line'][2], info['line'][3]]])
            if do_full_enrichment:
                info['photometry'] = self._analyze_streak_photometry(gray, line_arr)
                info['curvature'] = self._fit_trail_curvature(
                    gray, line_arr, diff_image=diff_img)
            else:
                info['photometry'] = None
                info['curvature'] = None
            info['velocity'] = self._estimate_angular_velocity(
                info['length'], frame_width,
                exposure_time=exposure_time, fov_degrees=fov_degrees)

        return all_merged


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Network Detector — model-based detection via _NNBackend
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralNetDetector(SatelliteTrailDetector):
    """Neural network-based satellite and airplane trail detector.

    Uses a trained object detection model (YOLOv8/v11, ONNX, etc.) for
    primary detection, with optional hybrid fallback to the classical
    pipeline for additional recall.

    Supports three inference backends:
      - **ultralytics**: Full YOLOv8/v11 API (auto-installed on first use)
      - **cv2dnn**: OpenCV DNN module — zero extra deps, supports ONNX
      - **onnxruntime**: ONNX Runtime — broad acceleration (TensorRT, etc.)

    Detection results are converted to the standard Mnemosky format so all
    downstream processing (freeze overlays, debug panels, dataset export,
    HITL review) works identically to the classical algorithms.

    Inherits drawing, debug visualisation, classification and enrichment
    methods from SatelliteTrailDetector.

    Usage::

        python satellite_trail_detector.py input.mp4 output.mp4 \\
            --algorithm nn --model trail_detector.pt

        python satellite_trail_detector.py input.mp4 output.mp4 \\
            --algorithm nn --model trail_detector.onnx --nn-backend cv2dnn
    """

    def __init__(self, sensitivity='medium', preprocessing_params=None,
                 skip_aspect_ratio_check=False, signal_envelope=None,
                 model_path=None, backend='ultralytics', device='auto',
                 confidence=0.25, nms_iou=0.45, input_size=640,
                 half_precision=False, class_map=None, hybrid_mode=False,
                 no_gpu=False):
        """
        Args:
            model_path: Path to model file (.pt, .onnx, .engine, etc.).
            backend: Inference backend ('ultralytics', 'cv2dnn', 'onnxruntime').
            device: 'auto', 'cpu', or GPU index string (e.g. '0').
            confidence: Detection confidence threshold (0-1).
            nms_iou: NMS IoU threshold (0-1).
            input_size: Square input resolution for the model.
            half_precision: Use FP16 inference (GPU only).
            class_map: Mapping of trail types to model class IDs,
                e.g. ``{'satellite': [0], 'airplane': [1]}``.
            hybrid_mode: Also run classical pipeline and merge results.
            no_gpu: Force CPU inference even if GPU is available.
        """
        super().__init__(sensitivity, preprocessing_params,
                         skip_aspect_ratio_check, signal_envelope)

        self.model_path = model_path
        self.backend_name = backend
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.input_size = input_size
        self.half_precision = half_precision
        self.class_map = class_map or {'satellite': [0], 'airplane': [1]}
        self.hybrid_mode = hybrid_mode
        self._no_gpu = no_gpu

        # Invert class_map for fast lookup: model_class_id → trail_type
        self._class_id_to_type = {}
        for trail_type, class_ids in self.class_map.items():
            for cid in class_ids:
                self._class_id_to_type[cid] = trail_type

        # Deferred backend initialisation (supports multiprocessing — the
        # _NNBackend object is NOT pickled; each worker recreates it on
        # first detect_trails() call).
        self._backend = None
        self._backend_config = {
            'model_path': model_path,
            'backend': backend,
            'device': device,
            'confidence': confidence,
            'nms_iou': nms_iou,
            'input_size': input_size,
            'half_precision': half_precision,
            'no_gpu': no_gpu,
        }

    def _ensure_backend(self):
        """Lazy-initialise the inference backend on first use."""
        if self._backend is None:
            self._backend = _NNBackend(**self._backend_config)

    # ── Post-hoc detection_info from model bbox ──────────────────────

    def _bbox_to_detection_info(self, bbox, gray, color_frame, nn_confidence):
        """Convert an NN-predicted bounding box to a standard detection_info dict.

        Estimates trail line, angle, center, length, and brightness metrics
        from the bounding box region of the grayscale frame.

        Returns:
            detection_info dict, or None if bbox is degenerate.
        """
        x_min, y_min, x_max, y_max = bbox
        bw = x_max - x_min
        bh = y_max - y_min
        if bw < 3 or bh < 3:
            return None

        fh, fw = gray.shape
        center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        # Estimate trail line from bbox major axis
        if bw >= bh:
            line = (x_min, (y_min + y_max) // 2, x_max, (y_min + y_max) // 2)
            angle = np.degrees(np.arctan2(bh, bw)) % 180
        else:
            line = ((x_min + x_max) // 2, y_min, (x_min + x_max) // 2, y_max)
            angle = (90.0 + np.degrees(np.arctan2(bw, bh))) % 180

        length = float(np.sqrt(bw ** 2 + bh ** 2))

        # Measure brightness within the bbox
        roi = gray[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None
        avg_brightness = float(np.mean(roi))
        max_brightness = int(np.max(roi))
        brightness_std = float(np.std(roi))

        # Contrast ratio: trail ROI vs surrounding background
        pad = max(10, int(length * 0.1))
        bg_x1, bg_y1 = max(0, x_min - pad), max(0, y_min - pad)
        bg_x2, bg_y2 = min(fw, x_max + pad), min(fh, y_max + pad)
        bg_roi = gray[bg_y1:bg_y2, bg_x1:bg_x2]
        bg_mean = float(np.median(bg_roi)) if bg_roi.size > 0 else 1.0
        contrast_ratio = avg_brightness / max(1.0, bg_mean)

        # Colour analysis
        color_roi = color_frame[y_min:y_max, x_min:x_max]
        if color_roi.size > 0:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            avg_saturation = float(np.mean(hsv[:, :, 1]))
        else:
            avg_saturation = 0.0

        return {
            'bbox': bbox,
            'angle': angle,
            'center': center,
            'length': length,
            'avg_brightness': avg_brightness,
            'max_brightness': max_brightness,
            'line': line,
            'contrast_ratio': contrast_ratio,
            'brightness_std': brightness_std,
            'avg_saturation': avg_saturation,
            'has_dotted_pattern': False,
            'nn_confidence': nn_confidence,
        }

    # ── Hybrid merge ─────────────────────────────────────────────────

    def _merge_nn_classical(self, nn_trails, classical_trails):
        """Merge NN and classical detections, preferring NN on overlap.

        For each classical detection whose bbox centre falls inside an
        existing NN bbox, the classical detection is dropped (the NN
        version is kept since it has an explicit class prediction).
        Classical detections that do not overlap are added (the model
        may have missed them).
        """
        if not classical_trails:
            return nn_trails
        if not nn_trails:
            return classical_trails

        nn_bboxes = [info['bbox'] for _, info in nn_trails]
        merged = list(nn_trails)
        for trail_type, info in classical_trails:
            cx = (info['bbox'][0] + info['bbox'][2]) / 2.0
            cy = (info['bbox'][1] + info['bbox'][3]) / 2.0
            overlap = False
            for nb in nn_bboxes:
                if nb[0] <= cx <= nb[2] and nb[1] <= cy <= nb[3]:
                    overlap = True
                    break
            if not overlap:
                merged.append((trail_type, info))
        return merged

    # ── Main detection pipeline ──────────────────────────────────────

    def detect_trails(self, frame, debug_info=None, temporal_context=None,
                      exposure_time=13.0, fov_degrees=None):
        """Neural network detection pipeline.

        1. Run model inference → bounding boxes + class predictions
        2. Map model class IDs to satellite/airplane trail types
        3. Compute post-hoc detection_info from bboxes
        4. Optionally merge with classical pipeline results (hybrid mode)
        5. Merge overlapping detections
        6. Enrich with photometry, curvature, velocity

        Args / Returns: Same as ``SatelliteTrailDetector.detect_trails()``.
        """
        self._ensure_backend()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]

        # ── Stage 1: Model inference ─────────────────────────────────
        raw_detections = self._backend.predict(frame)

        # ── Stage 2: Convert to standard format ──────────────────────
        classified_trails = []
        all_classifications = []

        for det in raw_detections:
            trail_type = self._class_id_to_type.get(det['class_id'])
            if trail_type is None:
                continue

            x1 = max(0, int(round(det['bbox'][0])))
            y1 = max(0, int(round(det['bbox'][1])))
            x2 = min(fw, int(round(det['bbox'][2])))
            y2 = min(fh, int(round(det['bbox'][3])))
            bbox = (x1, y1, x2, y2)

            detection_info = self._bbox_to_detection_info(
                bbox, gray, frame, det['confidence'])

            if debug_info is not None:
                all_classifications.append({
                    'line': np.array([[x1, y1, x2, y2]]),
                    'type': trail_type,
                    'detection_info': detection_info,
                    'bbox': bbox,
                    'nn_confidence': det['confidence'],
                })

            if detection_info is not None:
                classified_trails.append((trail_type, detection_info))

        # ── Stage 3: Optional hybrid merge with classical pipeline ───
        if self.hybrid_mode:
            classical = super().detect_trails(
                frame, debug_info=None,
                temporal_context=temporal_context,
                exposure_time=exposure_time,
                fov_degrees=fov_degrees)
            classified_trails = self._merge_nn_classical(
                classified_trails, classical)

        # ── Stage 4: Merge overlapping detections ────────────────────
        satellite_infos = [info for t, info in classified_trails
                          if t == 'satellite']
        airplane_infos = [info for t, info in classified_trails
                         if t == 'airplane']

        satellite_boxes = [info['bbox'] for info in satellite_infos]
        merged_satellite_boxes = self.merge_overlapping_boxes(satellite_boxes)

        merged_satellite_infos = []
        for mbox in merged_satellite_boxes:
            mx = (mbox[0] + mbox[2]) / 2.0
            my = (mbox[1] + mbox[3]) / 2.0
            best, best_dist = None, float('inf')
            for info in satellite_infos:
                dx = info['center'][0] - mx
                dy = info['center'][1] - my
                d = dx * dx + dy * dy
                if d < best_dist:
                    best_dist, best = d, info
            if best:
                merged_info = dict(best)
                merged_info['bbox'] = mbox
                merged_satellite_infos.append(merged_info)

        merged_airplane_infos = self.merge_airplane_detections(airplane_infos)

        # ── Stage 5: Enrich with photometry, curvature, velocity ─────
        frame_width = frame.shape[1]
        diff_img = (temporal_context['diff_image']
                    if temporal_context else None)

        all_merged = (
            [('satellite', info) for info in merged_satellite_infos] +
            [('airplane', info) for info in merged_airplane_infos]
        )
        do_full = len(all_merged) <= 20
        for trail_type, info in all_merged:
            line_arr = np.array([[info['line'][0], info['line'][1],
                                  info['line'][2], info['line'][3]]])
            if do_full:
                info['photometry'] = self._analyze_streak_photometry(
                    gray, line_arr)
                info['curvature'] = self._fit_trail_curvature(
                    gray, line_arr, diff_image=diff_img)
            else:
                info['photometry'] = None
                info['curvature'] = None
            info['velocity'] = self._estimate_angular_velocity(
                info['length'], frame_width,
                exposure_time=exposure_time, fov_degrees=fov_degrees)

        # ── Debug info ───────────────────────────────────────────────
        if debug_info is not None:
            debug_info['all_classifications'] = all_classifications
            debug_info['gray_frame'] = gray
            debug_info['edges'] = None
            debug_info['all_lines'] = []
            debug_info['nn_raw_detections'] = raw_detections

        return all_merged


def _apply_nn_params(detector, np_dict):
    """Apply NN preview parameters to a NeuralNetDetector instance.

    Only updates lightweight scalar parameters; does NOT reload the model.
    """
    if 'confidence' in np_dict:
        detector.confidence = np_dict['confidence']
        if detector._backend is not None:
            detector._backend.confidence = np_dict['confidence']
    if 'nms_iou' in np_dict:
        detector.nms_iou = np_dict['nms_iou']
        if detector._backend is not None:
            detector._backend.nms_iou = np_dict['nms_iou']
    if 'input_size' in np_dict:
        detector.input_size = np_dict['input_size']
        if detector._backend is not None:
            detector._backend.input_size = np_dict['input_size']
    if 'class_map' in np_dict:
        detector.class_map = np_dict['class_map']
        detector._class_id_to_type = {}
        for ttype, cids in np_dict['class_map'].items():
            for cid in cids:
                detector._class_id_to_type[cid] = ttype


# ── Multiprocessing worker for parallel frame detection ────────────────

def _apply_radon_preview_params(detector, rp):
    """Apply Radon preview parameters to a RadonStreakDetector instance."""
    if 'radon_snr_threshold' in rp:
        detector.radon_snr_threshold = rp['radon_snr_threshold']
    if 'pcf_ratio_threshold' in rp:
        detector.pcf_ratio_threshold = rp['pcf_ratio_threshold']
    if 'pcf_kernel_len' in rp:
        detector.pcf_kernel_length = rp['pcf_kernel_len']
    if 'star_mask_sigma' in rp:
        detector._star_mask_sigma = rp['star_mask_sigma']
    if 'lsd_log_eps' in rp:
        detector._lsd_log_eps = rp['lsd_log_eps']
    if 'min_streak_length' in rp:
        detector.params['satellite_min_length'] = rp['min_streak_length']


_worker_detector = None


def _worker_init(algorithm, sensitivity, preprocessing_params,
                 skip_aspect_ratio_check, signal_envelope, groundtruth_dir,
                 use_gpu, radon_params=None, nn_params=None,
                 gt_profiles=None):
    """Create a detector instance in each worker process.

    When gt_profiles is provided, skips the expensive _calibrate_from_groundtruth()
    (which loads and analyzes all PNG files) and directly applies the pre-computed
    calibration.  This saves ~1-2s per worker at startup.
    """
    global _worker_detector
    if algorithm == 'nn' and nn_params:
        _worker_detector = NeuralNetDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope,
            model_path=nn_params['model_path'],
            backend=nn_params['backend'],
            device='cpu' if not use_gpu else nn_params.get('device', 'auto'),
            confidence=nn_params['confidence'],
            nms_iou=nn_params['nms_iou'],
            input_size=nn_params['input_size'],
            half_precision=nn_params.get('half_precision', False),
            class_map=nn_params.get('class_map'),
            hybrid_mode=nn_params.get('hybrid_mode', False),
            no_gpu=not use_gpu,
        )
    elif algorithm == 'radon':
        # Pass groundtruth_dir=None to skip per-worker calibration;
        # apply the pre-computed profiles instead.
        _worker_detector = RadonStreakDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope,
            groundtruth_dir=None)
        if gt_profiles is not None:
            _worker_detector.gt_profiles = gt_profiles
            _worker_detector._apply_gt_calibration(gt_profiles)
        if radon_params:
            _apply_radon_preview_params(_worker_detector, radon_params)
    else:
        _worker_detector = SatelliteTrailDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope)
    _worker_detector._use_gpu = use_gpu
    _worker_detector._use_gpu_filter = use_gpu
    _worker_detector._use_gpu_warp = use_gpu
    _worker_detector._use_gpu_median = use_gpu


def _worker_detect(args):
    """Run detection on a single frame in a worker process."""
    frame, temporal_context, need_debug, exposure_time, fov_degrees = args
    debug_info = {} if need_debug else None
    detected_trails = _worker_detector.detect_trails(
        frame, debug_info=debug_info,
        temporal_context=temporal_context,
        exposure_time=exposure_time,
        fov_degrees=fov_degrees)
    return detected_trails, debug_info


# ═══════════════════════════════════════════════════════════════════════════════
# ML Dataset Export — utility functions and DatasetExporter class
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_obb_corners(x1, y1, x2, y2, half_width=8, pad_along=10):
    """Compute 4 corners of an oriented bounding box around a line segment.

    Returns corners as [(cx1,cy1), (cx2,cy2), (cx3,cy3), (cx4,cy4)]
    in clockwise order from the 'top-left' of the oriented box.

    Args:
        x1, y1, x2, y2: Line endpoint coordinates.
        half_width: Perpendicular half-width of the box in pixels.
        pad_along: Padding added along the line direction in pixels.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return [(x1, y1)] * 4

    # Unit vectors along and perpendicular to the trail
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # perpendicular (rotated 90 degrees CCW)

    # Four corners: start-side then end-side, clockwise
    return [
        (x1 - ux * pad_along + nx * half_width,
         y1 - uy * pad_along + ny * half_width),
        (x2 + ux * pad_along + nx * half_width,
         y2 + uy * pad_along + ny * half_width),
        (x2 + ux * pad_along - nx * half_width,
         y2 + uy * pad_along - ny * half_width),
        (x1 - ux * pad_along - nx * half_width,
         y1 - uy * pad_along - ny * half_width),
    ]


def _trail_to_polygon(x1, y1, x2, y2, half_width, frame_w, frame_h):
    """Return normalized polygon coordinates for a trail segment.

    Returns a flat list [x1, y1, x2, y2, x3, y3, x4, y4] with values
    clamped to [0, 1], suitable for YOLO segment format.
    """
    corners = _compute_obb_corners(x1, y1, x2, y2, half_width)
    normalized = []
    for cx, cy in corners:
        normalized.append(max(0.0, min(1.0, cx / frame_w)))
        normalized.append(max(0.0, min(1.0, cy / frame_h)))
    return normalized


def _compute_phash(frame_gray, hash_size=8):
    """Compute a simple perceptual hash using downscaled median threshold.

    Returns a compact byte array (hash_size * hash_size / 8 bytes).
    No external dependencies beyond OpenCV and NumPy.
    """
    small = cv2.resize(frame_gray, (hash_size, hash_size),
                       interpolation=cv2.INTER_AREA)
    median = np.median(small)
    return np.packbits((small > median).flatten())


def _hamming_distance(h1, h2):
    """Hamming distance between two perceptual hash byte arrays."""
    return int(np.unpackbits(np.bitwise_xor(h1, h2)).sum())


class DatasetExporter:
    """Manages ML dataset export with format support, splitting, and dedup.

    Supports four annotation formats:
        aabb    — Standard YOLO axis-aligned bounding boxes
        obb     — YOLO OBB (oriented bounding boxes) with 4 corner points
        segment — YOLO instance segmentation polygons
        coco    — COCO JSON format (bbox + optional segmentation)

    Features:
        - Train/val/test splitting with temporal-episode grouping to prevent
          data leakage from near-identical consecutive frames.
        - Perceptual-hash-based near-duplicate filtering.
        - Configurable frame skip interval.
        - Negative (background) sample export for hard-negative mining.
        - Dataset statistics and health report.
    """

    # Default OBB half-width covers 3-5px trail mask + margin
    DEFAULT_HALF_WIDTH = 8

    def __init__(self, dataset_dir, video_stem, frame_w, frame_h, fps,
                 fmt='aabb', split_ratios=(0.7, 0.2, 0.1),
                 skip_frames=0, dedup_threshold=5, negative_ratio=0.2,
                 image_format='jpg', image_quality=95, seed=42,
                 freeze_duration=1.0):
        self.dataset_dir = Path(dataset_dir)
        self.video_stem = video_stem
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.fmt = fmt
        self.split_ratios = tuple(split_ratios)
        self.skip_frames = skip_frames
        self.dedup_threshold = dedup_threshold
        self.negative_ratio = negative_ratio
        self.image_format = image_format
        self.image_quality = image_quality
        self.seed = seed
        self.freeze_duration = freeze_duration

        # Episode gap: frames further apart than this start a new episode
        self._episode_gap = max(1, int(freeze_duration * fps)) + 2

        # State
        self._entries = []           # (frame_idx, img_name, label_lines, dominant_class)
        self._negative_entries = []  # (frame_idx, img_name)
        self._last_export_frame = -9999
        self._recent_hashes = deque(maxlen=30)
        self._positive_count = 0
        self._negative_count = 0
        self._class_counts = {'satellite': 0, 'airplane': 0}
        # For bbox statistics
        self._bbox_widths = []
        self._bbox_heights = []

        # Create staging directories
        (self.dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)

    # ── Frame filtering ──────────────────────────────────────────────

    def should_export_frame(self, frame_idx, frame_gray):
        """Check skip interval and perceptual-hash dedup.

        Returns True if the frame should be exported.
        """
        # Skip interval check
        if self.skip_frames > 0:
            if (frame_idx - self._last_export_frame) < self.skip_frames:
                return False

        # Perceptual hash dedup check
        if self.dedup_threshold > 0:
            h = _compute_phash(frame_gray)
            for prev_h in self._recent_hashes:
                if _hamming_distance(h, prev_h) < self.dedup_threshold:
                    return False
            self._recent_hashes.append(h)

        return True

    # ── Positive frame export ────────────────────────────────────────

    def export_frame(self, frame_idx, frame, detected_trails):
        """Export one frame with its detections as image + label file."""
        ext = 'png' if self.image_format == 'png' else 'jpg'
        img_name = f"{self.video_stem}_f{frame_idx:06d}.{ext}"

        # Write image
        img_path = self.dataset_dir / 'images' / img_name
        if ext == 'png':
            cv2.imwrite(str(img_path), frame)
        else:
            cv2.imwrite(str(img_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

        # Build label lines
        label_lines = []
        per_frame_class = {'satellite': 0, 'airplane': 0}

        for trail_type, det_info in detected_trails:
            cls_id = 0 if trail_type == 'satellite' else 1
            per_frame_class[trail_type] += 1

            if self.fmt == 'obb':
                line_pts = det_info.get('line')
                if line_pts:
                    lx1, ly1, lx2, ly2 = line_pts
                else:
                    bx0, by0, bx1, by1 = det_info['bbox']
                    lx1, ly1, lx2, ly2 = bx0, (by0 + by1) / 2, bx1, (by0 + by1) / 2
                corners = _compute_obb_corners(lx1, ly1, lx2, ly2,
                                               self.DEFAULT_HALF_WIDTH)
                coords = []
                for cx, cy in corners:
                    coords.append(max(0.0, min(1.0, cx / self.frame_w)))
                    coords.append(max(0.0, min(1.0, cy / self.frame_h)))
                label_lines.append(
                    f"{cls_id} " + " ".join(f"{c:.6f}" for c in coords))

            elif self.fmt == 'segment':
                line_pts = det_info.get('line')
                if line_pts:
                    lx1, ly1, lx2, ly2 = line_pts
                else:
                    bx0, by0, bx1, by1 = det_info['bbox']
                    lx1, ly1, lx2, ly2 = bx0, (by0 + by1) / 2, bx1, (by0 + by1) / 2
                poly = _trail_to_polygon(lx1, ly1, lx2, ly2,
                                         self.DEFAULT_HALF_WIDTH,
                                         self.frame_w, self.frame_h)
                label_lines.append(
                    f"{cls_id} " + " ".join(f"{c:.6f}" for c in poly))

            else:  # aabb (default)
                bx0, by0, bx1, by1 = det_info['bbox']
                xc = ((bx0 + bx1) / 2) / self.frame_w
                yc = ((by0 + by1) / 2) / self.frame_h
                bw = (bx1 - bx0) / self.frame_w
                bh = (by1 - by0) / self.frame_h
                label_lines.append(
                    f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                self._bbox_widths.append(bw)
                self._bbox_heights.append(bh)

            self._class_counts[trail_type] += 1

            # Track bbox stats for obb/segment too (use AABB of the corners)
            if self.fmt in ('obb', 'segment'):
                bx0, by0, bx1, by1 = det_info['bbox']
                self._bbox_widths.append((bx1 - bx0) / self.frame_w)
                self._bbox_heights.append((by1 - by0) / self.frame_h)

        # Write label file
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = self.dataset_dir / 'labels' / lbl_name
        label_path.write_text('\n'.join(label_lines) + '\n')

        # Track for splitting
        dominant = max(per_frame_class, key=per_frame_class.get)
        self._entries.append((frame_idx, img_name, label_lines, dominant))
        self._last_export_frame = frame_idx
        self._positive_count += 1

    # ── Negative frame export ────────────────────────────────────────

    def maybe_export_negative(self, frame_idx, frame, frame_gray):
        """Conditionally export a frame with no detections as a negative sample."""
        if self.negative_ratio <= 0:
            return
        # Export enough negatives to maintain the target ratio
        target = int(self._positive_count * self.negative_ratio)
        if self._negative_count >= target:
            return
        if not self.should_export_frame(frame_idx, frame_gray):
            return

        ext = 'png' if self.image_format == 'png' else 'jpg'
        img_name = f"{self.video_stem}_f{frame_idx:06d}.{ext}"

        img_path = self.dataset_dir / 'images' / img_name
        if ext == 'png':
            cv2.imwrite(str(img_path), frame)
        else:
            cv2.imwrite(str(img_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

        # Empty label file
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        (self.dataset_dir / 'labels' / lbl_name).write_text('')

        self._negative_entries.append((frame_idx, img_name))
        self._negative_count += 1

    # ── Finalization ─────────────────────────────────────────────────

    def finalize(self):
        """Run train/val/test split, reorganize files, write configs, print stats."""
        total_entries = len(self._entries) + len(self._negative_entries)
        if total_entries == 0:
            print("\nNo detections found — dataset directory is empty.")
            return

        # Build and split episodes
        episodes = self._build_episodes()
        splits = self._split_episodes(episodes)

        # Reorganize files into split subdirectories
        split_counts = {}
        split_class_counts = {}
        split_negative_counts = {}
        for split_name in ('train', 'val', 'test'):
            ep_list = splits.get(split_name, [])
            if not ep_list:
                split_counts[split_name] = 0
                split_class_counts[split_name] = {'satellite': 0, 'airplane': 0}
                split_negative_counts[split_name] = 0
                continue

            split_img_dir = self.dataset_dir / split_name / 'images'
            split_lbl_dir = self.dataset_dir / split_name / 'labels'
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            cls_counts = {'satellite': 0, 'airplane': 0}
            neg_count = 0
            for episode in ep_list:
                for frame_idx, img_name, label_lines, dom_class in episode:
                    src_img = self.dataset_dir / 'images' / img_name
                    lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
                    src_lbl = self.dataset_dir / 'labels' / lbl_name
                    if src_img.exists():
                        os.rename(str(src_img),
                                  str(split_img_dir / img_name))
                    if src_lbl.exists():
                        os.rename(str(src_lbl),
                                  str(split_lbl_dir / lbl_name))
                    count += 1
                    if dom_class == 'background':
                        neg_count += 1
                    else:
                        # Count annotations from label lines
                        for line in label_lines:
                            parts = line.strip().split()
                            if parts:
                                cid = int(parts[0])
                                if cid == 0:
                                    cls_counts['satellite'] += 1
                                else:
                                    cls_counts['airplane'] += 1

            split_counts[split_name] = count
            split_class_counts[split_name] = cls_counts
            split_negative_counts[split_name] = neg_count

        # Remove empty staging directories
        for subdir in ('images', 'labels'):
            d = self.dataset_dir / subdir
            if d.exists():
                try:
                    if not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    pass

        # Write data.yaml
        self._write_data_yaml(split_counts)

        # Write COCO JSON if format is coco
        if self.fmt == 'coco':
            self._export_coco_json(splits)

        # Print statistics report
        self._print_statistics(split_counts, split_class_counts,
                               split_negative_counts)

    # ── Episode building ─────────────────────────────────────────────

    def _build_episodes(self):
        """Group frames into temporal episodes for leakage-free splitting.

        Consecutive frames within ``_episode_gap`` of each other belong to the
        same episode.  All frames in an episode are assigned to the same split.
        """
        # Merge positive and negative entries, sorted by frame index
        all_entries = []
        for entry in self._entries:
            all_entries.append(entry)  # (frame_idx, img_name, label_lines, dom_class)
        for frame_idx, img_name in self._negative_entries:
            all_entries.append((frame_idx, img_name, [], 'background'))

        all_entries.sort(key=lambda x: x[0])

        episodes = []
        current = []
        for entry in all_entries:
            if current and (entry[0] - current[-1][0]) > self._episode_gap:
                episodes.append(current)
                current = []
            current.append(entry)
        if current:
            episodes.append(current)

        return episodes

    def _split_episodes(self, episodes):
        """Assign episodes to train/val/test with class stratification."""
        rng = random.Random(self.seed)

        # Classify each episode by its dominant positive class
        sat_eps, air_eps, neg_eps = [], [], []
        for ep in episodes:
            classes = [e[3] for e in ep if e[3] != 'background']
            if not classes:
                neg_eps.append(ep)
            elif classes.count('satellite') >= classes.count('airplane'):
                sat_eps.append(ep)
            else:
                air_eps.append(ep)

        def _assign(eps_list):
            rng.shuffle(eps_list)
            n = len(eps_list)
            n_train = max(1, round(n * self.split_ratios[0])) if n > 0 else 0
            n_val = max(0, round(n * self.split_ratios[1])) if n > 1 else 0
            # Guard: don't exceed total
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            return (eps_list[:n_train],
                    eps_list[n_train:n_train + n_val],
                    eps_list[n_train + n_val:])

        splits = {'train': [], 'val': [], 'test': []}
        for group in (sat_eps, air_eps, neg_eps):
            if not group:
                continue
            tr, va, te = _assign(group)
            splits['train'].extend(tr)
            splits['val'].extend(va)
            splits['test'].extend(te)

        return splits

    # ── data.yaml generation ─────────────────────────────────────────

    def _write_data_yaml(self, split_counts):
        """Write a YOLOv8-compatible data.yaml with task type and metadata."""
        task_map = {'aabb': 'detect', 'obb': 'obb', 'segment': 'segment',
                    'coco': 'detect'}
        task = task_map.get(self.fmt, 'detect')

        lines = ["path: .\n"]
        if split_counts.get('train', 0) > 0:
            lines.append("train: train/images\n")
        if split_counts.get('val', 0) > 0:
            lines.append("val: val/images\n")
        if split_counts.get('test', 0) > 0:
            lines.append("test: test/images\n")
        lines.append(f"\ntask: {task}\n")
        lines.append("\nnames:\n  0: satellite\n  1: airplane\n")

        # Metadata comment block
        total_imgs = sum(split_counts.values())
        total_ann = self._class_counts['satellite'] + self._class_counts['airplane']
        lines.append(f"\n# Mnemosky dataset metadata\n")
        lines.append(f"# resolution: {self.frame_w}x{self.frame_h}\n")
        lines.append(f"# total_images: {total_imgs}\n")
        lines.append(f"# total_annotations: {total_ann}\n")
        lines.append(f"# satellites: {self._class_counts['satellite']}\n")
        lines.append(f"# airplanes: {self._class_counts['airplane']}\n")
        lines.append(f"# negative_samples: {self._negative_count}\n")
        lines.append(f"# created: {datetime.now(timezone.utc).isoformat()}\n")

        yaml_path = self.dataset_dir / 'data.yaml'
        yaml_path.write_text(''.join(lines))

    # ── COCO JSON export ─────────────────────────────────────────────

    def _export_coco_json(self, splits):
        """Write per-split COCO-format annotation JSON files."""
        for split_name in ('train', 'val', 'test'):
            ep_list = splits.get(split_name, [])
            if not ep_list:
                continue

            coco = {
                'info': {
                    'description': f'Mnemosky {split_name} dataset',
                    'version': '1.0',
                    'year': datetime.now().year,
                    'contributor': 'Mnemosky',
                    'date_created': datetime.now(timezone.utc).isoformat(),
                },
                'licenses': [],
                'categories': [
                    {'id': 0, 'name': 'satellite', 'supercategory': 'trail'},
                    {'id': 1, 'name': 'airplane', 'supercategory': 'trail'},
                ],
                'images': [],
                'annotations': [],
            }

            img_id = 0
            ann_id = 0
            for episode in ep_list:
                for frame_idx, img_name, label_lines, dom_class in episode:
                    img_id += 1
                    coco['images'].append({
                        'id': img_id,
                        'file_name': img_name,
                        'width': self.frame_w,
                        'height': self.frame_h,
                        'frame_index': frame_idx,
                    })

                    for lbl_line in label_lines:
                        parts = lbl_line.strip().split()
                        if not parts:
                            continue
                        ann_id += 1
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]

                        if self.fmt in ('aabb', 'coco') and len(coords) == 4:
                            xc, yc, bw, bh = coords
                            # Denormalize to pixel COCO bbox [x, y, w, h]
                            px = (xc - bw / 2) * self.frame_w
                            py = (yc - bh / 2) * self.frame_h
                            pw = bw * self.frame_w
                            ph = bh * self.frame_h
                            bbox = [round(px, 1), round(py, 1),
                                    round(pw, 1), round(ph, 1)]
                            area = pw * ph
                            seg = []
                        elif len(coords) == 8:
                            # OBB or segment — 4 corners normalized
                            xs = [coords[i] * self.frame_w for i in range(0, 8, 2)]
                            ys = [coords[i] * self.frame_h for i in range(1, 8, 2)]
                            px, py = min(xs), min(ys)
                            pw, ph = max(xs) - px, max(ys) - py
                            bbox = [round(px, 1), round(py, 1),
                                    round(pw, 1), round(ph, 1)]
                            area = pw * ph
                            # Segmentation as polygon
                            seg_pts = []
                            for i in range(4):
                                seg_pts.extend([round(xs[i], 1),
                                                round(ys[i], 1)])
                            seg = [seg_pts]
                        else:
                            continue

                        coco['annotations'].append({
                            'id': ann_id,
                            'image_id': img_id,
                            'category_id': cls_id,
                            'bbox': bbox,
                            'area': round(area, 1),
                            'segmentation': seg,
                            'iscrowd': 0,
                        })

            json_path = self.dataset_dir / split_name / 'annotations.json'
            with open(json_path, 'w') as f:
                json.dump(coco, f, indent=2)

    # ── Statistics report ────────────────────────────────────────────

    def _print_statistics(self, split_counts, split_class_counts,
                          split_negative_counts):
        """Print a dataset summary with class balance and health warnings."""
        total_imgs = sum(split_counts.values())
        total_ann = self._class_counts['satellite'] + self._class_counts['airplane']
        sat = self._class_counts['satellite']
        air = self._class_counts['airplane']

        fmt_label = {'aabb': 'YOLOv8', 'obb': 'YOLOv8-OBB',
                     'segment': 'YOLOv8-Seg', 'coco': 'COCO JSON'}

        print(f"\n{'=' * 50}")
        print(f"  ML Dataset Summary")
        print(f"{'─' * 50}")
        print(f"  Format:       {fmt_label.get(self.fmt, self.fmt)}")
        print(f"  Location:     {self.dataset_dir}")
        print(f"  Resolution:   {self.frame_w}x{self.frame_h}")
        print(f"  Images:       {total_imgs}")
        print(f"  Annotations:  {total_ann} "
              f"({sat} satellite, {air} airplane)")
        print(f"  Negatives:    {self._negative_count}")

        # Split breakdown
        print(f"\n  Split Breakdown:")
        for s in ('train', 'val', 'test'):
            cnt = split_counts.get(s, 0)
            if cnt == 0:
                continue
            sc = split_class_counts.get(s, {})
            nc = split_negative_counts.get(s, 0)
            s_sat = sc.get('satellite', 0)
            s_air = sc.get('airplane', 0)
            pct = 100 * cnt / total_imgs if total_imgs else 0
            print(f"    {s:5s}: {cnt:5d} images ({pct:4.1f}%)  "
                  f"[sat={s_sat}, air={s_air}, neg={nc}]")

        # BBox statistics
        if self._bbox_widths:
            ws = np.array(self._bbox_widths)
            hs = np.array(self._bbox_heights)
            print(f"\n  BBox Statistics (normalized):")
            print(f"    Width:  min={ws.min():.4f}  max={ws.max():.4f}  "
                  f"mean={ws.mean():.4f}  std={ws.std():.4f}")
            print(f"    Height: min={hs.min():.4f}  max={hs.max():.4f}  "
                  f"mean={hs.mean():.4f}  std={hs.std():.4f}")

        # Warnings
        warnings = []
        if sat > 0 and air > 0:
            ratio = max(sat, air) / min(sat, air)
            if ratio > 3.0:
                majority = 'satellite' if sat > air else 'airplane'
                warnings.append(
                    f"Class imbalance: {majority} has {ratio:.1f}x more "
                    f"annotations than the other class")
        if self._negative_count == 0 and self.negative_ratio > 0:
            warnings.append(
                "No negative samples exported (all frames had detections)")
        if total_ann > 0 and total_imgs > 0:
            ann_per_img = total_ann / (total_imgs - self._negative_count) \
                if (total_imgs - self._negative_count) > 0 else 0
            if ann_per_img > 5:
                warnings.append(
                    f"High annotation density ({ann_per_img:.1f}/image) — "
                    f"check for false positives")

        if warnings:
            print(f"\n  Warnings:")
            for w in warnings:
                print(f"    * {w}")

        print(f"\n  Class map:    0=satellite, 1=airplane")
        print(f"{'=' * 50}")


def export_dataset_from_annotations(input_path, annotations_path,
                                    dataset_dir_override=None,
                                    output_path=None,
                                    fmt='aabb',
                                    split_ratios=(0.7, 0.2, 0.1),
                                    skip_frames=1,
                                    dedup_threshold=0,
                                    negative_ratio=0.0,
                                    image_format='jpg',
                                    image_quality=95,
                                    freeze_duration=1.0):
    """Export a dataset from an existing HITL-verified annotation file.

    Only confirmed detections (and user-drawn missed annotations) are included.
    This bypasses full video processing and produces the highest-quality labels
    from human-reviewed data.

    Args:
        input_path: Path to the original video file.
        annotations_path: Path to the Mnemosky annotation JSON file.
        dataset_dir_override: Explicit dataset output directory (optional).
        output_path: Fallback for deriving the dataset directory name.
        fmt: Annotation format ('aabb', 'obb', 'segment', 'coco').
        split_ratios: Train/val/test split ratios.
        skip_frames: Frame skip interval (default 1 = no skip).
        dedup_threshold: Perceptual hash dedup threshold (default 0 = off).
        negative_ratio: Ratio of negative samples (default 0).
        image_format: Image format ('jpg' or 'png').
        image_quality: JPEG quality (1-100).
        freeze_duration: Used for episode gap calculation.
    """
    # Load annotation database
    with open(annotations_path, 'r') as f:
        ann_data = json.load(f)

    categories = {c['id']: c['name'] for c in ann_data.get('categories', [])}
    images_by_id = {img['id']: img for img in ann_data.get('images', [])}

    # Collect confirmed annotations grouped by image_id
    confirmed = {}  # image_id -> list of (trail_type, det_info)
    for ann in ann_data.get('annotations', []):
        ext = ann.get('mnemosky_ext', {})
        if ext.get('status') != 'confirmed':
            continue
        img_id = ann['image_id']
        cat_name = categories.get(ann['category_id'], 'satellite')
        # Convert COCO bbox [x, y, w, h] to internal (x_min, y_min, x_max, y_max)
        bx, by, bw, bh = ann['bbox']
        bbox = (bx, by, bx + bw, by + bh)
        # Reconstruct detection_info from annotation metadata
        meta = ext.get('detection_meta', {})
        det_info = {
            'bbox': bbox,
            'angle': meta.get('angle', 0),
            'center': tuple(meta.get('center', ((bx + bw / 2), (by + bh / 2)))),
            'length': meta.get('length', math.sqrt(bw * bw + bh * bh)),
            'avg_brightness': meta.get('avg_brightness', 0),
            'max_brightness': meta.get('max_brightness', 0),
            'line': tuple(meta.get('line', (bx, by + bh / 2, bx + bw, by + bh / 2))),
        }
        confirmed.setdefault(img_id, []).append((cat_name, det_info))

    # Also include missed annotations (user-drawn false negatives)
    for missed in ann_data.get('missed_annotations', []):
        img_id = missed['image_id']
        cat_name = categories.get(missed['category_id'], 'satellite')
        bx, by, bw, bh = missed['bbox']
        bbox = (bx, by, bx + bw, by + bh)
        det_info = {
            'bbox': bbox,
            'angle': 0,
            'center': (bx + bw / 2, by + bh / 2),
            'length': math.sqrt(bw * bw + bh * bh),
            'avg_brightness': 0,
            'max_brightness': 0,
            'line': (bx, by + bh / 2, bx + bw, by + bh / 2),
        }
        confirmed.setdefault(img_id, []).append((cat_name, det_info))

    if not confirmed:
        print("No confirmed annotations found in the annotation file.")
        return

    # Determine frame indices to extract
    frame_indices = {}
    for img_id, trails in confirmed.items():
        img_info = images_by_id.get(img_id)
        if img_info and 'frame_index' in img_info:
            frame_indices[img_info['frame_index']] = (img_id, trails)

    if not frame_indices:
        print("No frame indices found in the annotation database images.")
        return

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up DatasetExporter
    if dataset_dir_override:
        ds_dir = dataset_dir_override
    elif output_path:
        out_p = Path(output_path)
        ds_dir = str(out_p.parent / (out_p.stem + '_dataset'))
    else:
        ds_dir = str(Path(input_path).with_suffix('') / '_hitl_dataset')

    exporter = DatasetExporter(
        dataset_dir=ds_dir,
        video_stem=Path(input_path).stem,
        frame_w=width, frame_h=height, fps=fps,
        fmt=fmt,
        split_ratios=split_ratios,
        skip_frames=skip_frames,
        dedup_threshold=dedup_threshold,
        negative_ratio=negative_ratio,
        image_format=image_format,
        image_quality=image_quality,
        freeze_duration=freeze_duration,
    )

    fmt_label = {'aabb': 'YOLOv8', 'obb': 'YOLOv8-OBB',
                 'segment': 'YOLOv8-Seg', 'coco': 'COCO JSON'}
    print(f"Exporting HITL-verified dataset from {annotations_path}")
    print(f"  Format: {fmt_label.get(fmt, fmt)}")
    print(f"  Confirmed annotations: {sum(len(v) for v in confirmed.values())}")
    print(f"  Frames to extract: {len(frame_indices)}")
    print(f"  Output: {ds_dir}")

    sorted_frames = sorted(frame_indices.keys())
    exported = 0

    for target_frame in sorted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        img_id, trails = frame_indices[target_frame]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if exporter.should_export_frame(target_frame, gray):
            exporter.export_frame(target_frame, frame, trails)
            exported += 1

        if exported % 10 == 0:
            print(f"\r  Exported {exported}/{len(sorted_frames)} frames",
                  end="", flush=True)

    cap.release()
    print(f"\r  Exported {exported}/{len(sorted_frames)} frames")

    exporter.finalize()


def process_video(input_path, output_path, sensitivity='medium', freeze_duration=1.0, max_duration=None, detect_type='both', show_labels=True, debug_mode=False, debug_only=False, preprocessing_params=None, skip_aspect_ratio_check=False, signal_envelope=None, save_dataset=False, exposure_time=13.0, fov_degrees=None, temporal_buffer_size=7, algorithm='default', groundtruth_dir=None, num_workers=0, no_gpu=False, review_mode=False, review_only=False, annotations_path=None, hitl_profile='default', auto_accept=0.9, no_learn=False, dataset_format='aabb', dataset_split=(0.7, 0.2, 0.1), dataset_skip=0, dataset_dedup=5, dataset_negatives=0.2, dataset_image_format='jpg', dataset_image_quality=95, dataset_dir_override=None, radon_params=None, nn_params=None, enable_ledger=False, loss_profile='balanced', observer_context=None):
    """
    Process video to detect and highlight satellite and airplane trails.

    Output video maintains the same resolution and frame rate as input.
    Uses MPEG-4 (mp4v) codec by default for broad compatibility.

    Args:
        input_path: Path to input MP4 video
        output_path: Path to output MP4 video
        sensitivity: Detection sensitivity ('low', 'medium', 'high')
        freeze_duration: How long to freeze frame when trail detected (seconds)
        max_duration: Maximum duration to process in seconds (None = process entire video)
        detect_type: What to detect - 'both', 'satellites', or 'airplanes' (default: 'both')
        show_labels: Whether to show labels on bounding boxes (default: True)
        debug_mode: If True, creates side-by-side view with debug visualization (default: False)
        debug_only: If True, outputs ONLY debug visualization without normal output (default: False)
        preprocessing_params: Optional dict with custom preprocessing parameters from preview
        skip_aspect_ratio_check: If True, disables aspect ratio filtering (default: False)
        signal_envelope: Optional dict with signal characteristics from user-marked trail
            examples. Used to dynamically adapt detection thresholds.
        save_dataset: If True, save detections as a YOLO-format ML dataset (default: False)
        exposure_time: Exposure time per frame in seconds (default: 13.0).
            Used for angular velocity estimation and streak photometry.
        fov_degrees: Horizontal field of view in degrees (optional).
            Enables angular velocity output in degrees/second.
        temporal_buffer_size: Number of frames in the temporal rolling buffer
            (default: 7).  The temporal median of this many surrounding frames
            is used as a reference background — stars, vignetting, and sky
            gradients are removed perfectly, leaving only transient trails.
            Set to 0 to disable temporal integration.
        dataset_format: Annotation format for dataset export ('aabb', 'obb',
            'segment', 'coco'). Default: 'aabb'.
        dataset_split: Train/val/test split ratios as a 3-tuple. Default: (0.7, 0.2, 0.1).
        dataset_skip: Export every Nth frame (0 = auto fps/2). Default: 0.
        dataset_dedup: Perceptual hash dedup threshold (0 = off). Default: 5.
        dataset_negatives: Negative sample ratio relative to positives. Default: 0.2.
        dataset_image_format: Image format for export ('jpg' or 'png'). Default: 'jpg'.
        dataset_image_quality: JPEG quality 1-100. Default: 95.
        dataset_dir_override: Explicit dataset output directory (optional).
    """
    # Validate input
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_path}")
        sys.exit(1)
    
    # Get video properties (guard against zero/invalid values)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = float('inf')  # unknown length — process until EOF

    # Adjust output dimensions for debug mode (side-by-side or debug-only)
    if debug_only:
        output_width = width
        output_height = height
        debug_mode = True  # Enable debug collection
    elif debug_mode:
        output_width = width * 2
        output_height = height
    else:
        output_width = width
        output_height = height

    print(f"Input video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    print(f"Sensitivity: {sensitivity}")
    print(f"Freeze duration: {freeze_duration}s")
    print(f"Detection filter: {detect_type}")
    print(f"Show labels: {show_labels}")
    if debug_only:
        print(f"Debug mode: debug only")
    else:
        print(f"Debug mode: {debug_mode}")

    # Calculate maximum frames to process
    if max_duration is not None:
        max_frames = int(fps * max_duration)
        print(f"Processing limit: {max_duration}s ({max_frames} frames)")
    else:
        max_frames = total_frames

    # Calculate freeze frames
    freeze_frame_count = int(fps * freeze_duration)

    # Initialize video writer with codec
    # Try mp4v (MPEG-4) first for broad compatibility, fall back to H.264 variants
    codecs_to_try = [
        ('mp4v', 'MPEG-4'),
        ('avc1', 'H.264'),
        ('h264', 'H.264'),
        ('H264', 'H.264'),
        ('X264', 'H.264')
    ]

    out = None
    used_codec = None

    for codec_name, codec_desc in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        test_out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        if test_out.isOpened():
            out = test_out
            used_codec = f"{codec_name} ({codec_desc})"
            print(f"Using codec: {used_codec}")
            break
        test_out.release()

    if out is None:
        # Last resort - try with -1 to let system choose
        fourcc = -1
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))
        used_codec = "system default"
        print(f"Using codec: {used_codec}")
    
    if out is None or not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        print("Please ensure you have the necessary codecs installed.")
        cap.release()
        if out:
            out.release()
        sys.exit(1)
    
    # Initialize detector (used in sequential mode; parallel mode creates
    # per-worker detectors via _worker_init)
    if algorithm == 'nn' and nn_params:
        detector = NeuralNetDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope,
            model_path=nn_params['model_path'],
            backend=nn_params['backend'],
            device='cpu' if no_gpu else nn_params.get('device', 'auto'),
            confidence=nn_params['confidence'],
            nms_iou=nn_params['nms_iou'],
            input_size=nn_params['input_size'],
            half_precision=nn_params.get('half_precision', False),
            class_map=nn_params.get('class_map'),
            hybrid_mode=nn_params.get('hybrid_mode', False),
            no_gpu=no_gpu,
        )
        backend_label = nn_params['backend']
        model_label = Path(nn_params['model_path']).name
        print(f"Algorithm: Neural Network ({backend_label}, model={model_label})")
        if nn_params.get('hybrid_mode'):
            print(f"  Hybrid mode: classical pipeline also active")
    elif algorithm == 'radon':
        detector = RadonStreakDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope,
            groundtruth_dir=groundtruth_dir)
        if radon_params:
            _apply_radon_preview_params(detector, radon_params)
        print(f"Algorithm: Radon + LSD + PCF (advanced)")
    else:
        detector = SatelliteTrailDetector(
            sensitivity, preprocessing_params=preprocessing_params,
            skip_aspect_ratio_check=skip_aspect_ratio_check,
            signal_envelope=signal_envelope)

    # ── STS: Translation Ledger (Callon) ──────────────────────────
    # Tracks what the detector rejected and why, making the filtering
    # assumptions transparent and auditable.
    if enable_ledger:
        detector.ledger = TranslationLedger()
        print("Translation ledger: active (tracks rejection statistics)")

    # ── STS: Observer Context (Haraway — situated knowledges) ─────
    # Record the observation conditions so the detector doesn't claim
    # to see from nowhere.  Bortle class adjusts contrast thresholds.
    if observer_context:
        bortle = observer_context.get('bortle_class')
        if bortle is not None and isinstance(bortle, (int, float)):
            # Bortle 1 (pristine dark) → multiplier ~0.85 (lower contrast needed)
            # Bortle 5 (suburban) → multiplier ~1.0 (default)
            # Bortle 9 (inner city) → multiplier ~1.15 (raise contrast bar)
            bortle_multiplier = 0.85 + (bortle - 1) * 0.0375
            old_contrast = detector.params.get('satellite_contrast_min', 1.08)
            new_contrast = round(old_contrast * bortle_multiplier, 4)
            # Clamp to safety bounds
            new_contrast = max(1.01, min(1.30, new_contrast))
            detector.params['satellite_contrast_min'] = new_contrast
            print(f"Observer context: Bortle {bortle} → contrast threshold "
                  f"adjusted {old_contrast:.3f} → {new_contrast:.3f}")
        ctx_parts = []
        if observer_context.get('lat') is not None:
            ctx_parts.append(f"({observer_context['lat']:.2f}, {observer_context['lon']:.2f})")
        if observer_context.get('notes'):
            ctx_parts.append(observer_context['notes'])
        if ctx_parts:
            print(f"Observer: {', '.join(ctx_parts)}")
    detector._observer_context = observer_context

    # ── Temporal frame buffer for background subtraction ──────────
    # The temporal median of N surrounding frames removes stars, sky
    # gradients, vignetting, and hot pixels — leaving only transient
    # features (trails).  This is the single biggest SNR improvement.
    temporal_buffer = None
    if temporal_buffer_size >= 5:
        temporal_buffer = TemporalFrameBuffer(capacity=temporal_buffer_size)
        print(f"Temporal integration: {temporal_buffer_size}-frame rolling buffer")
    else:
        print("Temporal integration: disabled")
    print(f"Exposure time: {exposure_time}s")
    if fov_degrees:
        print(f"Field of view: {fov_degrees}°")

    # ── Temporal detection tracker (Radon pipeline only) ──────────
    # Filters out single-frame noise and builds tracklets across frames.
    # Runs post-detection in the main process — works with both sequential
    # and parallel modes.
    detection_tracker = None
    if algorithm == 'radon':
        detection_tracker = DetectionTracker(
            window=4, min_hits=2, angle_thresh=10, dist_thresh=30)
        print("Temporal detection tracker: active (window=4, min_hits=2)")

    frame_count = 0
    satellites_detected = 0
    airplanes_detected = 0
    anomalous_detected = 0

    # ── HITL review mode: collect detections per frame ────────────
    detections_by_frame = {} if review_mode else None

    # ── ML dataset setup ───────────────────────────────────────────
    dataset_exporter = None
    if save_dataset:
        out_p = Path(output_path)
        ds_dir = dataset_dir_override or str(
            out_p.parent / (out_p.stem + '_dataset'))
        auto_skip = max(1, int(fps / 2)) if dataset_skip == 0 else dataset_skip
        dataset_exporter = DatasetExporter(
            dataset_dir=ds_dir,
            video_stem=Path(input_path).stem,
            frame_w=width, frame_h=height, fps=fps,
            fmt=dataset_format,
            split_ratios=dataset_split,
            skip_frames=auto_skip,
            dedup_threshold=dataset_dedup,
            negative_ratio=dataset_negatives,
            image_format=dataset_image_format,
            image_quality=dataset_image_quality,
            freeze_duration=freeze_duration,
        )
        fmt_label = {'aabb': 'YOLOv8', 'obb': 'YOLOv8-OBB',
                     'segment': 'YOLOv8-Seg', 'coco': 'COCO JSON'}
        print(f"Dataset export enabled → {ds_dir} "
              f"(format={fmt_label.get(dataset_format, dataset_format)})")

    # Track frozen regions: list of (frozen_region, bbox, trail_type, frames_remaining)
    frozen_regions = []

    # Track debug panels for detections (only in debug mode)
    # Each entry: {'panel': image, 'frames_remaining': int}
    debug_panels = []
    debug_panel_duration = int(fps * 2)  # 2 seconds

    # ── Parallel worker pool setup ─────────────────────────────────
    pool = None
    if num_workers >= 1:
        use_gpu_workers = _HAS_CUDA and not no_gpu
        # Pass pre-computed GT calibration to workers to avoid N redundant
        # _calibrate_from_groundtruth() runs (each loads all PNG files).
        gt_profiles = getattr(detector, 'gt_profiles', None)
        # Warn about GPU contention with multiple NN workers
        if algorithm == 'nn' and num_workers > 1 and use_gpu_workers:
            print(f"Warning: --algorithm nn with {num_workers} GPU workers "
                  f"may cause VRAM contention. Consider --workers 1 for GPU "
                  f"or --no-gpu for CPU multi-worker parallelism.")
        pool = multiprocessing.Pool(
            num_workers,
            initializer=_worker_init,
            initargs=(algorithm, sensitivity, preprocessing_params,
                      skip_aspect_ratio_check, signal_envelope,
                      groundtruth_dir, use_gpu_workers, radon_params,
                      nn_params, gt_profiles))
        pending = deque()  # (frame, frame_idx, AsyncResult)
        prefetch = num_workers * 2
        print(f"Parallel workers: {num_workers}" +
              (f" + CUDA" if use_gpu_workers else ""))
    else:
        if no_gpu:
            detector._use_gpu = False
        print(f"Workers: sequential" +
              (f" + CUDA" if detector._use_gpu else ""))

    print("\nProcessing video...")

    # ── Unified frame-result generator ─────────────────────────────
    # Abstracts sequential vs parallel detection so the post-processing
    # loop below is shared.  Yields (frame, frame_idx, detected_trails,
    # debug_info) tuples in frame order.
    def _frame_results():
        nonlocal frame_count

        if pool is not None:
            # ── Parallel path: read-ahead + async detection ────────
            try:
                reading_done = False
                while True:
                    # Fill the pipeline up to prefetch depth
                    while not reading_done and len(pending) < prefetch:
                        ret, frame = cap.read()
                        if not ret:
                            reading_done = True
                            break
                        frame_count += 1
                        if frame_count > max_frames:
                            reading_done = True
                            break

                        # Feed temporal buffer (must be sequential)
                        temporal_context = None
                        if temporal_buffer is not None:
                            gray_fb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            temporal_buffer.add(gray_fb)
                            if temporal_buffer.is_ready():
                                temporal_context = \
                                    temporal_buffer.get_temporal_context(gray_fb)

                        # Copy temporal context arrays for the worker process
                        tc_copy = None
                        if temporal_context is not None:
                            tc_copy = {
                                'diff_image': temporal_context['diff_image'].copy(),
                                'noise_map': temporal_context['noise_map'].copy(),
                                'reference': temporal_context['reference'].copy(),
                                'buffer_depth': temporal_context['buffer_depth'],
                            }

                        async_result = pool.apply_async(
                            _worker_detect,
                            ((frame.copy(), tc_copy, debug_mode,
                              exposure_time, fov_degrees),))
                        pending.append((frame, frame_count, async_result))

                    if not pending:
                        break

                    orig_frame, fidx, async_result = pending.popleft()
                    try:
                        det_trails, dbg_info = async_result.get(timeout=300)
                    except Exception as e:
                        print(f"\r  [Worker error on frame {fidx}: {e}]")
                        det_trails, dbg_info = [], {} if debug_mode else None
                    yield orig_frame, fidx, det_trails, dbg_info
            finally:
                pool.terminate()
                pool.join()

        else:
            # ── Sequential path (original logic) ──────────────────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if frame_count > max_frames:
                    break

                temporal_context = None
                if temporal_buffer is not None:
                    gray_fb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    temporal_buffer.add(gray_fb)
                    if temporal_buffer.is_ready():
                        temporal_context = \
                            temporal_buffer.get_temporal_context(gray_fb)

                dbg_info = {} if debug_mode else None
                det_trails = detector.detect_trails(
                    frame, debug_info=dbg_info,
                    temporal_context=temporal_context,
                    exposure_time=exposure_time,
                    fov_degrees=fov_degrees)
                yield frame, frame_count, det_trails, dbg_info

    for frame, fc, detected_trails, debug_info in _frame_results():
        # Progress indicator
        if fc % 30 == 0:
            if max_frames != float('inf'):
                progress = (fc / max_frames) * 100
                print(f"\rProgress: {progress:.1f}% ({fc}/{max_frames})",
                      end="", flush=True)
            else:
                print(f"\rProcessed: {fc} frames",
                      end="", flush=True)

        # Start with current frame
        output_frame = frame.copy()

        # Filter trails based on detect_type parameter
        if detect_type == 'satellites':
            detected_trails = [(t, b) for t, b in detected_trails if t == 'satellite']
        elif detect_type == 'airplanes':
            detected_trails = [(t, b) for t, b in detected_trails if t == 'airplane']
        elif detect_type == 'anomalous':
            detected_trails = [(t, b) for t, b in detected_trails if t == 'anomalous']
        # If detect_type == 'both' or 'all', no filtering needed

        # Apply temporal consistency filter (Radon pipeline only).
        # Requires detections to appear in multiple recent frames to confirm,
        # suppressing single-frame noise.  Also builds tracklets.
        if detection_tracker is not None:
            detected_trails = detection_tracker.update(fc, detected_trails)

        # Collect detections for HITL review mode
        # fc is 1-based frame_count; store as 0-based index for cap.set()
        if detections_by_frame is not None and detected_trails:
            detections_by_frame[fc - 1] = list(detected_trails)

        if detected_trails:
            # Count detections by type and add new frozen regions
            for trail_type, detection_info in detected_trails:
                bbox = detection_info['bbox']

                if trail_type == 'satellite':
                    satellites_detected += 1
                elif trail_type == 'airplane':
                    airplanes_detected += 1
                elif trail_type == 'anomalous':
                    anomalous_detected += 1

                # Log enrichment metadata for this detection
                _meta_parts = [f"f{fc}",
                               f"{trail_type}",
                               f"L={detection_info['length']:.0f}px"]
                vel = detection_info.get('velocity')
                if vel:
                    _meta_parts.append(f"{vel['px_per_sec']:.1f}px/s")
                    _meta_parts.append(vel['orbit_class'])
                phot = detection_info.get('photometry')
                if phot:
                    _meta_parts.append(f"lc={phot['classification']}")
                curv = detection_info.get('curvature')
                if curv and curv['is_curved']:
                    _meta_parts.append(f"curved({curv['curvature']:.2e})")
                # Epistemic profile: show detection path (Latour)
                ep = detection_info.get('epistemic_profile')
                if ep and ep.get('detection_path'):
                    _meta_parts.append(f"via:{ep['detection_path']}")
                print(f"\r  [{' | '.join(_meta_parts)}]" + " " * 20)

                # Extract the region to freeze (with highlights)
                x_min, y_min, x_max, y_max = bbox

                # Create a frame with the highlight
                highlighted_frame = frame.copy()
                detector.draw_highlight(highlighted_frame, trail_type, bbox, show_label=show_labels)

                # Extract the frozen region (expanded to include dotted lines and labels)
                # Need extra padding to capture labels which can extend ~40px above/below the box
                # Use less padding if labels are hidden
                padding = 50 if show_labels else 20
                freeze_x_min = max(0, x_min - padding)
                freeze_y_min = max(0, y_min - padding)
                freeze_x_max = min(frame.shape[1], x_max + padding)
                freeze_y_max = min(frame.shape[0], y_max + padding)

                frozen_region = highlighted_frame[freeze_y_min:freeze_y_max, freeze_x_min:freeze_x_max].copy()
                freeze_bbox = (freeze_x_min, freeze_y_min, freeze_x_max, freeze_y_max)

                # Add to frozen regions list with full detection metadata
                frozen_regions.append({
                    'region': frozen_region,
                    'bbox': freeze_bbox,
                    'trail_type': trail_type,
                    'detection_info': detection_info,
                    'frames_remaining': freeze_frame_count
                })

                # Create debug panel for this detection (if in debug mode)
                if debug_mode and debug_info and 'edges' in debug_info and 'gray_frame' in debug_info and debug_info.get('all_lines') is not None:
                    # Find the corresponding line for this detection
                    for classification in debug_info['all_classifications']:
                        if classification['type'] == trail_type and classification['bbox'] == bbox:
                            line = classification['line']
                            panel = detector.create_detection_debug_panel(
                                frame, line, trail_type, debug_info['edges'], debug_info['gray_frame']
                            )
                            debug_panels.append({
                                'panel': panel,
                                'frames_remaining': debug_panel_duration
                            })
                            break  # Only create one panel per detection

            # ── ML dataset export (positive frame) ─────────────────────
            if dataset_exporter is not None:
                gray_for_hash = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if dataset_exporter.should_export_frame(fc, gray_for_hash):
                    dataset_exporter.export_frame(fc, frame, detected_trails)
        else:
            # ── ML dataset export (negative / background frame) ────────
            if dataset_exporter is not None:
                gray_for_hash = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dataset_exporter.maybe_export_negative(fc, frame, gray_for_hash)

        # Apply all active frozen regions to the output frame
        active_regions = []
        for frozen_data in frozen_regions:
            if frozen_data['frames_remaining'] > 0:
                # Overlay the frozen region
                x_min, y_min, x_max, y_max = frozen_data['bbox']
                region = frozen_data['region']

                # Ensure region fits (handle edge cases)
                frame_h, frame_w = output_frame.shape[:2]

                # Adjust if region extends beyond frame boundaries
                x_min_c = max(0, x_min)
                y_min_c = max(0, y_min)
                actual_y_max = min(y_max, frame_h)
                actual_x_max = min(x_max, frame_w)
                actual_region_h = actual_y_max - y_min_c
                actual_region_w = actual_x_max - x_min_c

                if actual_region_h > 0 and actual_region_w > 0:
                    output_frame[y_min_c:actual_y_max, x_min_c:actual_x_max] = region[:actual_region_h, :actual_region_w]

                # Decrement frames remaining
                frozen_data['frames_remaining'] -= 1
                active_regions.append(frozen_data)

        # Keep only active frozen regions
        frozen_regions = active_regions

        # Create final output frame (side-by-side if debug mode, debug only if debug_only)
        if debug_only:
            # Output ONLY the debug visualization
            final_frame = detector.create_debug_frame(frame, debug_info)
        elif debug_mode:
            # Create debug visualization
            debug_frame = detector.create_debug_frame(frame, debug_info)

            # Combine output_frame and debug_frame side by side
            final_frame = np.hstack([output_frame, debug_frame])
        else:
            final_frame = output_frame

        # Overlay debug panels if any are active (only in debug or debug_only mode)
        # Place them on the final frame after it's been constructed
        if debug_mode and debug_panels:
            active_panels = []
            # Position panels in top-right corner with some spacing
            y_offset = 10
            x_offset = final_frame.shape[1] - 160  # 10 pixels from right edge

            for panel_data in debug_panels:
                if panel_data['frames_remaining'] > 0:
                    panel = panel_data['panel']
                    panel_h, panel_w = panel.shape[:2]

                    # Ensure panel fits in frame
                    if y_offset + panel_h < final_frame.shape[0] and x_offset + panel_w <= final_frame.shape[1]:
                        # Overlay panel on frame
                        final_frame[y_offset:y_offset + panel_h, x_offset:x_offset + panel_w] = panel
                        y_offset += panel_h + 10  # Add spacing between panels

                    # Decrement frames remaining
                    panel_data['frames_remaining'] -= 1
                    active_panels.append(panel_data)

            # Keep only active panels
            debug_panels = active_panels

        # Write the output frame
        out.write(final_frame)
    
    # Cleanup
    cap.release()
    out.release()

    print(f"\n\nProcessing complete!")
    print(f"Frames processed: {frame_count}")

    if detect_type in ['both', 'all', 'satellites']:
        print(f"Satellites detected: {satellites_detected}")
    if detect_type in ['both', 'all', 'airplanes']:
        print(f"Airplanes detected: {airplanes_detected}")
    if anomalous_detected > 0:
        print(f"Anomalous detected: {anomalous_detected}")

    print(f"Total trails detected: {satellites_detected + airplanes_detected + anomalous_detected}")
    print(f"Output saved to: {output_path}")

    # ── STS: Translation Ledger summary (Callon) ─────────────────
    if detector.ledger is not None:
        print()
        for line in detector.ledger.summary_lines():
            print(line)

    # ── ML dataset finalisation ──────────────────────────────────
    if dataset_exporter is not None:
        dataset_exporter.finalize()

    # ── HITL Review Mode ──────────────────────────────────────────
    if review_mode or review_only:
        if annotations_path:
            ann_path = Path(annotations_path)
        else:
            ann_path = Path(output_path).with_suffix('.json')

        if review_only and ann_path.exists():
            ann_db = AnnotationDatabase(ann_path)
        else:
            ann_db = AnnotationDatabase()

        # Initialize parameter adapter
        param_adapter = None
        if not no_learn:
            param_adapter = ParameterAdapter(detector.params, PARAMETER_SAFETY_BOUNDS,
                                              loss_profile=loss_profile)
            if hitl_profile:
                param_adapter.load_profile(hitl_profile)

        # Populate annotations from detection results
        if not review_only and detections_by_frame:
            ann_db.start_session(str(input_path), sensitivity, algorithm, detector.params,
                                   observer_context=observer_context, loss_profile=loss_profile)

            for frame_idx, detections in detections_by_frame.items():
                img_id = ann_db.add_image(frame_idx, str(input_path), width, height)
                for trail_type, det_info in detections:
                    cat_id = AnnotationDatabase.CATEGORY_ID.get(trail_type, 0)
                    confidence = ParameterAdapter.compute_confidence(det_info, detector.params)
                    ann_db.add_detection(img_id, cat_id, det_info['bbox'],
                                         det_info, detector.params, confidence)

            # Auto-accept high-confidence detections
            if auto_accept < 1.0:
                auto_accepted = 0
                for ann in ann_db.data['annotations']:
                    if (ann['mnemosky_ext']['status'] == 'pending' and
                            ann['mnemosky_ext'].get('confidence', 0) >= auto_accept):
                        ann_db.record_correction(ann['id'], 'accept')
                        auto_accepted += 1
                if auto_accepted:
                    print(f"Auto-accepted {auto_accepted} high-confidence detections (>= {auto_accept:.2f})")

        # Launch review UI
        print(f"\nLaunching review UI with {len(ann_db.data['annotations'])} annotations...")
        review_ui = ReviewUI(
            str(input_path), detections_by_frame or {}, detector,
            ann_db, param_adapter,
            auto_accept_threshold=auto_accept,
        )
        review_ui.run()

        # After review: save
        ann_db.end_session(param_adapter.get_params() if param_adapter else None)
        ann_db.save(ann_path)

        if param_adapter and hitl_profile:
            param_adapter.save_profile(hitl_profile)

        print(f"\nAnnotations saved to: {ann_path}")
        if param_adapter:
            print(f"Learned parameters saved to profile: {hitl_profile}")

        # Print review summary
        cal_set = ann_db.get_calibration_set()
        confirmed = sum(1 for a in ann_db.data['annotations'] if a['mnemosky_ext']['status'] == 'confirmed')
        rejected = sum(1 for a in ann_db.data['annotations'] if a['mnemosky_ext']['status'] == 'rejected')
        missed_count = len(ann_db.data['missed_annotations'])
        print(f"\n{'=' * 42}")
        print(f"  Review Session Summary")
        print(f"{'-' * 42}")
        print(f"  Confirmed: {confirmed}")
        print(f"  Rejected:  {rejected}")
        print(f"  Missed:    {missed_count}")
        print(f"  Calibration set: {len(cal_set)} entries")
        print(f"{'=' * 42}")


def main():
    parser = argparse.ArgumentParser(
        description="Detect and classify satellite and airplane trails in MP4 videos. Output preserves input quality and resolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python satellite_trail_detector.py input.mp4 output.mp4
    python satellite_trail_detector.py night_sky.mp4 analyzed.mp4 --sensitivity high
    python satellite_trail_detector.py timelapse.mp4 result.mp4 --freeze-duration 2.0
    python satellite_trail_detector.py long_video.mp4 test.mp4 --max-duration 30
    python satellite_trail_detector.py video.mp4 satellites_only.mp4 --detect-type satellites
    python satellite_trail_detector.py video.mp4 airplanes_only.mp4 --detect-type airplanes
    python satellite_trail_detector.py video.mp4 clean_output.mp4 --no-labels
    python satellite_trail_detector.py video.mp4 debug_output.mp4 --debug
    python satellite_trail_detector.py video.mp4 debug_viz.mp4 --debug-only
    python satellite_trail_detector.py video.mp4 output.mp4 --preview

Notes:
    - Output video maintains same resolution and quality as input
    - Uses MPEG-4 (mp4v) codec by default for broad compatibility
    - Satellites: GOLD boxes - smooth, uniform trails (180-300px)
    - Airplanes: ORANGE boxes - dotted/point-like bright features (any length)
    - Detection parameters optimized for 1920x1080 resolution
    - Key distinction: Airplanes have bright point features, satellites are smooth
    - Use --max-duration to process only the first N seconds (useful for testing)
    - Use --detect-type to filter for only satellites or only airplanes
    - Use --no-labels to hide text labels (shows only colored boxes)
    - Use --debug to create side-by-side view showing all detected lines (2x width output)
    - Use --debug-only to output ONLY the debug visualization (same width as input)
    - Use --preview to interactively tune preprocessing parameters before processing
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to input MP4 video file'
    )
    
    parser.add_argument(
        'output',
        help='Path to output MP4 video file'
    )
    
    parser.add_argument(
        '--sensitivity', '-s',
        choices=['low', 'medium', 'high'],
        default='medium',
        help='Detection sensitivity (default: medium)'
    )
    
    parser.add_argument(
        '--freeze-duration', '-f',
        type=float,
        default=1.0,
        help='Duration in seconds to freeze frame when trail detected (default: 1.0)'
    )

    parser.add_argument(
        '--max-duration', '-d',
        type=float,
        default=None,
        help='Maximum duration in seconds to process from start of video (default: process entire video)'
    )

    parser.add_argument(
        '--detect-type', '-t',
        choices=['both', 'all', 'satellites', 'airplanes', 'anomalous'],
        default='both',
        help='What to detect: "both" (default, sat+airplane), "all" (sat+airplane+anomalous), '
             '"satellites" only, "airplanes" only, or "anomalous" only'
    )

    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide labels on detection boxes (only show colored boxes)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: creates side-by-side view showing all detected lines and classifications'
    )

    parser.add_argument(
        '--debug-only',
        action='store_true',
        help='Output ONLY debug visualization (no normal output, no side-by-side)'
    )

    parser.add_argument(
        '--preview', '-p',
        action='store_true',
        help='Show interactive preprocessing preview to tune CLAHE, blur, and edge detection parameters before processing'
    )

    parser.add_argument(
        '--no-aspect-ratio-check',
        action='store_true',
        help='Disable aspect ratio filtering (may improve performance but increase false positives)'
    )

    parser.add_argument(
        '--dataset',
        action='store_true',
        help='Save detections as a YOLO-format ML dataset (images + labels) alongside the output video'
    )

    parser.add_argument(
        '--dataset-format',
        choices=['aabb', 'obb', 'segment', 'coco'],
        default='aabb',
        help='Dataset annotation format: aabb (axis-aligned bbox, default), '
             'obb (oriented bbox — ideal for thin trails), '
             'segment (instance segmentation polygon), '
             'coco (COCO JSON with bbox + segmentation)'
    )

    parser.add_argument(
        '--dataset-split',
        nargs=3, type=float, default=[0.7, 0.2, 0.1],
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='Train/val/test split ratios (default: 0.7 0.2 0.1). '
             'Uses temporal-episode grouping to prevent data leakage.'
    )

    parser.add_argument(
        '--dataset-skip',
        type=int, default=0,
        help='Export every Nth frame with detections. '
             '0 = auto (fps/2, i.e. one frame per 0.5s). '
             '1 = export all frames (no skip).'
    )

    parser.add_argument(
        '--dataset-dedup',
        type=int, default=5,
        help='Perceptual hash dedup threshold (Hamming distance 0-64). '
             '0 = disable dedup. Default: 5.'
    )

    parser.add_argument(
        '--dataset-negatives',
        type=float, default=0.2,
        help='Ratio of negative (no-detection) samples to include relative to '
             'positive samples. 0 = no negatives. Default: 0.2'
    )

    parser.add_argument(
        '--dataset-image-format',
        choices=['jpg', 'png'], default='jpg',
        help='Image format for dataset export (default: jpg)'
    )

    parser.add_argument(
        '--dataset-image-quality',
        type=int, default=95,
        help='JPEG quality for dataset images, 1-100 (default: 95). '
             'Ignored when --dataset-image-format is png.'
    )

    parser.add_argument(
        '--dataset-dir',
        type=str, default=None,
        help='Explicit dataset output directory (overrides default '
             '<output_stem>_dataset/)'
    )

    parser.add_argument(
        '--dataset-from-annotations',
        type=str, default=None,
        help='Export dataset from an existing HITL-verified annotation file '
             '(only confirmed detections). Bypasses video processing.'
    )

    parser.add_argument(
        '--exposure-time',
        type=float,
        default=13.0,
        help='Exposure time per frame in seconds (default: 13.0). Used for angular velocity estimation and streak photometry.'
    )

    parser.add_argument(
        '--fov',
        type=float,
        default=None,
        help='Horizontal field of view in degrees (optional). Enables angular velocity in degrees/second.'
    )

    parser.add_argument(
        '--temporal-buffer',
        type=int,
        default=7,
        help='Size of the temporal rolling buffer for background subtraction (default: 7). '
             'The per-pixel temporal median removes stars, sky gradients, and vignetting. '
             'Set to 0 to disable temporal integration.'
    )

    parser.add_argument(
        '--algorithm', '-a',
        choices=['default', 'radon', 'nn'],
        default='default',
        help='Detection algorithm: "default" (Canny+Hough+MF), '
             '"radon" (LSD+Radon+PCF — advanced, catches dimmer trails), '
             'or "nn" (neural network model — requires --model). '
             'Default: default'
    )

    parser.add_argument(
        '--groundtruth',
        type=str,
        default=None,
        help='Path to directory containing ground truth trail patch images (PNG). '
             'Used by the "radon" algorithm to calibrate detection thresholds. '
             'If not specified with --algorithm radon, uses default thresholds.'
    )

    # --- Neural Network Algorithm ---
    parser.add_argument(
        '--model',
        type=str, default=None,
        help='Path to neural network model file (.pt, .onnx, .engine). '
             'Required when --algorithm nn is used. Supports YOLOv8/v11 '
             'models trained on satellite/airplane trail detection.'
    )

    parser.add_argument(
        '--nn-backend',
        choices=['ultralytics', 'cv2dnn', 'onnxruntime'],
        default=None,
        help='Neural network inference backend. Default: auto-detect from '
             'model file extension (.pt → ultralytics, .onnx → cv2dnn).'
    )

    parser.add_argument(
        '--confidence',
        type=float, default=None,
        help='Detection confidence threshold for NN algorithm (0-1). '
             'Default: 0.25 (or value from config file).'
    )

    parser.add_argument(
        '--nms-iou',
        type=float, default=None,
        help='NMS IoU threshold for NN algorithm (0-1). Default: 0.45.'
    )

    parser.add_argument(
        '--nn-input-size',
        type=int, default=None,
        help='Model input resolution for NN algorithm. Default: 640.'
    )

    parser.add_argument(
        '--nn-class-map',
        type=str, default=None,
        help='Class mapping as JSON string, e.g. \'{"satellite": [0], "airplane": [1]}\'. '
             'Maps Mnemosky trail types to model class IDs.'
    )

    parser.add_argument(
        '--nn-hybrid',
        action='store_true',
        help='Run NN detection alongside classical pipeline and merge results. '
             'Improves recall at the cost of speed.'
    )

    parser.add_argument(
        '--config',
        type=str, default=None,
        help='Path to config file (default: ~/.mnemosky/config.json). '
             'Stores model paths, backend choice, and all algorithm parameters.'
    )

    parser.add_argument(
        '--save-config',
        action='store_true',
        help='Save current CLI parameters to config file for future runs.'
    )

    _default_workers = min(max(1, (os.cpu_count() or 4) - 1), 8)
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=_default_workers,
        help=f'Number of parallel detection workers (default: {_default_workers}, '
             f'auto-detected from CPU count). Set to 0 for sequential processing.'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable CUDA GPU acceleration even if available'
    )

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

    # --- STS-Inspired Features ---
    parser.add_argument(
        '--ledger',
        action='store_true',
        help='Enable the Translation Ledger (Callon): track and report what '
             'the detector rejected and why. Prints a summary after processing.'
    )

    parser.add_argument(
        '--loss-profile',
        choices=['discovery', 'precision', 'balanced', 'catalog'],
        default='balanced',
        help='Named loss profile for HITL parameter learning (Winner). '
             '"discovery" maximizes recall, "precision" minimizes false alarms, '
             '"balanced" (default) slightly favors recall, '
             '"catalog" prioritizes correct classification.'
    )

    parser.add_argument(
        '--observer-lat',
        type=float, default=None,
        help='Observer latitude in decimal degrees (Haraway — situated knowledges)'
    )

    parser.add_argument(
        '--observer-lon',
        type=float, default=None,
        help='Observer longitude in decimal degrees'
    )

    parser.add_argument(
        '--observer-elevation',
        type=float, default=None,
        help='Observer elevation in metres above sea level'
    )

    parser.add_argument(
        '--observer-bortle',
        type=int, default=None, choices=range(1, 10),
        metavar='1-9',
        help='Bortle dark sky scale (1=pristine dark, 9=inner city). '
             'Adjusts contrast thresholds: darker skies allow dimmer detections.'
    )

    parser.add_argument(
        '--observer-fov',
        type=float, default=None,
        help='Observer field of view in degrees (overrides --fov for observer context)'
    )

    parser.add_argument(
        '--observer-notes',
        type=str, default=None,
        help='Free-text observation notes (e.g. "4-inch Newtonian, partly cloudy")'
    )

    args = parser.parse_args()

    # Validate parameter ranges
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if hasattr(args, 'auto_accept') and not (0.0 <= args.auto_accept <= 1.0):
        parser.error("--auto-accept must be in range [0.0, 1.0]")

    # ── Load application config ──────────────────────────────────────
    config = load_config(args.config)
    nn_config = config.get('algorithms', {}).get('nn', {})

    # ── Build nn_params if --algorithm nn ────────────────────────────
    nn_params = None
    if args.algorithm == 'nn':
        model_path = args.model or nn_config.get('model_path')
        if not model_path:
            parser.error(
                "--model is required when --algorithm nn is used "
                "(or set model_path in ~/.mnemosky/config.json)")

        # Auto-detect backend from file extension if not specified
        backend = args.nn_backend or nn_config.get('backend')
        if backend is None:
            ext = Path(model_path).suffix.lower()
            backend = {'.pt': 'ultralytics', '.onnx': 'cv2dnn',
                       '.engine': 'ultralytics'}.get(ext, 'ultralytics')

        # Parse class map from JSON string or config
        class_map = nn_config.get('class_map', {'satellite': [0], 'airplane': [1]})
        if args.nn_class_map:
            try:
                class_map = json.loads(args.nn_class_map)
            except json.JSONDecodeError as e:
                parser.error(f"--nn-class-map is not valid JSON: {e}")

        nn_params = {
            'model_path': str(Path(model_path).resolve()),
            'backend': backend,
            'confidence': (args.confidence if args.confidence is not None
                           else nn_config.get('confidence', 0.25)),
            'nms_iou': (args.nms_iou if args.nms_iou is not None
                        else nn_config.get('nms_iou', 0.45)),
            'input_size': (args.nn_input_size if args.nn_input_size is not None
                           else nn_config.get('input_size', 640)),
            'device': 'cpu' if args.no_gpu else nn_config.get('device', 'auto'),
            'half_precision': nn_config.get('half_precision', False),
            'class_map': class_map,
            'hybrid_mode': args.nn_hybrid,
        }

    # ── Save config if requested ─────────────────────────────────────
    if args.save_config:
        if nn_params:
            config['algorithms']['nn'] = {
                'model_path': nn_params['model_path'],
                'backend': nn_params['backend'],
                'confidence': nn_params['confidence'],
                'nms_iou': nn_params['nms_iou'],
                'input_size': nn_params['input_size'],
                'device': nn_params['device'],
                'half_precision': nn_params['half_precision'],
                'class_map': nn_params['class_map'],
            }
        config['algorithms']['default']['sensitivity'] = args.sensitivity
        config['algorithms']['radon']['sensitivity'] = args.sensitivity
        save_config(config, args.config)
        print(f"Config saved to {args.config or _CONFIG_PATH}")

    # Handle preprocessing preview if requested
    preprocessing_params = None
    signal_envelope = None
    radon_preview_params = None
    nn_preview_params = None
    if args.preview:
        if args.algorithm == 'radon':
            radon_preview_params = show_radon_preview(args.input)
            if radon_preview_params is None:
                print("Using default Radon pipeline parameters.")
        elif args.algorithm == 'nn':
            nn_preview_params = show_nn_preview(args.input, nn_params=nn_params)
            if nn_preview_params is not None:
                nn_params.update(nn_preview_params)
            else:
                print("Using default NN pipeline parameters.")
        else:
            preprocessing_params = show_preprocessing_preview(args.input)
            if preprocessing_params is None:
                print("Using default preprocessing parameters.")
            else:
                # Extract signal envelope (if user marked trail examples)
                signal_envelope = preprocessing_params.pop('signal_envelope', None)

    # Load learned parameters profile if available
    if args.hitl_profile and not args.review and not args.review_only:
        adapter = ParameterAdapter({})
        if adapter.load_profile(args.hitl_profile):
            learned = adapter.get_params()
            if preprocessing_params is None:
                preprocessing_params = {}
            for key in ['canny_low', 'canny_high']:
                if key in learned:
                    preprocessing_params[key] = learned[key]
            print(f"Loaded learned parameters from profile: {args.hitl_profile}")

    # ── Build observer context (Haraway — situated knowledges) ─────
    _observer_context = None
    if any([args.observer_lat, args.observer_lon, args.observer_bortle,
            args.observer_elevation, args.observer_notes, args.observer_fov]):
        _observer_context = {
            'lat': args.observer_lat,
            'lon': args.observer_lon,
            'elevation_m': args.observer_elevation,
            'bortle_class': args.observer_bortle,
            'fov_degrees': args.observer_fov or args.fov,
            'notes': args.observer_notes,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        }

    # ── Print loss profile (Winner — artifacts have politics) ────
    if args.loss_profile != 'balanced':
        profile = LOSS_PROFILES.get(args.loss_profile, LOSS_PROFILES['balanced'])
        print(f"Loss profile: {args.loss_profile} — {profile['description']}")

    # Handle --dataset-from-annotations: export from HITL-verified annotations
    if args.dataset_from_annotations:
        export_dataset_from_annotations(
            input_path=args.input,
            annotations_path=args.dataset_from_annotations,
            dataset_dir_override=args.dataset_dir,
            output_path=args.output,
            fmt=args.dataset_format,
            split_ratios=tuple(args.dataset_split),
            skip_frames=args.dataset_skip,
            dedup_threshold=args.dataset_dedup,
            negative_ratio=args.dataset_negatives,
            image_format=args.dataset_image_format,
            image_quality=args.dataset_image_quality,
            freeze_duration=args.freeze_duration,
        )
        return

    process_video(
        args.input,
        args.output,
        sensitivity=args.sensitivity,
        freeze_duration=args.freeze_duration,
        max_duration=args.max_duration,
        detect_type=args.detect_type,
        show_labels=not args.no_labels,
        debug_mode=args.debug,
        debug_only=args.debug_only,
        preprocessing_params=preprocessing_params,
        skip_aspect_ratio_check=args.no_aspect_ratio_check,
        signal_envelope=signal_envelope,
        save_dataset=args.dataset,
        exposure_time=args.exposure_time,
        fov_degrees=args.fov,
        temporal_buffer_size=args.temporal_buffer,
        algorithm=args.algorithm,
        groundtruth_dir=args.groundtruth,
        num_workers=args.workers,
        no_gpu=args.no_gpu,
        review_mode=args.review,
        review_only=args.review_only,
        annotations_path=args.annotations,
        hitl_profile=args.hitl_profile,
        auto_accept=args.auto_accept,
        no_learn=args.no_learn,
        dataset_format=args.dataset_format,
        dataset_split=tuple(args.dataset_split),
        dataset_skip=args.dataset_skip,
        dataset_dedup=args.dataset_dedup,
        dataset_negatives=args.dataset_negatives,
        dataset_image_format=args.dataset_image_format,
        dataset_image_quality=args.dataset_image_quality,
        dataset_dir_override=args.dataset_dir,
        radon_params=radon_preview_params,
        nn_params=nn_params,
        enable_ledger=args.ledger,
        loss_profile=args.loss_profile,
        observer_context=_observer_context,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
