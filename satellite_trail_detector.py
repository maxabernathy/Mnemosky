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
from pathlib import Path
from abc import ABC, abstractmethod


def show_preprocessing_preview(video_path, initial_params=None):
    """
    Show an interactive preview window for tuning preprocessing parameters.

    Uses a sleek dark-grey GUI with minimal fluorescent accent highlights.
    All controls — including custom-drawn sliders — live inside the single
    main window.  The layout is a 2x2 panel grid (Original, CLAHE,
    MF Response, Edges) with a parameter sidebar containing interactive
    sliders on the right, and a status bar at the bottom.

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

    # ── Layout constants (derived from screen size) ──────────────────
    target_win_w = int(_screen_w * 0.92)
    target_win_h = int(_screen_h * 0.88)
    sidebar_w = max(280, min(380, int(target_win_w * 0.18)))
    status_bar_h = 56               # Taller to hold the wide frame slider
    gap = 2
    content_w = target_win_w - sidebar_w
    content_h = target_win_h - status_bar_h
    # Original panel: large, full left column
    orig_w = int(content_w * 0.58)
    orig_h = content_h
    # Processing panels: stacked vertically in the right column
    small_w = content_w - orig_w - gap
    small_h = (content_h - 2 * gap) // 3
    canvas_w = orig_w + gap + small_w + sidebar_w
    canvas_h = content_h + status_bar_h
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

    # Temporal reference cache — built from surrounding frames when the user
    # navigates to a new frame.  Much slower to compute than the MF cache
    # (requires reading N frames from disk), so it's done lazily and only
    # when the frame changes.
    _TEMPORAL_N = 15  # Frames to each side for temporal median (total = 2N+1)
    temporal_ref_cache = {'frame_idx': -1, 'diff_image': None, 'noise_map': None}

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
        import math as _math

        h, w = gray_frm.shape

        # Downsample at 2/3 resolution (matches detector, good detail)
        scale = 2.0 / 3.0
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
            noise_map_small = None
            use_noise_map = False

        # Multi-scale directional filter bank
        # 72 angles (2.5° steps), 3 kernel lengths, Gaussian cross-section.
        # Scale-corrected SNR: noise_factor = sqrt(sum(kernel²)) accounts
        # for the per-kernel noise reduction so SNR values are comparable
        # across scales.
        num_angles = 72
        kernel_lengths = [15, 31, 51]
        best_snr = np.zeros_like(signal)

        for klen in kernel_lengths:
            for i in range(num_angles):
                angle_deg = i * 180.0 / num_angles

                # Gaussian-profile kernel (vectorised, no raster aliasing)
                ksize = klen if klen % 2 == 1 else klen + 1
                center = ksize // 2
                rad = _math.radians(angle_deg)
                cos_a, sin_a = _math.cos(rad), _math.sin(rad)

                yc, xc = np.mgrid[:ksize, :ksize]
                dx = (xc - center).astype(np.float32)
                dy = (yc - center).astype(np.float32)
                perp = np.abs(-sin_a * dx + cos_a * dy)
                along = np.abs(cos_a * dx + sin_a * dy)

                kern = np.exp(-0.5 * perp ** 2).astype(np.float32)  # sigma_perp=1.0
                kern[along > center] = 0
                ksum = np.sum(kern)
                if ksum > 0:
                    kern /= ksum

                noise_factor = np.sqrt(np.sum(kern ** 2))
                response = cv2.filter2D(signal, cv2.CV_32F, kern)

                if use_noise_map:
                    snr = response / (noise_map_small * noise_factor + 1e-10)
                else:
                    snr = response / (noise_std * noise_factor + 1e-10)

                better = snr > best_snr
                best_snr[better] = snr[better]

        # Scale back to full resolution
        return cv2.resize(best_snr, (w, h), interpolation=cv2.INTER_LINEAR)

    def get_mf_snr_map(gray_frm, frame_idx):
        """Return the cached MF SNR map, recomputing only when frame changes."""
        if mf_cache['frame_idx'] != frame_idx:
            mf_cache['snr_map'] = compute_mf_response(gray_frm)
            mf_cache['frame_idx'] = frame_idx
        return mf_cache['snr_map']

    def extract_mf_lines(snr_map, snr_thresh):
        """Threshold the SNR map and extract candidate line segments."""
        h, w = snr_map.shape
        # Work at half resolution for Hough (matches detector behaviour)
        scale = 0.5
        small_snr = cv2.resize(snr_map, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        significant = (small_snr > snr_thresh).astype(np.uint8) * 255
        cleanup = np.ones((3, 3), np.uint8)
        significant = cv2.dilate(significant, cleanup, iterations=1)
        significant = cv2.erode(significant, cleanup, iterations=1)

        lines = cv2.HoughLinesP(significant, 1, np.pi / 180, 10,
                                minLineLength=15, maxLineGap=30)
        inv = 1.0 / scale
        scaled = []
        if lines is not None:
            for ln in lines:
                x1, y1, x2, y2 = ln[0]
                scaled.append((int(x1 * inv), int(y1 * inv),
                               int(x2 * inv), int(y2 * inv)))
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
        # Original: large left column.  Processing panels: stacked right.
        orig_small = cv2.resize(frm, (orig_w, orig_h))

        enh_resized = cv2.resize(enhanced, (small_w, small_h))
        enh_bgr = cv2.cvtColor(enh_resized, cv2.COLOR_GRAY2BGR)

        # ── Matched-filter response heatmap (replaces old Blur panel) ──
        snr_thresh = p['mf_snr_threshold'] / 10.0
        snr_map = get_mf_snr_map(gray, current_frame_idx)
        snr_resized = cv2.resize(snr_map, (small_w, small_h),
                                 interpolation=cv2.INTER_LINEAR)

        # Two-tone heatmap: dim below threshold, bright above
        mf_bgr = np.zeros((small_h, small_w, 3), dtype=np.uint8)
        mf_bgr[:] = BG_PANEL

        # Dim sub-threshold signal (subtle visibility so user sees the landscape)
        has_signal = snr_resized > 0.5
        below_thresh = has_signal & (snr_resized < snr_thresh)
        above_thresh = snr_resized >= snr_thresh

        # Intensity ramp for sub-threshold pixels
        if np.any(below_thresh):
            intensity_below = np.clip(snr_resized / snr_thresh, 0, 1)
            for c in range(3):
                channel = mf_bgr[:, :, c].astype(np.float32)
                channel[below_thresh] = (
                    BG_PANEL[c] + (ACCENT_MF_DIM[c] - BG_PANEL[c]) * intensity_below[below_thresh]
                )
                mf_bgr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

        # Bright above-threshold pixels
        if np.any(above_thresh):
            intensity_above = np.clip(snr_resized / (snr_thresh * 3), 0.5, 1)
            for c in range(3):
                channel = mf_bgr[:, :, c].astype(np.float32)
                channel[above_thresh] = (
                    ACCENT_MF_DIM[c] + (ACCENT_MF[c] - ACCENT_MF_DIM[c]) * intensity_above[above_thresh]
                )
                mf_bgr[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

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

        sx = orig_w + gap  # x-offset for right column
        for px, py, pw, ph, panel in [
            (0,  0,                          orig_w,  orig_h,  orig_small),
            (sx, 0,                          small_w, small_h, enh_bgr),
            (sx, small_h + gap,              small_w, small_h, mf_bgr),
            (sx, 2 * (small_h + gap),        small_w, small_h, edge_bgr),
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

        # Panel tags
        tag_y, tag_x = 18, 8
        trail_count_str = f"  [{len(marked_trails)}/{MAX_TRAILS}]" if marked_trails or pending_click[0] else ""
        _draw_tag(canvas, "ORIGINAL" + trail_count_str, tag_x, tag_y, BG_DARK, TEXT_DIM if not trail_count_str else TRAIL_MARK)
        clip_val = p['clahe_clip_limit'] / 10.0
        _draw_tag(canvas, f"CLAHE  clip {clip_val:.1f}  tile {p['clahe_tile_size']}",
                  sx + tag_x, tag_y, BG_DARK, ACCENT)
        mf_snr_val = p['mf_snr_threshold'] / 10.0
        mf_line_count = len(mf_lines)
        is_temporal = temporal_ref_cache['diff_image'] is not None
        mf_mode = "TEMPORAL" if is_temporal else "SPATIAL"
        mf_tag = f"MF {mf_mode}  SNR>={mf_snr_val:.1f}"
        if mf_line_count > 0:
            mf_tag += f"  [{mf_line_count}]"
        _draw_tag(canvas, mf_tag,
                  sx + tag_x, small_h + gap + tag_y, BG_DARK, ACCENT_MF)
        _draw_tag(canvas, f"EDGES  {p['canny_low']}-{p['canny_high']}",
                  sx + tag_x, 2 * (small_h + gap) + tag_y, BG_DARK, ACCENT_CYAN)

        # ── Sidebar ──────────────────────────────────────────────────
        sb_x = orig_w + gap + small_w
        _fill_rect(canvas, sb_x, 0, sidebar_w, canvas_h - status_bar_h, BG_SIDEBAR)
        _draw_border(canvas, sb_x, 0, sidebar_w, canvas_h - status_bar_h, BORDER)

        # Title
        _put_text(canvas, "MNEMOSKY", sb_x + 14, 24, ACCENT, 0.52, 1)
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

        if event == cv2.EVENT_LBUTTONDOWN:
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
    print("The MF RESPONSE panel (bottom-left) shows the directional matched")
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

    # ── Main loop ────────────────────────────────────────────────────
    while True:
        # Check if Frame slider was dragged to a new position
        if params['frame_idx'] != current_frame_idx:
            current_frame_idx = params['frame_idx']
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame

        gray, enhanced, blurred, edges = apply_preprocessing(frame, params)
        display = create_display(frame, gray, enhanced, blurred, edges, params)

        cv2.imshow(window_name, display)

        if first_render:
            cv2.resizeWindow(window_name, canvas_w, canvas_h)
            first_render = False

        key = cv2.waitKey(30) & 0xFF

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
            temporal_ref_cache['frame_idx'] = -1  # Invalidate temporal cache
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

        buf = TemporalFrameBuffer(capacity=21)
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            buf.add(gray)
            if buf.is_ready():
                ctx = buf.get_temporal_context(gray)
                # ctx['diff_image'], ctx['noise_map'], ctx['reference']
    """

    def __init__(self, capacity=21):
        """
        Args:
            capacity: Number of frames to keep in the buffer.  Should be odd
                so the current frame sits at the centre.  Larger values give
                cleaner backgrounds but use more RAM (~4 MB per 1080p frame).
                21 frames ≈ 84 MB for 1080p — a reasonable default.
        """
        self.capacity = capacity
        self._frames = []           # List of uint8 grayscale arrays
        self._reference = None      # Cached temporal median (uint8)
        self._noise_map = None      # Cached per-pixel MAD noise (float32)
        self._dirty = True          # True if buffer changed since last compute

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
        """Compute temporal median and noise map from the current buffer."""
        if not self._dirty:
            return

        # Stack frames into a 3D array: (N, H, W)
        stack = np.stack(self._frames, axis=0).astype(np.float32)

        # Per-pixel temporal median — stars and fixed pattern noise vanish
        self._reference = np.median(stack, axis=0)

        # Per-pixel MAD (Median Absolute Deviation) → robust noise estimate
        # MAD is unaffected by the transient trail pixels (outliers)
        abs_dev = np.abs(stack - self._reference[np.newaxis, :, :])
        mad = np.median(abs_dev, axis=0)
        self._noise_map = (mad * 1.4826).astype(np.float32)  # Gaussian σ equiv
        # Floor at 0.5 to avoid division by zero in SNR calculations
        self._noise_map = np.maximum(self._noise_map, 0.5)

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


class BaseDetectionAlgorithm(ABC):
    """
    Abstract base class for trail detection algorithms.

    Subclass this to implement custom detection algorithms while maintaining
    compatibility with the processing pipeline.
    """

    def __init__(self, params):
        """
        Initialize the detection algorithm with parameters.

        Args:
            params: Dictionary of detection parameters
        """
        self.params = params

    @abstractmethod
    def preprocess_frame(self, frame):
        """
        Preprocess a frame for detection.

        Args:
            frame: Input BGR frame

        Returns:
            Tuple of (original_gray, preprocessed) frames
        """
        pass

    @abstractmethod
    def detect_lines(self, preprocessed):
        """
        Detect lines in preprocessed frame.

        Args:
            preprocessed: Preprocessed frame

        Returns:
            Tuple of (lines, edges) where lines is array of detected lines
        """
        pass

    @abstractmethod
    def classify_trail(self, line, gray_frame, color_frame):
        """
        Classify a detected line as satellite, airplane, or neither.

        Args:
            line: Detected line coordinates
            gray_frame: Grayscale frame
            color_frame: Color frame

        Returns:
            Tuple of (trail_type, detection_info) where trail_type is
            'satellite', 'airplane', or None, and detection_info is a dict
            with at least a 'bbox' key (x_min, y_min, x_max, y_max) plus
            optional metadata like 'angle', 'center', 'length', etc.
        """
        pass

    def detect_trails(self, frame, debug_info=None):
        """
        Main detection pipeline. Can be overridden for completely custom logic.

        Args:
            frame: Input frame
            debug_info: Optional dict to collect debug information

        Returns:
            List of tuples: [('satellite', detection_info), ('airplane', detection_info), ...]
            where detection_info is a dict with at least 'bbox' key.
        """
        gray, preprocessed = self.preprocess_frame(frame)
        lines, edges = self.detect_lines(preprocessed)

        if lines is None:
            if debug_info is not None:
                debug_info['all_lines'] = []
                debug_info['all_classifications'] = []
                debug_info['edges'] = edges
                debug_info['gray_frame'] = gray
            return []

        classified_trails = []
        all_classifications = []

        for line in lines:
            trail_type, detection_info = self.classify_trail(line, gray, frame)

            if debug_info is not None:
                all_classifications.append({
                    'line': line,
                    'type': trail_type,
                    'detection_info': detection_info,
                    'bbox': detection_info['bbox'] if detection_info else None,
                })

            if trail_type and detection_info:
                classified_trails.append((trail_type, detection_info))

        if debug_info is not None:
            debug_info['all_lines'] = lines
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray

        # Separate by type for merging
        satellite_boxes = [info['bbox'] for t, info in classified_trails if t == 'satellite']
        airplane_boxes = [info['bbox'] for t, info in classified_trails if t == 'airplane']

        # Merge overlapping detections within each type
        merged_satellites = self.merge_overlapping_boxes(satellite_boxes)
        merged_airplanes = self.merge_overlapping_boxes(airplane_boxes)

        # Combine results with type labels (wrap in detection_info dicts)
        results = [('satellite', {'bbox': bbox}) for bbox in merged_satellites]
        results.extend([('airplane', {'bbox': bbox}) for bbox in merged_airplanes])

        return results

    def merge_overlapping_boxes(self, boxes, overlap_threshold=0.3):
        """Merge overlapping bounding boxes."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = []

        for box in boxes:
            if not merged:
                merged.append(list(box))
                continue

            found_overlap = False
            for i, mbox in enumerate(merged):
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


class DefaultDetectionAlgorithm(BaseDetectionAlgorithm):
    """
    Default detection algorithm using Hough line detection and brightness analysis.

    This is the original detection implementation that identifies trails based on:
    - Edge detection (Canny)
    - Line detection (Hough transform)
    - Brightness analysis (dotted vs smooth patterns)
    - Color saturation (for airplane navigation lights)
    """

    def __init__(self, params):
        super().__init__(params)

    def preprocess_frame(self, frame):
        """Convert frame to grayscale and enhance for trail detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement (helps with dim trails)
        # Increased clipLimit and smaller tiles for better detection of very dim satellites
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(6, 6))
        enhanced = clahe.apply(gray)

        # Very minimal Gaussian blur - reduced to preserve more signal and detail
        # Using sigma=0.3 for extremely light smoothing to reduce sensor noise only
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.3)

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
        # Enhanced to better connect dim, fragmented, and less steady satellite trails
        kernel = np.ones((3, 3), np.uint8)

        # Dilate more aggressively to connect gaps in dim trails
        edges = cv2.dilate(edges, kernel, iterations=3)
        # Erode less to preserve dim features
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

        return lines, edges

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
                'satellite_max_length': 800,
                'satellite_contrast_min': 1.10,  # Minimum trail-to-background contrast
            },
            'medium': {
                'min_line_length': 50,  # Lower to catch dim trail fragments
                'max_line_gap': 50,  # Wider gap tolerance for dim fragmented trails
                'canny_low': 4,  # Slightly more sensitive for dim trails
                'canny_high': 45,
                'hough_threshold': 30,  # Lower threshold to catch dim trails
                'min_aspect_ratio': 4,  # Require trails to be relatively long and thin
                'brightness_threshold': 18,
                'airplane_brightness_min': 75,
                'airplane_saturation_min': 8,
                'satellite_min_length': 100,  # Satellites can be shorter segments
                'satellite_max_length': 1200,  # Very long trails for full-frame crossings
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
                'satellite_max_length': 2000,  # No practical upper limit
                'satellite_contrast_min': 1.05,  # Very dim trails allowed
            }
        }

        self.params = presets.get(sensitivity, presets['medium'])
        # Yellowish complementary color palette (BGR format)
        self.satellite_color = (0, 185, 255)  # Gold/yellow for satellites
        self.airplane_color = (0, 140, 255)   # Orange/amber for airplanes
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
        # Higher CLAHE clip limit (6.0) enhances dim satellite trails more aggressively
        if self.preprocessing_params:
            clip_limit = self.preprocessing_params.get('clahe_clip_limit', 6.0)
            tile_size = self.preprocessing_params.get('clahe_tile_size', 6)
            blur_kernel = self.preprocessing_params.get('blur_kernel_size', 3)
            blur_sigma = self.preprocessing_params.get('blur_sigma', 0.3)
        else:
            clip_limit = 6.0
            tile_size = 6
            blur_kernel = 3
            blur_sigma = 0.3

        # Ensure blur kernel is odd
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # Apply CLAHE for contrast enhancement (helps with dim trails)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)

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
        # Enhanced to better connect dim, fragmented satellite trails
        kernel = np.ones((3, 3), np.uint8)

        # Dilate to connect gaps in dim trails
        edges = cv2.dilate(edges, kernel, iterations=3)
        # Light erosion to preserve dim features
        edges = cv2.erode(edges, kernel, iterations=1)

        # Additional directional dilation to bridge gaps in linear features.
        # Dim satellite trails fragment into short segments with small gaps;
        # elongated kernels reconnect them without bloating non-linear noise.
        for angle in [0, 45, 90, 135]:
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

        # --- Downsample for performance (2/3 resolution) ---
        scale = 2.0 / 3.0
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
            bg_kernel = 31  # Must be odd
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
        # Multiple kernel lengths catch both short bright fragments and
        # long dim trails.  72 angles (2.5° steps) for fine angular
        # resolution.  Gaussian-profile kernels avoid rasterisation
        # aliasing.  Scale-corrected SNR ensures consistent thresholding
        # across kernel sizes.
        num_angles = 72   # 2.5-degree angular resolution
        kernel_lengths = [21, 41, 61]  # Multi-scale: ~4.6×, 6.4×, 7.8× SNR

        best_snr = np.zeros_like(signal)
        best_angle = np.zeros_like(signal)

        for klen in kernel_lengths:
            for i in range(num_angles):
                angle_deg = i * 180.0 / num_angles
                kernel = self._create_matched_filter_kernel(klen, angle_deg)

                # Noise factor for this kernel: filtered noise std =
                # noise_std * sqrt(sum(kernel²)).  Pre-compute once.
                noise_factor = np.sqrt(np.sum(kernel ** 2))

                response = cv2.filter2D(signal, cv2.CV_32F, kernel)

                if use_noise_map:
                    # Per-pixel SNR: filter the noise map through the same
                    # kernel to get the local filtered noise std at each pixel.
                    # For a mean-normalised kernel, σ_out² = σ_in² · Σ(k²).
                    snr = response / (noise_map_small * noise_factor + 1e-10)
                else:
                    # Scalar noise: same SNR formula as before
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
        max_gap = int(self.params['max_line_gap'] * scale * 1.5)
        hough_thresh = max(10, int(self.params['hough_threshold'] * 0.4))

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

        return np.array(scaled_lines) if scaled_lines else None

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

        n_samples = max(20, int(length / 3))
        n_samples = min(n_samples, 200)

        h, w = gray_frame.shape
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        perp_dx = -dy   # Perpendicular direction
        perp_dy = dx

        trail_values = []
        flank_values = []
        flank_dist = 8  # Pixels from trail center to flank sample

        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0
            cx = x1 + t * (x2 - x1)
            cy = y1 + t * (y2 - y1)

            # Trail pixel
            tx, ty = int(round(cx)), int(round(cy))
            if 0 <= tx < w and 0 <= ty < h:
                trail_values.append(float(gray_frame[ty, tx]))

            # Flank pixels (both sides of the trail)
            for sign in (-1, 1):
                fx = int(round(cx + sign * flank_dist * perp_dx))
                fy = int(round(cy + sign * flank_dist * perp_dy))
                if 0 <= fx < w and 0 <= fy < h:
                    flank_values.append(float(gray_frame[fy, fx]))

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

    def classify_trail(self, line, gray_frame, color_frame, hsv_frame=None, reusable_mask=None, supplementary=False):
        """
        Classify a detected line as either a satellite or airplane trail.

        Key distinction:
        - Airplanes: DOTTED features - bright point-like lights along the trail (navigation lights)
                    Sometimes colorful dots (red, green, white). Can be any length including 180-300px.
        - Satellites: SMOOTH, consistent brightness along trail. No bright point features.
                     Dim, monochromatic, uniform appearance. Typically 180-300 pixels for 1920x1080.

        Args:
            line: Detected line from HoughLinesP
            gray_frame: Grayscale frame
            color_frame: BGR color frame
            hsv_frame: Pre-computed HSV frame (optional, for performance)
            reusable_mask: Pre-allocated mask array (optional, for performance)
            supplementary: If True, this candidate came from the matched-filter
                stage and has already passed an SNR gate.  Contrast thresholds
                are relaxed and an additional SNR-based detection path is enabled.

        Returns:
            trail_type: 'satellite', 'airplane', or None
            detection_info: dict with 'bbox' and metadata if trail detected, None otherwise.
                Keys: 'bbox' (x_min, y_min, x_max, y_max), 'angle' (degrees 0-180),
                'center' (x, y), 'length' (pixels), 'avg_brightness' (float),
                'max_brightness' (int), 'line' (original line endpoints)
        """
        x1, y1, x2, y2 = line[0]

        # Calculate line properties
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Angle in degrees (0-180 range, normalized so direction doesn't matter)
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        if length < self.params['min_line_length']:
            return None, None

        # Use reusable mask if provided, otherwise allocate new one
        if reusable_mask is not None:
            mask = reusable_mask
            mask.fill(0)  # Clear the mask
        else:
            mask = np.zeros(gray_frame.shape, dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=5)

        # Check brightness along the trail
        trail_pixels_gray = gray_frame[mask > 0]
        if len(trail_pixels_gray) == 0:
            return None, None

        # Require minimum number of pixels - ensures we're detecting actual trails
        if len(trail_pixels_gray) < 15:
            return None, None

        avg_brightness = np.mean(trail_pixels_gray)
        max_brightness = np.max(trail_pixels_gray)
        brightness_std = np.std(trail_pixels_gray)

        # Too dark (likely noise) - trails should have some minimum brightness
        if avg_brightness < 5:
            return None, None

        # Check minimum contrast - trail should stand out from background
        # Use per-sensitivity threshold so dim satellite trails aren't rejected
        surround_sample_size = 30
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Get background brightness from area around the trail
        bg_x_min = max(0, x_center - surround_sample_size)
        bg_y_min = max(0, y_center - surround_sample_size)
        bg_x_max = min(gray_frame.shape[1], x_center + surround_sample_size)
        bg_y_max = min(gray_frame.shape[0], y_center + surround_sample_size)

        background_region = gray_frame[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
        contrast_ratio = None
        if background_region.size > 0:
            background_brightness = np.median(background_region)
            contrast_ratio = avg_brightness / (background_brightness + 1e-5)
            min_contrast = self.params.get('satellite_contrast_min', 1.08)
            # Supplementary candidates already passed the matched-filter SNR
            # gate, so the contrast criterion can be relaxed to avoid rejecting
            # dim trails that the primary pipeline would never have seen.
            if supplementary:
                min_contrast = max(1.02, min_contrast * 0.7)
            if contrast_ratio < min_contrast:
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
                return None, None

            # Reject very bright uniform areas (likely daytime sky or lit structures)
            if surrounding_mean > 80:
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
                    return None, None

        # 3. Check maximum brightness - extremely bright lines are likely not trails
        # Real airplane/satellite trails in night sky shouldn't be extremely bright
        if max_brightness > 240:
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
            n_spread = max(20, int(length / 5))
            n_spread = min(n_spread, 100)
            spread_samples = []
            for si in range(n_spread):
                st = si / (n_spread - 1) if n_spread > 1 else 0
                spx = int(x1 + st * (x2 - x1))
                spy = int(y1 + st * (y2 - y1))
                if 0 <= spy < gray_frame.shape[0] and 0 <= spx < gray_frame.shape[1]:
                    spread_samples.append(gray_frame[spy, spx])
            if len(spread_samples) >= 10:
                spread_arr = np.array(spread_samples)
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

        # Build detection metadata (shared by both airplane and satellite results)
        def _make_detection_info():
            return {
                'bbox': bbox,
                'angle': angle,
                'center': center,
                'length': length,
                'avg_brightness': float(avg_brightness),
                'max_brightness': int(max_brightness),
                'line': (x1, y1, x2, y2),
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
            return 'airplane', _make_detection_info()

        # If multiple distinct point features detected (navigation lights pattern)
        # Require higher brightness to avoid false positives
        if has_multiple_points and max_brightness > 120 and is_bright:
            return 'airplane', _make_detection_info()

        # Calculate airplane score - require more evidence
        airplane_score = sum([is_bright, is_colorful, has_color_variation, has_dotted_pattern])

        # Require dotted pattern AND at least 2 other characteristics
        if has_dotted_pattern and airplane_score >= 3:
            return 'airplane', _make_detection_info()

        # Very strong dotted pattern with high brightness
        if has_dotted_pattern and brightness_peak_ratio > 2.0 and max_brightness > 120:
            return 'airplane', _make_detection_info()

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
        has_contrast = contrast_ratio is not None and contrast_ratio >= self.params.get('satellite_contrast_min', 1.08)

        satellite_score = sum([is_dim, is_monochrome, is_smooth, is_satellite_length])

        # --- Primary paths (strongest confidence) ---

        # All 4 characteristics met
        if satellite_score >= 4 and not has_dotted_pattern:
            return 'satellite', _make_detection_info()

        # 3 characteristics including both smoothness and length
        if satellite_score >= 3 and is_smooth and is_satellite_length and not has_dotted_pattern:
            return 'satellite', _make_detection_info()

        # Very dim, smooth trails in correct length range
        if is_smooth and avg_brightness <= self.params['brightness_threshold'] * 1.5 and is_satellite_length and not has_dotted_pattern:
            return 'satellite', _make_detection_info()

        # --- Extended paths for dim/long trails that miss primary criteria ---

        # Long smooth dim trail outside the "typical" length range but clearly
        # not an airplane: no dotted pattern, dim, monochrome, smooth
        if is_smooth and is_dim and is_monochrome and not has_dotted_pattern and length >= self.params['satellite_min_length']:
            return 'satellite', _make_detection_info()

        # Dim smooth trail with confirmed background contrast — even if
        # length or monochrome criteria aren't perfectly met
        if is_smooth and is_dim and has_contrast and not has_dotted_pattern and length >= self.params['satellite_min_length']:
            return 'satellite', _make_detection_info()

        # Very dim trail (below brightness_threshold) that is smooth and long
        # enough — relaxed monochrome requirement since very dim trails have
        # negligible color information anyway
        if is_smooth and avg_brightness <= self.params['brightness_threshold'] and not has_dotted_pattern and length >= self.params['satellite_min_length']:
            return 'satellite', _make_detection_info()

        # --- SNR-based path for matched-filter candidates ---
        # Trails found by the supplementary matched filter have already
        # passed a global SNR gate, but may fail the absolute-brightness or
        # contrast thresholds used above.  Compute a per-trail SNR using
        # perpendicular flank sampling — a statistically rigorous measure
        # that is independent of absolute brightness.  This rescues very dim
        # trails that are clearly above the local noise floor.
        if supplementary and not has_dotted_pattern and length >= self.params['satellite_min_length']:
            trail_snr = self._compute_trail_snr(gray_frame, line)
            if trail_snr >= 2.5 and is_smooth:
                return 'satellite', _make_detection_info()

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

        # --- Stage 1: Primary detection (Canny + Hough) ---
        if lines is not None:
            for line in lines:
                trail_type, detection_info = self.classify_trail(
                    line, gray, frame, hsv_frame, reusable_mask)

                # Store for debug (even if filtered out)
                if debug_info is not None:
                    all_classifications.append({
                        'line': line,
                        'type': trail_type,
                        'detection_info': detection_info,
                        # Keep 'bbox' for backward compat with debug panel lookup
                        'bbox': detection_info['bbox'] if detection_info else None,
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
            for line in supplementary_lines:
                trail_type, detection_info = self.classify_trail(
                    line, gray, frame, hsv_frame, reusable_mask,
                    supplementary=True)

                if debug_info is not None:
                    all_classifications.append({
                        'line': line,
                        'type': trail_type,
                        'detection_info': detection_info,
                        'bbox': detection_info['bbox'] if detection_info else None,
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
        satellite_infos = [info for t, info in classified_trails if t == 'satellite']
        airplane_infos = [info for t, info in classified_trails if t == 'airplane']

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

        # --- Enrich detections with photometry, curvature, velocity ---
        frame_width = frame.shape[1]
        diff_img = temporal_context['diff_image'] if temporal_context else None

        all_merged = (
            [('satellite', info) for info in merged_satellite_infos] +
            [('airplane', info) for info in merged_airplane_infos]
        )
        for trail_type, info in all_merged:
            line_arr = np.array([[info['line'][0], info['line'][1],
                                  info['line'][2], info['line'][3]]])

            # Streak photometry
            info['photometry'] = self._analyze_streak_photometry(gray, line_arr)

            # Trail curvature
            info['curvature'] = self._fit_trail_curvature(
                gray, line_arr, diff_image=diff_img)

            # Angular velocity
            info['velocity'] = self._estimate_angular_velocity(
                info['length'], frame_width,
                exposure_time=exposure_time,
                fov_degrees=fov_degrees)

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

            # Create semi-transparent background for label
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (label_x - 2, label_y - label_size[1] - 6),
                (label_x + label_size[0] + 4, label_y + 4),
                color,
                -1
            )
            # Blend the overlay with the original frame for transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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
        cv2.rectangle(overlay, (5, 5), (180, 95), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, debug_frame, 0.4, 0, debug_frame)

        # Legend text
        cv2.putText(debug_frame, "DEBUG VIEW", (legend_x, legend_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        legend_y += 20
        cv2.putText(debug_frame, "Green (A): Airplane", (legend_x, legend_y), font, font_scale - 0.1, (0, 255, 0), font_thickness, cv2.LINE_AA)
        legend_y += 15
        cv2.putText(debug_frame, "Cyan (S): Satellite", (legend_x, legend_y), font, font_scale - 0.1, (255, 255, 0), font_thickness, cv2.LINE_AA)
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


def process_video(input_path, output_path, sensitivity='medium', freeze_duration=1.0, max_duration=None, detect_type='both', show_labels=True, debug_mode=False, debug_only=False, preprocessing_params=None, skip_aspect_ratio_check=False, signal_envelope=None, save_dataset=False, exposure_time=13.0, fov_degrees=None, temporal_buffer_size=21):
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
            (default: 21).  The temporal median of this many surrounding frames
            is used as a reference background — stars, vignetting, and sky
            gradients are removed perfectly, leaving only transient trails.
            Set to 0 to disable temporal integration.
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
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    
    # Initialize detector
    detector = SatelliteTrailDetector(sensitivity, preprocessing_params=preprocessing_params, skip_aspect_ratio_check=skip_aspect_ratio_check, signal_envelope=signal_envelope)

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

    frame_count = 0
    satellites_detected = 0
    airplanes_detected = 0

    # ── YOLO dataset setup ────────────────────────────────────────
    dataset_dir = None
    dataset_images = 0
    dataset_annotations = {'satellite': 0, 'airplane': 0}
    if save_dataset:
        out_p = Path(output_path)
        dataset_dir = out_p.parent / (out_p.stem + '_dataset')
        (dataset_dir / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / 'labels').mkdir(parents=True, exist_ok=True)
        video_stem = Path(input_path).stem
        print(f"Dataset export enabled → {dataset_dir}")

    # Track frozen regions: list of (frozen_region, bbox, trail_type, frames_remaining)
    frozen_regions = []

    # Track debug panels for detections (only in debug mode)
    # Each entry: {'panel': image, 'frames_remaining': int}
    debug_panels = []
    debug_panel_duration = int(fps * 2)  # 2 seconds

    print("\nProcessing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Check if we've reached the maximum duration
        if frame_count > max_frames:
            print(f"\rReached maximum duration limit at frame {frame_count - 1}")
            break

        # Progress indicator
        if frame_count % 30 == 0:
            progress = (frame_count / max_frames) * 100
            print(f"\rProgress: {progress:.1f}% ({frame_count}/{max_frames})", end="", flush=True)

        # Start with current frame
        output_frame = frame.copy()

        # ── Feed temporal buffer and build context ────────────────
        temporal_context = None
        if temporal_buffer is not None:
            gray_for_buffer = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            temporal_buffer.add(gray_for_buffer)
            if temporal_buffer.is_ready():
                temporal_context = temporal_buffer.get_temporal_context(
                    gray_for_buffer)

        # Detect and classify trails (with debug info if needed)
        debug_info = {} if debug_mode else None
        detected_trails = detector.detect_trails(
            frame, debug_info=debug_info,
            temporal_context=temporal_context,
            exposure_time=exposure_time,
            fov_degrees=fov_degrees)

        # Filter trails based on detect_type parameter
        if detect_type == 'satellites':
            detected_trails = [(t, b) for t, b in detected_trails if t == 'satellite']
        elif detect_type == 'airplanes':
            detected_trails = [(t, b) for t, b in detected_trails if t == 'airplane']
        # If detect_type == 'both', no filtering needed

        if detected_trails:
            # Count detections by type and add new frozen regions
            for trail_type, detection_info in detected_trails:
                bbox = detection_info['bbox']

                if trail_type == 'satellite':
                    satellites_detected += 1
                elif trail_type == 'airplane':
                    airplanes_detected += 1

                # Log enrichment metadata for this detection
                _meta_parts = [f"f{frame_count}",
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
                if debug_mode and 'edges' in debug_info and 'gray_frame' in debug_info and debug_info['all_lines'] is not None:
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

            # ── YOLO dataset export (one image + label per frame with detections)
            if dataset_dir is not None:
                img_name = f"{video_stem}_f{frame_count:06d}.jpg"
                cv2.imwrite(str(dataset_dir / 'images' / img_name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                label_lines = []
                frame_h, frame_w = frame.shape[:2]
                for trail_type_d, det_info_d in detected_trails:
                    cls_id = 0 if trail_type_d == 'satellite' else 1
                    bx0, by0, bx1, by1 = det_info_d['bbox']
                    xc = ((bx0 + bx1) / 2) / frame_w
                    yc = ((by0 + by1) / 2) / frame_h
                    bw = (bx1 - bx0) / frame_w
                    bh = (by1 - by0) / frame_h
                    label_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    dataset_annotations[trail_type_d] += 1
                label_path = dataset_dir / 'labels' / img_name.replace('.jpg', '.txt')
                label_path.write_text('\n'.join(label_lines) + '\n')
                dataset_images += 1

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
                actual_y_max = min(y_max, frame_h)
                actual_x_max = min(x_max, frame_w)
                actual_region_h = actual_y_max - y_min
                actual_region_w = actual_x_max - x_min

                if actual_region_h > 0 and actual_region_w > 0:
                    output_frame[y_min:actual_y_max, x_min:actual_x_max] = region[:actual_region_h, :actual_region_w]

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

    if detect_type in ['both', 'satellites']:
        print(f"Satellites detected: {satellites_detected}")
    if detect_type in ['both', 'airplanes']:
        print(f"Airplanes detected: {airplanes_detected}")

    print(f"Total trails detected: {satellites_detected + airplanes_detected}")
    print(f"Output saved to: {output_path}")

    # ── YOLO dataset finalisation ─────────────────────────────────
    if dataset_dir is not None and dataset_images > 0:
        # Write data.yaml for Ultralytics / YOLO training
        yaml_path = dataset_dir / 'data.yaml'
        yaml_path.write_text(
            "path: .\n"
            "train: images\n"
            "val: images\n"
            "\n"
            "names:\n"
            "  0: satellite\n"
            "  1: airplane\n"
        )
        total_ann = dataset_annotations['satellite'] + dataset_annotations['airplane']
        sat_ann = dataset_annotations['satellite']
        air_ann = dataset_annotations['airplane']
        print(f"\n{'=' * 42}")
        print(f"  ML Dataset Summary")
        print(f"{'─' * 42}")
        print(f"  Format:       YOLOv8")
        print(f"  Location:     {dataset_dir}")
        print(f"  Images:       {dataset_images}")
        print(f"  Annotations:  {total_ann} ({sat_ann} satellite, {air_ann} airplane)")
        print(f"  Class map:    0=satellite, 1=airplane")
        print(f"{'=' * 42}")
    elif dataset_dir is not None:
        print("\nNo detections found — dataset directory is empty.")


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
        choices=['both', 'satellites', 'airplanes'],
        default='both',
        help='What to detect: "both" (default), "satellites" only, or "airplanes" only'
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
        default=21,
        help='Size of the temporal rolling buffer for background subtraction (default: 21). '
             'The per-pixel temporal median removes stars, sky gradients, and vignetting. '
             'Set to 0 to disable temporal integration.'
    )

    args = parser.parse_args()

    # Handle preprocessing preview if requested
    preprocessing_params = None
    signal_envelope = None
    if args.preview:
        preprocessing_params = show_preprocessing_preview(args.input)
        if preprocessing_params is None:
            print("Using default preprocessing parameters.")
        else:
            # Extract signal envelope (if user marked trail examples)
            signal_envelope = preprocessing_params.pop('signal_envelope', None)

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
    )


if __name__ == '__main__':
    main()
