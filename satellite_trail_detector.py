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
    main window.  The layout is a 2x2 panel grid (Original, CLAHE, Blur,
    Edges) with a parameter sidebar containing interactive sliders on the
    right, and a status bar at the bottom.

    Controls:
    - Use Frame slider to select exact frame containing trail signal
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
        Dict with selected preprocessing parameters, or None if cancelled
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
    SLIDER_TRACK = (50, 50, 50)      # Slider track background
    SLIDER_FILL = (200, 255, 80)     # Slider filled portion (accent)
    SLIDER_THUMB = (240, 255, 160)   # Slider thumb highlight

    # Default parameters
    defaults = {
        'clahe_clip_limit': 60,      # Stored as int, divide by 10 for actual value (6.0)
        'clahe_tile_size': 6,
        'blur_kernel_size': 3,       # Must be odd
        'blur_sigma': 3,             # Stored as int, divide by 10 for actual value
        'canny_low': 4,
        'canny_high': 45,
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
    slider_defs = [
        ('frame_idx',        'Frame',       lambda v: f"{v}",        0, max(1, total_frames - 1)),
        ('clahe_clip_limit', 'CLAHE Clip',  lambda v: f"{v/10:.1f}", 0, 100),
        ('clahe_tile_size',  'CLAHE Tile',  lambda v: f"{v}",        2, 16),
        ('blur_kernel_size', 'Blur Kernel', lambda v: f"{v if v%2==1 else v+1}", 1, 15),
        ('blur_sigma',       'Blur Sigma',  lambda v: f"{v/10:.1f}", 0, 50),
        ('canny_low',        'Canny Low',   lambda v: f"{v}",        0, 100),
        ('canny_high',       'Canny High',  lambda v: f"{v}",        0, 200),
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
    status_bar_h = 36
    gap = 2
    panel_w = (target_win_w - sidebar_w - gap) // 2
    panel_h = (target_win_h - status_bar_h - gap) // 2
    canvas_w = panel_w * 2 + gap + sidebar_w
    canvas_h = panel_h * 2 + gap + status_bar_h
    slider_row_h = 52           # Height per slider row
    slider_pad_x = 20           # Horizontal padding inside sidebar
    slider_track_h = 5          # Track bar height
    slider_thumb_r = 8          # Thumb radius
    slider_section_top = 72     # Y offset where sliders begin (below title)

    # Mutable state for mouse interaction
    # These will be set once we know the canvas geometry in the first frame.
    dragging = {'idx': -1}      # Index of slider being dragged (-1 = none)
    # Slider hit-test regions (populated by create_display, read by mouse_cb)
    slider_regions = []         # List of (x_start, x_end, y_center, min_val, max_val, param_key)

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

    # ── Composite display builder ────────────────────────────────────

    def create_display(frm, gray, enhanced, blurred, edges, p):
        nonlocal slider_regions

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = BG_DARK

        # ── Panels ───────────────────────────────────────────────────
        orig_small = cv2.resize(frm, (panel_w, panel_h))
        enh_bgr = cv2.cvtColor(cv2.resize(enhanced, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)
        blur_bgr = cv2.cvtColor(cv2.resize(blurred, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)

        edge_gray_r = cv2.resize(edges, (panel_w, panel_h))
        edge_bgr = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        edge_bgr[:] = BG_PANEL
        edge_bgr[edge_gray_r > 0] = ACCENT_CYAN

        for px, py, panel in [
            (0, 0, orig_small),
            (panel_w + gap, 0, enh_bgr),
            (0, panel_h + gap, blur_bgr),
            (panel_w + gap, panel_h + gap, edge_bgr),
        ]:
            canvas[py:py + panel_h, px:px + panel_w] = panel
            _draw_border(canvas, px, py, panel_w, panel_h, BORDER)

        # Panel tags
        tag_y, tag_x = 18, 8
        _draw_tag(canvas, "ORIGINAL", tag_x, tag_y, BG_DARK, TEXT_DIM)
        clip_val = p['clahe_clip_limit'] / 10.0
        _draw_tag(canvas, f"CLAHE  clip {clip_val:.1f}  tile {p['clahe_tile_size']}",
                  panel_w + gap + tag_x, tag_y, BG_DARK, ACCENT)
        blur_k = p['blur_kernel_size']
        if blur_k % 2 == 0:
            blur_k += 1
        _draw_tag(canvas, f"BLUR  k={blur_k}  s={p['blur_sigma']/10:.1f}",
                  tag_x, panel_h + gap + tag_y, BG_DARK, TEXT_PRIMARY)
        _draw_tag(canvas, f"EDGES  {p['canny_low']}-{p['canny_high']}",
                  panel_w + gap + tag_x, panel_h + gap + tag_y, BG_DARK, ACCENT_CYAN)

        # ── Sidebar ──────────────────────────────────────────────────
        sb_x = panel_w * 2 + gap
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

        slider_regions = new_regions

        # ── Controls help (below sliders) ────────────────────────────
        help_y = slider_section_top + len(slider_defs) * slider_row_h + 12
        cv2.line(canvas, (sb_x + 14, help_y - 6), (sb_x + sidebar_w - 14, help_y - 6), BORDER, 1)
        _put_text(canvas, "CONTROLS", sb_x + 14, help_y + 10, TEXT_HEADING, 0.36)
        help_y += 26
        for key_str, desc in [
            ("SPACE / ENTER", "Accept"),
            ("ESC", "Cancel"),
            ("R", "Reset"),
            ("N / P", "Next / Prev frame"),
        ]:
            _put_text(canvas, key_str, sb_x + 14, help_y, ACCENT_DIM, 0.30)
            _put_text(canvas, desc, sb_x + 126, help_y, TEXT_DIM, 0.30)
            help_y += 16

        # ── Status bar ───────────────────────────────────────────────
        sb_y = canvas_h - status_bar_h
        _fill_rect(canvas, 0, sb_y, canvas_w, status_bar_h, BG_PANEL)
        cv2.line(canvas, (0, sb_y), (canvas_w, sb_y), BORDER, 1)

        _put_text(canvas, f"Frame {current_frame_idx}/{total_frames}", 12, sb_y + 21, TEXT_DIM, 0.36)
        _put_text(canvas, f"{src_w}x{src_h}", canvas_w - 90, sb_y + 21, TEXT_DIM, 0.36)
        cv2.circle(canvas, (canvas_w // 2, sb_y + 16), 4, ACCENT, -1)
        _put_text(canvas, "LIVE", canvas_w // 2 + 10, sb_y + 21, ACCENT_DIM, 0.33)

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
        if event == cv2.EVENT_LBUTTONDOWN:
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
    print("\nControls:")
    print("  Click+drag   - Adjust sliders in the sidebar")
    print("  SPACE/ENTER  - Accept current settings and continue")
    print("  ESC          - Cancel and use default settings")
    print("  R            - Reset to default values")
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
            }
            print(f"\nAccepted preprocessing parameters:")
            print(f"  CLAHE clip limit: {final_params['clahe_clip_limit']:.1f}")
            print(f"  CLAHE tile size: {final_params['clahe_tile_size']}")
            print(f"  Blur kernel size: {final_params['blur_kernel_size']}")
            print(f"  Blur sigma: {final_params['blur_sigma']:.1f}")
            print(f"  Canny thresholds: {final_params['canny_low']}-{final_params['canny_high']}")
            cv2.destroyWindow(window_name)
            cap.release()
            return final_params

        elif key == ord('r') or key == ord('R'):  # Reset
            saved_frame_idx = params['frame_idx']
            params = defaults.copy()
            params['frame_idx'] = saved_frame_idx
            print("Parameters reset to defaults.")

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

    def __init__(self, sensitivity='medium', preprocessing_params=None, skip_aspect_ratio_check=False):
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
        """
        # Store custom preprocessing parameters
        self.preprocessing_params = preprocessing_params
        self.skip_aspect_ratio_check = skip_aspect_ratio_check

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

        # Apply custom canny thresholds to params if provided
        if self.preprocessing_params:
            if 'canny_low' in self.preprocessing_params:
                self.params['canny_low'] = self.preprocessing_params['canny_low']
            if 'canny_high' in self.preprocessing_params:
                self.params['canny_high'] = self.preprocessing_params['canny_high']

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

    def classify_trail(self, line, gray_frame, color_frame, hsv_frame=None, reusable_mask=None):
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

    def detect_trails(self, frame, debug_info=None):
        """
        Detect and classify trails in a frame as satellites or airplanes.

        Supports multiple simultaneous detections of each type. Airplane
        detections use angle-aware merging so that two airplanes with
        crossing or nearby paths are kept as separate detections.

        Args:
            frame: Input frame
            debug_info: Optional dict to collect debug information

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

        # Pre-compute HSV frame once for all line classifications (performance optimization)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pre-allocate reusable mask array (performance optimization)
        reusable_mask = np.zeros(gray.shape, dtype=np.uint8)

        classified_trails = []
        all_classifications = []  # For debug: store all attempted classifications

        for line in lines:
            trail_type, detection_info = self.classify_trail(line, gray, frame, hsv_frame, reusable_mask)

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

        # Store debug info
        if debug_info is not None:
            debug_info['all_lines'] = lines
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray

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

        # Combine results with type labels
        results = [('satellite', info) for info in merged_satellite_infos]
        results.extend([('airplane', info) for info in merged_airplane_infos])

        return results
    
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


def process_video(input_path, output_path, sensitivity='medium', freeze_duration=1.0, max_duration=None, detect_type='both', show_labels=True, debug_mode=False, debug_only=False, preprocessing_params=None, skip_aspect_ratio_check=False):
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
    detector = SatelliteTrailDetector(sensitivity, preprocessing_params=preprocessing_params, skip_aspect_ratio_check=skip_aspect_ratio_check)

    frame_count = 0
    satellites_detected = 0
    airplanes_detected = 0

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

        # Detect and classify trails (with debug info if needed)
        debug_info = {} if debug_mode else None
        detected_trails = detector.detect_trails(frame, debug_info=debug_info)

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

    args = parser.parse_args()

    # Handle preprocessing preview if requested
    preprocessing_params = None
    if args.preview:
        preprocessing_params = show_preprocessing_preview(args.input)
        if preprocessing_params is None:
            print("Using default preprocessing parameters.")

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
        skip_aspect_ratio_check=args.no_aspect_ratio_check
    )


if __name__ == '__main__':
    main()
