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

    Opens a window with a sample frame from the video and trackbars to adjust:
    - Frame selection (slider to choose exact frame with trail signal)
    - CLAHE clip limit (contrast enhancement strength)
    - CLAHE tile grid size (local region size for contrast)
    - Gaussian blur kernel size (smoothing extent)
    - Gaussian blur sigma (smoothing intensity)
    - Canny edge detection thresholds (for visualization)

    Controls:
    - Use Frame slider to select exact frame containing trail signal
    - Adjust other sliders to see real-time preprocessing effect
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
    # Default parameters
    defaults = {
        'clahe_clip_limit': 40,      # Stored as int, divide by 10 for actual value
        'clahe_tile_size': 6,
        'blur_kernel_size': 3,       # Must be odd
        'blur_sigma': 3,             # Stored as int, divide by 10 for actual value
        'canny_low': 5,
        'canny_high': 50,
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

    # Sample frames at different points in the video for user to choose from
    # Start with frame at 10% into the video (to skip any intro)
    current_frame_idx = max(0, int(total_frames * 0.1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

    ret, frame = cap.read()
    if not ret:
        # Try from beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        current_frame_idx = 0
        if not ret:
            print("Error: Could not read any frame from video")
            cap.release()
            return None

    # Store original frame for reset
    original_frame = frame.copy()
    height, width = frame.shape[:2]

    # Create window
    window_name = "Preprocessing Preview - Adjust parameters and press SPACE/ENTER to confirm, ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Resize window to fit screen while maintaining aspect ratio
    screen_scale = min(1.0, 1400 / width, 900 / height)
    display_width = int(width * screen_scale)
    display_height = int(height * screen_scale)
    cv2.resizeWindow(window_name, display_width, display_height)

    # Current parameter values (mutable)
    params = defaults.copy()

    # Flag to trigger update
    needs_update = [True]  # Using list to allow modification in nested function

    # Track current frame index (mutable for callback access)
    frame_state = {
        'current_idx': current_frame_idx,
        'frame': frame,
        'needs_frame_update': False
    }

    def on_trackbar_change(val):
        """Callback when any trackbar changes."""
        needs_update[0] = True

    def on_frame_trackbar_change(val):
        """Callback when frame slider changes."""
        if val != frame_state['current_idx']:
            frame_state['needs_frame_update'] = True
        needs_update[0] = True

    # Create frame selection trackbar FIRST (appears at top)
    # Use frame index from 0 to total_frames-1
    cv2.createTrackbar("Frame", window_name, current_frame_idx, max(1, total_frames - 1), on_frame_trackbar_change)

    # Create preprocessing parameter trackbars
    # CLAHE clip limit: 1-100 (divided by 10 = 0.1 to 10.0)
    cv2.createTrackbar("CLAHE Clip (x0.1)", window_name, params['clahe_clip_limit'], 100, on_trackbar_change)

    # CLAHE tile size: 2-16
    cv2.createTrackbar("CLAHE Tile Size", window_name, params['clahe_tile_size'], 16, on_trackbar_change)

    # Blur kernel size: 1-15 (will be forced to odd)
    cv2.createTrackbar("Blur Kernel", window_name, params['blur_kernel_size'], 15, on_trackbar_change)

    # Blur sigma: 0-50 (divided by 10 = 0.0 to 5.0)
    cv2.createTrackbar("Blur Sigma (x0.1)", window_name, params['blur_sigma'], 50, on_trackbar_change)

    # Canny thresholds for edge visualization
    cv2.createTrackbar("Canny Low", window_name, params['canny_low'], 100, on_trackbar_change)
    cv2.createTrackbar("Canny High", window_name, params['canny_high'], 200, on_trackbar_change)

    def apply_preprocessing(frame, params):
        """Apply preprocessing with current parameters."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get actual parameter values
        clip_limit = max(0.1, params['clahe_clip_limit'] / 10.0)
        tile_size = max(2, params['clahe_tile_size'])
        blur_kernel = params['blur_kernel_size']
        blur_sigma = params['blur_sigma'] / 10.0

        # Ensure blur kernel is odd
        if blur_kernel < 1:
            blur_kernel = 1
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur
        if blur_kernel >= 1 and blur_sigma > 0:
            blurred = cv2.GaussianBlur(enhanced, (blur_kernel, blur_kernel), blur_sigma)
        else:
            blurred = enhanced

        # Apply Canny edge detection for visualization
        canny_low = max(1, params['canny_low'])
        canny_high = max(canny_low + 1, params['canny_high'])
        edges = cv2.Canny(blurred, canny_low, canny_high)

        return gray, enhanced, blurred, edges

    def create_display(frame, gray, enhanced, blurred, edges, params):
        """Create a display showing original, preprocessed stages, and edges."""
        h, w = frame.shape[:2]

        # Create 2x2 grid: Original | CLAHE Enhanced
        #                  Blurred  | Edges
        # Each panel is half size
        panel_h = h // 2
        panel_w = w // 2

        # Resize panels
        original_small = cv2.resize(frame, (panel_w, panel_h))
        enhanced_bgr = cv2.cvtColor(cv2.resize(enhanced, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)
        blurred_bgr = cv2.cvtColor(cv2.resize(blurred, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)
        edges_bgr = cv2.cvtColor(cv2.resize(edges, (panel_w, panel_h)), cv2.COLOR_GRAY2BGR)

        # Create labels background
        def add_label(img, text, position="top"):
            overlay = img.copy()
            if position == "top":
                cv2.rectangle(overlay, (0, 0), (len(text) * 12 + 10, 25), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
                cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return img

        # Add labels
        add_label(original_small, "1. Original Frame")
        add_label(enhanced_bgr, f"2. CLAHE (clip={params['clahe_clip_limit']/10:.1f}, tile={params['clahe_tile_size']})")

        blur_k = params['blur_kernel_size']
        if blur_k % 2 == 0:
            blur_k += 1
        add_label(blurred_bgr, f"3. Gaussian Blur (k={blur_k}, s={params['blur_sigma']/10:.1f})")
        add_label(edges_bgr, f"4. Canny Edges ({params['canny_low']}-{params['canny_high']})")

        # Combine into grid
        top_row = np.hstack([original_small, enhanced_bgr])
        bottom_row = np.hstack([blurred_bgr, edges_bgr])
        display = np.vstack([top_row, bottom_row])

        # Add instructions at the bottom
        instruction_bar = np.zeros((40, display.shape[1], 3), dtype=np.uint8)
        instructions = "SPACE/ENTER: Accept | ESC: Cancel | R: Reset | N/P: Next/Prev frame"
        cv2.putText(instruction_bar, instructions, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Add frame info
        frame_info = f"Frame {current_frame_idx}/{total_frames} | {width}x{height}"
        cv2.putText(instruction_bar, frame_info, (display.shape[1] - 250, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        display = np.vstack([display, instruction_bar])

        return display

    print("\n" + "=" * 60)
    print("PREPROCESSING PREVIEW")
    print("=" * 60)
    print("Use the Frame slider to find a frame with satellite trail signal.")
    print("Then adjust other sliders to tune preprocessing parameters.")
    print("The goal is to preserve dim satellite trails while reducing noise.")
    print("\nControls:")
    print("  Frame slider - Select exact frame to preview")
    print("  SPACE/ENTER  - Accept current settings and continue")
    print("  ESC          - Cancel and use default settings")
    print("  R            - Reset to default values")
    print("  N            - Jump forward 1 second")
    print("  P            - Jump back 1 second")
    print("=" * 60 + "\n")

    # Main loop
    while True:
        # Read frame slider and load new frame if changed
        slider_frame_idx = cv2.getTrackbarPos("Frame", window_name)
        if slider_frame_idx != current_frame_idx:
            current_frame_idx = slider_frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                frame_state['current_idx'] = current_frame_idx
                frame_state['frame'] = frame

        # Read current trackbar values
        params['clahe_clip_limit'] = cv2.getTrackbarPos("CLAHE Clip (x0.1)", window_name)
        params['clahe_tile_size'] = cv2.getTrackbarPos("CLAHE Tile Size", window_name)
        params['blur_kernel_size'] = cv2.getTrackbarPos("Blur Kernel", window_name)
        params['blur_sigma'] = cv2.getTrackbarPos("Blur Sigma (x0.1)", window_name)
        params['canny_low'] = cv2.getTrackbarPos("Canny Low", window_name)
        params['canny_high'] = cv2.getTrackbarPos("Canny High", window_name)

        # Apply preprocessing
        gray, enhanced, blurred, edges = apply_preprocessing(frame, params)

        # Create display
        display = create_display(frame, gray, enhanced, blurred, edges, params)

        # Show
        cv2.imshow(window_name, display)

        # Wait for key
        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC - cancel
            print("Preview cancelled. Using default parameters.")
            cv2.destroyWindow(window_name)
            cap.release()
            return None

        elif key in [13, 32]:  # ENTER or SPACE - accept
            # Convert to actual values
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
            params = defaults.copy()
            cv2.setTrackbarPos("CLAHE Clip (x0.1)", window_name, params['clahe_clip_limit'])
            cv2.setTrackbarPos("CLAHE Tile Size", window_name, params['clahe_tile_size'])
            cv2.setTrackbarPos("Blur Kernel", window_name, params['blur_kernel_size'])
            cv2.setTrackbarPos("Blur Sigma (x0.1)", window_name, params['blur_sigma'])
            cv2.setTrackbarPos("Canny Low", window_name, params['canny_low'])
            cv2.setTrackbarPos("Canny High", window_name, params['canny_high'])
            print("Parameters reset to defaults.")

        elif key == ord('n') or key == ord('N'):  # Next frame
            # Jump forward by 1 second worth of frames
            current_frame_idx = min(total_frames - 1, current_frame_idx + int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if not ret:
                current_frame_idx = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                frame_state['current_idx'] = current_frame_idx
                cv2.setTrackbarPos("Frame", window_name, current_frame_idx)

        elif key == ord('p') or key == ord('P'):  # Previous frame
            # Jump back by 1 second worth of frames
            current_frame_idx = max(0, current_frame_idx - int(fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, new_frame = cap.read()
            if ret:
                frame = new_frame
                frame_state['current_idx'] = current_frame_idx
                cv2.setTrackbarPos("Frame", window_name, current_frame_idx)

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
            Tuple of (trail_type, bbox) where trail_type is 'satellite', 'airplane', or None
        """
        pass

    def detect_trails(self, frame, debug_info=None):
        """
        Main detection pipeline. Can be overridden for completely custom logic.

        Args:
            frame: Input frame
            debug_info: Optional dict to collect debug information

        Returns:
            List of tuples: [('satellite', bbox), ('airplane', bbox), ...]
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
            trail_type, bbox = self.classify_trail(line, gray, frame)

            if debug_info is not None:
                all_classifications.append({
                    'line': line,
                    'type': trail_type,
                    'bbox': bbox
                })

            if trail_type and bbox:
                classified_trails.append((trail_type, bbox))

        if debug_info is not None:
            debug_info['all_lines'] = lines
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray

        # Separate by type for merging
        satellite_boxes = [bbox for t, bbox in classified_trails if t == 'satellite']
        airplane_boxes = [bbox for t, bbox in classified_trails if t == 'airplane']

        # Merge overlapping detections within each type
        merged_satellites = self.merge_overlapping_boxes(satellite_boxes)
        merged_airplanes = self.merge_overlapping_boxes(airplane_boxes)

        # Combine results with type labels
        results = [('satellite', bbox) for bbox in merged_satellites]
        results.extend([('airplane', bbox) for bbox in merged_airplanes])

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

        # Count distinct peaks (consecutive peaks count as one)
        num_peaks = 0
        in_peak = False
        peak_indices = []  # Store indices of peak starts for debug visualization

        for i, is_peak in enumerate(peaks):
            if is_peak and not in_peak:
                num_peaks += 1
                peak_indices.append(i)
                in_peak = True
            elif not is_peak:
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

    def __init__(self, sensitivity='medium', preprocessing_params=None):
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
        """
        # Store custom preprocessing parameters
        self.preprocessing_params = preprocessing_params

        # Sensitivity presets - rebalanced to reduce false positives
        presets = {
            'low': {
                'min_line_length': 80,  # Longer minimum to reduce noise
                'max_line_gap': 30,  # Moderate gap tolerance
                'canny_low': 8,  # Less sensitive to reduce edge noise
                'canny_high': 60,
                'hough_threshold': 45,  # Higher threshold for fewer false detections
                'min_aspect_ratio': 4,  # Stricter aspect ratio for true trails
                'brightness_threshold': 25,
                'airplane_brightness_min': 90,
                'airplane_saturation_min': 10,
                'satellite_min_length': 180,  # Satellite trail length (1920x1080)
                'satellite_max_length': 300,
            },
            'medium': {
                'min_line_length': 60,  # Balanced length requirement
                'max_line_gap': 35,  # Balanced gap tolerance
                'canny_low': 5,  # Balanced edge detection
                'canny_high': 50,
                'hough_threshold': 35,  # Balanced threshold
                'min_aspect_ratio': 4,  # Require trails to be relatively long and thin
                'brightness_threshold': 18,
                'airplane_brightness_min': 75,
                'airplane_saturation_min': 8,
                'satellite_min_length': 180,
                'satellite_max_length': 300,
            },
            'high': {
                'min_line_length': 45,  # Still catches shorter trails
                'max_line_gap': 40,  # More tolerant of breaks
                'canny_low': 3,  # More sensitive edge detection
                'canny_high': 40,
                'hough_threshold': 25,  # Lower threshold for more detections
                'min_aspect_ratio': 3,  # More relaxed but not too permissive
                'brightness_threshold': 12,
                'airplane_brightness_min': 45,
                'airplane_saturation_min': 2,
                'satellite_min_length': 100,  # Slightly lower for high sensitivity
                'satellite_max_length': 500,
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

    def preprocess_frame(self, frame):
        """Convert frame to grayscale and enhance for trail detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get preprocessing parameters (use custom if available, otherwise defaults)
        if self.preprocessing_params:
            clip_limit = self.preprocessing_params.get('clahe_clip_limit', 4.0)
            tile_size = self.preprocessing_params.get('clahe_tile_size', 6)
            blur_kernel = self.preprocessing_params.get('blur_kernel_size', 3)
            blur_sigma = self.preprocessing_params.get('blur_sigma', 0.3)
        else:
            clip_limit = 4.0
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

        # Count distinct peaks (consecutive peaks count as one)
        num_peaks = 0
        in_peak = False
        peak_indices = []  # Store indices of peak starts for debug visualization

        for i, is_peak in enumerate(peaks):
            if is_peak and not in_peak:
                num_peaks += 1
                peak_indices.append(i)
                in_peak = True
            elif not is_peak:
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

    def classify_trail(self, line, gray_frame, color_frame):
        """
        Classify a detected line as either a satellite or airplane trail.

        Key distinction:
        - Airplanes: DOTTED features - bright point-like lights along the trail (navigation lights)
                    Sometimes colorful dots (red, green, white). Can be any length including 180-300px.
        - Satellites: SMOOTH, consistent brightness along trail. No bright point features.
                     Dim, monochromatic, uniform appearance. Typically 180-300 pixels for 1920x1080.

        Returns:
            trail_type: 'satellite', 'airplane', or None
            bbox: Bounding box if trail detected, None otherwise
        """
        x1, y1, x2, y2 = line[0]

        # Calculate line properties
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if length < self.params['min_line_length']:
            return None, None

        # Create a mask for the line region
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
        # Sample surrounding area to check if trail is actually brighter
        surround_sample_size = 30
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Get background brightness from area around the trail
        bg_x_min = max(0, x_center - surround_sample_size)
        bg_y_min = max(0, y_center - surround_sample_size)
        bg_x_max = min(gray_frame.shape[1], x_center + surround_sample_size)
        bg_y_max = min(gray_frame.shape[0], y_center + surround_sample_size)

        background_region = gray_frame[bg_y_min:bg_y_max, bg_x_min:bg_x_max]
        if background_region.size > 0:
            background_brightness = np.median(background_region)
            # Trail should be at least 20% brighter than background
            contrast_ratio = avg_brightness / (background_brightness + 1e-5)
            if contrast_ratio < 1.2:
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

        if aspect_ratio < self.params['min_aspect_ratio']:
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

        # Convert to HSV to check color saturation
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
        # Sort pixels by brightness to find brightest spots
        if len(trail_pixels_gray) > 20:
            sorted_pixels = np.sort(trail_pixels_gray)
            top_10_percent_count = max(1, len(sorted_pixels) // 10)
            top_10_percent_mean = np.mean(sorted_pixels[-top_10_percent_count:])

            # If the brightest 10% of pixels are significantly brighter than average, it's dotted
            brightness_peak_ratio = top_10_percent_mean / (avg_brightness + 1e-5)
            has_bright_spots = brightness_peak_ratio > 1.5  # Bright spots significantly brighter
        else:
            has_bright_spots = False
            brightness_peak_ratio = 1.0

        # Check for high brightness variance (indicates non-uniform, dotted pattern)
        has_high_variance = brightness_variation > 0.30  # Balanced threshold

        # SPATIAL ANALYSIS: Detect distinct point-like features along the trail
        num_point_features = self.detect_point_features(line, gray_frame)
        has_multiple_points = num_point_features >= 2  # At least 2 distinct bright points

        bbox = (x_min, y_min, x_max, y_max)

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
            return 'airplane', bbox

        # If multiple distinct point features detected (navigation lights pattern)
        # Require higher brightness to avoid false positives
        if has_multiple_points and max_brightness > 120 and is_bright:
            return 'airplane', bbox

        # Calculate airplane score - require more evidence
        airplane_score = sum([is_bright, is_colorful, has_color_variation, has_dotted_pattern])

        # Require dotted pattern AND at least 2 other characteristics
        if has_dotted_pattern and airplane_score >= 3:
            return 'airplane', bbox

        # Very strong dotted pattern with high brightness
        if has_dotted_pattern and brightness_peak_ratio > 2.0 and max_brightness > 120:
            return 'airplane', bbox

        # SATELLITE DETECTION CRITERIA
        # Satellites have SMOOTH, consistent brightness (no dotted features)
        # Typically dim, monochromatic, 180-300px length
        is_dim = avg_brightness < self.params['airplane_brightness_min']
        is_monochrome = avg_saturation < self.params['airplane_saturation_min']
        is_smooth = brightness_variation < 0.35 and not has_bright_spots  # Smooth, no bright points
        is_satellite_length = self.params['satellite_min_length'] <= length <= self.params['satellite_max_length']

        satellite_score = sum([is_dim, is_monochrome, is_smooth, is_satellite_length])

        # Require ALL 4 satellite characteristics to be confident
        if satellite_score >= 4 and not has_dotted_pattern:
            return 'satellite', bbox

        # Require at least 3 characteristics including smoothness and length
        if satellite_score >= 3 and is_smooth and is_satellite_length and not has_dotted_pattern:
            return 'satellite', bbox

        # Very dim, smooth trails in correct length range
        if is_smooth and avg_brightness <= self.params['brightness_threshold'] * 1.5 and is_satellite_length and not has_dotted_pattern:
            return 'satellite', bbox

        return None, None
    
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
    
    def detect_trails(self, frame, debug_info=None):
        """
        Detect and classify trails in a frame as satellites or airplanes.

        Args:
            frame: Input frame
            debug_info: Optional dict to collect debug information

        Returns:
            List of tuples: [('satellite', bbox), ('airplane', bbox), ...]
            where bbox is (x_min, y_min, x_max, y_max)
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
        all_classifications = []  # For debug: store all attempted classifications

        for line in lines:
            trail_type, bbox = self.classify_trail(line, gray, frame)

            # Store for debug (even if filtered out)
            if debug_info is not None:
                all_classifications.append({
                    'line': line,
                    'type': trail_type,
                    'bbox': bbox
                })

            if trail_type and bbox:
                classified_trails.append((trail_type, bbox))

        # Store debug info
        if debug_info is not None:
            debug_info['all_lines'] = lines
            debug_info['all_classifications'] = all_classifications
            debug_info['edges'] = edges
            debug_info['gray_frame'] = gray

        # Separate by type for merging
        satellite_boxes = [bbox for t, bbox in classified_trails if t == 'satellite']
        airplane_boxes = [bbox for t, bbox in classified_trails if t == 'airplane']

        # Merge overlapping detections within each type
        merged_satellites = self.merge_overlapping_boxes(satellite_boxes)
        merged_airplanes = self.merge_overlapping_boxes(airplane_boxes)

        # Combine results with type labels
        results = [('satellite', bbox) for bbox in merged_satellites]
        results.extend([('airplane', bbox) for bbox in merged_airplanes])

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


def process_video(input_path, output_path, sensitivity='medium', freeze_duration=1.0, max_duration=None, detect_type='both', show_labels=True, debug_mode=False, debug_only=False, preprocessing_params=None):
    """
    Process video to detect and highlight satellite and airplane trails.

    Output video maintains the same resolution and frame rate as input.
    Uses H.264 codec when available for best quality preservation.

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

    # Initialize video writer with high-quality codec
    # Try H.264 codec first (best quality), fall back to mp4v if not available
    codecs_to_try = [
        ('avc1', 'H.264'),
        ('h264', 'H.264'),
        ('H264', 'H.264'),
        ('X264', 'H.264'),
        ('mp4v', 'MPEG-4')
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
    detector = SatelliteTrailDetector(sensitivity, preprocessing_params=preprocessing_params)

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
            for trail_type, bbox in detected_trails:
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

                # Add to frozen regions list
                frozen_regions.append({
                    'region': frozen_region,
                    'bbox': freeze_bbox,
                    'trail_type': trail_type,
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
    - Uses H.264 codec for best quality when available
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
        preprocessing_params=preprocessing_params
    )


if __name__ == '__main__':
    main()
