"""Detector instantiation + basic call tests for all 3 algorithms."""
import unittest

import numpy as np

import satellite_trail_detector as m


class TestDetectors(unittest.TestCase):

    @staticmethod
    def _synthetic_frame_with_trail(h=240, w=320, seed=0):
        """Dim background + bright diagonal streak."""
        rng = np.random.default_rng(seed)
        frame = rng.integers(20, 50, (h, w, 3), dtype=np.uint8)
        import cv2
        cv2.line(frame, (40, 40), (280, 200), (200, 200, 200), 2)
        return frame

    def test_default_detector_preprocess(self):
        det = m.SatelliteTrailDetector(sensitivity='medium')
        frame = self._synthetic_frame_with_trail()
        gray, blurred = det.preprocess_frame(frame)
        self.assertEqual(gray.shape, frame.shape[:2])
        self.assertEqual(blurred.shape, frame.shape[:2])

    def test_default_detector_detect_lines(self):
        det = m.SatelliteTrailDetector(sensitivity='medium')
        frame = self._synthetic_frame_with_trail()
        _, blurred = det.preprocess_frame(frame)
        lines, edges = det.detect_lines(blurred)
        # (N, 1, 4) array of Hough line endpoints + edge map.
        self.assertEqual(lines.ndim, 3)
        self.assertEqual(lines.shape[2], 4)
        self.assertEqual(edges.shape, frame.shape[:2])

    def test_default_detector_full_detect_trails(self):
        det = m.SatelliteTrailDetector(sensitivity='high')
        frame = self._synthetic_frame_with_trail()
        trails = det.detect_trails(frame)
        self.assertIsInstance(trails, list)
        # Each detection is (trail_type, info) and info carries a bbox.
        for trail_type, info in trails:
            self.assertIn(trail_type, ('satellite', 'airplane', 'anomalous'))
            self.assertIn('bbox', info)
            self.assertEqual(len(info['bbox']), 4)

    def test_radon_detector_instantiation(self):
        det = m.RadonStreakDetector(sensitivity='medium')
        self.assertIsInstance(det, m.SatelliteTrailDetector)

    def test_radon_detector_detect_trails(self):
        det = m.RadonStreakDetector(sensitivity='high')
        frame = self._synthetic_frame_with_trail()
        trails = det.detect_trails(frame)
        self.assertIsInstance(trails, list)

    def test_nn_detector_fusion_flag_plumbing(self):
        """Construct NeuralNetDetector with fusion_enabled=True and verify
        the field is stored without needing a real model file."""
        det = m.NeuralNetDetector(
            sensitivity='medium',
            model_path='does-not-exist.pt',
            backend='ultralytics',
            fusion_enabled=True,
        )
        self.assertTrue(det.fusion_enabled)
        self.assertIsNone(det._fusion_head)  # lazy

    def test_nn_detector_fusion_lazy_load_gracefully(self):
        det = m.NeuralNetDetector(
            sensitivity='medium',
            model_path='does-not-exist.pt',
            fusion_enabled=True,
        )
        # Fusion head loader must not raise even if no saved weights exist.
        fh = det._get_fusion_head()
        # Either a valid AlgorithmFusionHead (degrading to max-score) or None.
        self.assertTrue(fh is None or isinstance(fh, m.AlgorithmFusionHead))


if __name__ == '__main__':
    unittest.main()
