"""End-to-end pipeline test: synthetic video in, annotated video out."""
import os
import tempfile
import unittest

import cv2
import numpy as np

import satellite_trail_detector as m


def _synthesize_video(path, n_frames=10, h=240, w=320, fps=15):
    """Write an MP4 with a dim background and a bright diagonal trail on
    frames 3-5 (simulating a satellite transit)."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    rng = np.random.default_rng(42)
    try:
        for i in range(n_frames):
            frame = rng.integers(25, 50, (h, w, 3), dtype=np.uint8)
            if 3 <= i <= 5:
                # Draw a dim streak shifting with frame index.
                dx = (i - 3) * 4
                cv2.line(frame,
                         (40 + dx, 40 + dx), (280 + dx, 200 + dx),
                         (180, 180, 180), 2)
            writer.write(frame)
    finally:
        writer.release()


class TestPipelineEndToEnd(unittest.TestCase):

    def test_default_algorithm_produces_output(self):
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, 'synth.mp4')
            out_path = os.path.join(td, 'out.mp4')
            _synthesize_video(in_path)
            self.assertTrue(os.path.exists(in_path))
            self.assertGreater(os.path.getsize(in_path), 1000)

            m.process_video(
                input_path=in_path, output_path=out_path,
                sensitivity='high', freeze_duration=0.0,
                num_workers=0,            # sequential for determinism
                no_gpu=True,              # exercise CPU path
                show_labels=False,
            )

            self.assertTrue(os.path.exists(out_path),
                            f"output video not created: {out_path}")
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_translation_ledger_emits_summary(self):
        """TranslationLedger accumulates rejections and emits a summary."""
        ledger = m.TranslationLedger()
        # Reasons are stripped of the 'rejected_' prefix at the call site.
        ledger.record_rejection('too_short')
        ledger.record_rejection('too_short')
        ledger.record_rejection('low_contrast')
        ledger.record_classification('satellite')
        # Counters must have incremented.
        self.assertEqual(ledger.rejected_too_short, 2)
        self.assertEqual(ledger.rejected_low_contrast, 1)
        self.assertEqual(ledger.classified_satellite, 1)
        # Summary renders once we also have primary-line detections.
        ledger.total_lines_detected = 10
        lines = ledger.summary_lines()
        self.assertIsInstance(lines, list)
        self.assertTrue(any('too short' in ln for ln in lines))
        # to_dict must be JSON-serialisable.
        import json
        json.dumps(ledger.to_dict())

    def test_annotation_database_roundtrip(self):
        """Write, save, and reload an annotation DB."""
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'ann.json')
            db = m.AnnotationDatabase()
            db.start_session('synth.mp4', 'medium', 'default', {})
            img_id = db.add_image(0, 'synth.mp4', 320, 240)
            db.add_detection(img_id, 0, (100, 100, 300, 120),
                              {'angle': 45.0, 'length': 200.0,
                               'avg_brightness': 20.0, 'contrast_ratio': 1.2,
                               'trail_snr': 3.5},
                              {'satellite_contrast_min': 1.08}, 0.8)
            db.save(p)
            self.assertTrue(os.path.exists(p))

            db2 = m.AnnotationDatabase(p)
            self.assertEqual(len(db2.data['annotations']), 1)


if __name__ == '__main__':
    unittest.main()
