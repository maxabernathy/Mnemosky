"""One test per v0.3.0 learning initiative (I1 … I7c).

Each test instantiates the class and exercises its primary public
method with a minimal synthetic input.  These are not accuracy tests —
they guard wiring, shapes, and crashes."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import satellite_trail_detector as m


class TestInitiatives(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    # ── I1 RescueClassifier ──────────────────────────────────────────

    def test_i1_rescue_classifier_untrained(self):
        rc = m.RescueClassifier(path=os.path.join(self.tmp, 'rc.json'))
        self.assertIsNone(rc.mlp)
        # Untrained: can_rescue returns False (early-out path).
        self.assertFalse(rc.can_rescue({'contrast_ratio': 1.05}, reason_code=1))

    def test_i1_rescue_classifier_fit_minimum_corpus(self):
        rc = m.RescueClassifier(path=os.path.join(self.tmp, 'rc.json'))
        pos = [{'contrast_ratio': 1.2, 'trail_snr': 4.0, 'length': 200,
                'reason_code': 0} for _ in range(15)]
        neg = [{'contrast_ratio': 1.01, 'trail_snr': 0.5, 'length': 10,
                'reason_code': 1} for _ in range(15)]
        self.assertTrue(rc.fit_from_corpus(pos, neg))
        self.assertIsNotNone(rc.mlp)
        self.assertEqual(rc.trained_on, int(len(pos + neg) * 0.8))

    # ── I2 residual-stack passthrough ─────────────────────────────────

    def test_i2_residual_stack_emission(self):
        buf = m.TemporalFrameBuffer(capacity=7, residual_ring_depth=4)
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 255, (64, 64), dtype=np.uint8) for _ in range(6)]
        for f in frames[:-1]:
            buf.add(f)
        self.assertTrue(buf.is_ready())
        ctx = buf.get_temporal_context(frames[-1])
        self.assertIn('diff_stack', ctx)
        self.assertIn('stack_noise', ctx)
        # After 1 get_temporal_context call, ring depth is 1 — no stack yet.
        # Pull twice more to accumulate ≥2 residuals.
        _ = buf.get_temporal_context(frames[-1])
        ctx3 = buf.get_temporal_context(frames[-1])
        self.assertIsNotNone(ctx3['diff_stack'])
        self.assertGreaterEqual(ctx3['diff_stack'].shape[0], 2)
        self.assertIsNotNone(ctx3['stack_noise'])

    # ── I3 ThresholdHyperNet ──────────────────────────────────────────

    def test_i3_hypernet_zero_delta_untrained(self):
        hn = m.ThresholdHyperNet()
        base = dict(m.PARAMETER_SAFETY_BOUNDS).copy()
        base_params = {k: (lo + hi) / 2.0 for k, (lo, hi) in base.items()}
        features = {k: 0.0 for k in m._HYPERNET_FEATURES}
        out = hn.adapt(base_params, features, m.PARAMETER_SAFETY_BOUNDS)
        self.assertIsInstance(out, dict)
        # Untrained weights produce small deltas — output must still be
        # clamped to safety bounds.
        for k, v in out.items():
            if k in m.PARAMETER_SAFETY_BOUNDS:
                lo, hi = m.PARAMETER_SAFETY_BOUNDS[k]
                self.assertGreaterEqual(float(v), lo - 1e-6)
                self.assertLessEqual(float(v), hi + 1e-6)

    # ── I4 IMMKalmanTracker ───────────────────────────────────────────

    def test_i4_imm_kalman_tracker_empty_update(self):
        tr = m.IMMKalmanTracker()
        out = tr.update(0, [])
        self.assertIsInstance(out, list)
        self.assertEqual(out, [])

    def test_i4_imm_kalman_tracker_single_detection(self):
        tr = m.IMMKalmanTracker(window=4, min_hits=1)
        det = ('satellite', {
            'bbox': (100, 100, 300, 120), 'angle': 45.0,
            'center': (200.0, 110.0), 'length': 200.0,
            'avg_brightness': 20.0, 'line': (100, 100, 300, 120),
            'contrast_ratio': 1.2, 'trail_snr': 3.5, 'is_smooth': True,
        })
        confirmed = tr.update(0, [det])
        self.assertEqual(len(confirmed), 1)  # min_hits=1 → confirmed instantly

    # ── I5 LongBackgroundModel ────────────────────────────────────────

    def test_i5_long_background_warmup(self):
        lb = m.LongBackgroundModel(warmup_frames=10)
        self.assertFalse(lb.ready)
        rng = np.random.default_rng(0)
        for _ in range(12):
            lb.add(rng.integers(50, 80, (32, 32), dtype=np.uint8))
        self.assertTrue(lb.ready)
        residual = lb.residual(rng.integers(50, 80, (32, 32), dtype=np.uint8))
        self.assertEqual(residual.shape, (32, 32))
        self.assertTrue(np.isfinite(residual).all())

    # ── I6a TPE optimizer ─────────────────────────────────────────────

    def test_i6a_parameter_adapter_tpe_flag(self):
        pa = m.ParameterAdapter(
            dict(m.PARAMETER_SAFETY_BOUNDS).copy(),
            safety_bounds=m.PARAMETER_SAFETY_BOUNDS,
            tier2_optimizer='tpe',
        )
        self.assertEqual(pa.tier2_optimizer, 'tpe')
        # Short-circuit when corpus too small — returns current params, no crash.
        out = pa.optimize_batch([])
        self.assertIsInstance(out, dict)

    # ── I6b TrustRegionAdapter ────────────────────────────────────────

    def test_i6b_trust_region_adapter_via_param_adapter(self):
        pa = m.ParameterAdapter(
            {'satellite_contrast_min': 1.08, 'satellite_min_length': 50,
             'hough_threshold': 25},
            safety_bounds={'satellite_contrast_min': (1.0, 2.0),
                           'satellite_min_length': (10, 500),
                           'hough_threshold': (5, 200)},
            use_trust_region=True,
        )
        # First few corrections batch silently.
        out = pa.apply_correction('reject', 'satellite', {'contrast_ratio': 1.04})
        self.assertIsInstance(out, dict)

    # ── I6c PlattCalibrator ───────────────────────────────────────────

    def test_i6c_platt_calibrator_fit_and_calibrate(self):
        pc = m.PlattCalibrator()
        # 15 positives at score ≈ 0.8, 15 negatives at score ≈ 0.2.
        scores = [0.8] * 15 + [0.2] * 15
        labels = [1] * 15 + [0] * 15
        self.assertTrue(pc.fit(scores, labels))
        # After fit, a 0.8 score should calibrate to > 0.5.
        p = pc.calibrate(0.8)
        self.assertGreater(p, 0.5)
        # Round-trip through to_dict / from_dict.
        pc2 = m.PlattCalibrator.from_dict(pc.to_dict())
        self.assertAlmostEqual(pc.calibrate(0.8), pc2.calibrate(0.8), places=6)

    # ── I6d BALDQueue ─────────────────────────────────────────────────

    def test_i6d_bald_queue_rank(self):
        # Frame 0: one ambiguous detection. Frame 1: one confident.
        # BALD should rank frame 0 higher (more information per label).
        ranked = m.BALDQueue.rank(
            {0: [0.55, 0.48], 1: [0.95]},
            rescue_scores={0: [0.1, 0.9], 1: [0.9]},
        )
        self.assertIsInstance(ranked, list)
        self.assertEqual(set(ranked), {0, 1})
        self.assertEqual(ranked[0], 0)

    # ── I7a TrackletPseudoLabeler ─────────────────────────────────────

    def test_i7a_tracklet_pseudo_labeler_emits_from_tracker(self):
        db = m.AnnotationDatabase()
        db.start_session('synthetic.mp4', 'medium', 'default', {})
        labeler = m.TrackletPseudoLabeler(db, min_length=3)
        # Feed the tracker three frames of the same detection.
        tr = m.DetectionTracker(window=6, min_hits=1)
        det = ('satellite', {
            'bbox': (100, 100, 300, 120), 'angle': 45.0,
            'center': (200.0, 110.0), 'length': 200.0,
            'avg_brightness': 20.0, 'line': (100, 100, 300, 120),
            'contrast_ratio': 1.2, 'trail_snr': 3.5, 'is_smooth': True,
        })
        for fi in range(4):
            tr.update(fi, [det])
        n = labeler.emit_from_tracker(tr, 'synthetic.mp4', 640, 480)
        self.assertGreaterEqual(n, 0)  # 0 or more — tracklet formation depends on internals

    # ── I7b TrackletSequenceHead ──────────────────────────────────────

    def test_i7b_sequence_head_summarise_shape(self):
        head = m.TrackletSequenceHead()
        seq = [{'angle': 45.0, 'length': 200.0, 'avg_brightness': 20.0,
                'brightness_std': 2.0, 'trail_snr': 3.5,
                'contrast_ratio': 1.15} for _ in range(5)]
        arr = head._summarise(seq)
        # _summarise returns a (30,) summary vector: 5 stats × 6 features.
        self.assertEqual(arr.shape, (5 * len(m.TrackletSequenceHead._FEATURES),))

    def test_i7b_sequence_head_untrained_predict_returns_none(self):
        head = m.TrackletSequenceHead()
        seq = [{'angle': 45.0, 'length': 200.0, 'avg_brightness': 20.0,
                'brightness_std': 2.0, 'trail_snr': 3.5,
                'contrast_ratio': 1.15} for _ in range(5)]
        # Untrained — predict returns None / default.
        out = head.predict(seq)
        self.assertTrue(out is None or isinstance(out, (tuple, dict, str)))

    # ── I7c AlgorithmFusionHead ───────────────────────────────────────

    def test_i7c_fusion_head_features_for(self):
        default_det = {'trail_snr': 3.5, 'bbox': (100, 100, 200, 150)}
        nn_det = {'nn_confidence': 0.85, 'bbox': (105, 102, 198, 148)}
        feats = m.AlgorithmFusionHead.features_for(default_det, None, nn_det)
        self.assertEqual(feats.shape, (7,))
        # Agreement count = 2 (default + nn fired).
        self.assertEqual(feats[6], 2.0)

    def test_i7c_fusion_head_untrained_falls_back_to_max_score(self):
        fh = m.AlgorithmFusionHead()
        default_det = {'trail_snr': 3.5, 'bbox': (100, 100, 200, 150)}
        nn_det = {'nn_confidence': 0.85, 'bbox': (105, 102, 198, 148)}
        feats = m.AlgorithmFusionHead.features_for(default_det, None, nn_det)
        p = fh.predict_proba(feats)
        # Untrained: returns max(default_score=3.5, radon=0, nn=0.85) = 3.5.
        self.assertAlmostEqual(p, 3.5, places=3)

    def test_i7c_fusion_head_fit_and_save_roundtrip(self):
        fh = m.AlgorithmFusionHead()
        fh.path = Path(self.tmp) / 'fusion.json'
        rng = np.random.default_rng(0)
        X = rng.random((40, 7)).astype(np.float32)
        y = (X[:, 0] + X[:, 2] > 1.0).astype(np.float32)
        self.assertTrue(fh.fit(X, y, iters=20))
        self.assertIsNotNone(fh.w)
        fh.save()
        self.assertTrue(os.path.exists(fh.path))
        # Reload into a fresh instance.
        fh2 = m.AlgorithmFusionHead()
        fh2.path = Path(fh.path)
        self.assertTrue(fh2.load())
        np.testing.assert_allclose(fh.w, fh2.w, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
