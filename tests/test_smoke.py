"""Smoke tests: module imports, version, constants, top-level functions."""
import unittest

import satellite_trail_detector as m


class TestModuleSmoke(unittest.TestCase):

    def test_version(self):
        self.assertIsInstance(m.__version__, str)
        self.assertTrue(len(m.__version__) > 0)

    def test_core_classes_exist(self):
        for name in (
            'SatelliteTrailDetector', 'RadonStreakDetector', 'NeuralNetDetector',
            '_NNBackend', 'TemporalFrameBuffer', 'DetectionTracker',
            'TranslationLedger', 'AnnotationDatabase', 'ParameterAdapter',
            'ReviewUI', 'DatasetExporter', 'ProcessingWindow',
        ):
            self.assertTrue(hasattr(m, name), f"missing: {name}")

    def test_initiative_classes_exist(self):
        """All 12 v0.3.0 initiative classes are importable."""
        for name in (
            'RescueClassifier',          # I1
            'LongBackgroundModel',       # I5
            'ThresholdHyperNet',         # I3
            'IMMKalmanTracker',          # I4
            'PlattCalibrator',           # I6c
            'TrustRegionAdapter',        # I6b
            'BALDQueue',                 # I6d
            'TrackletPseudoLabeler',     # I7a
            'TrackletSequenceHead',      # I7b
            'AlgorithmFusionHead',       # I7c
        ):
            self.assertTrue(hasattr(m, name), f"missing: {name}")

    def test_constants(self):
        self.assertIsInstance(m.LOSS_PROFILES, dict)
        self.assertGreaterEqual(len(m.LOSS_PROFILES), 4)
        for profile in ('discovery', 'precision', 'balanced', 'catalog'):
            self.assertIn(profile, m.LOSS_PROFILES)
        self.assertIsInstance(m.PARAMETER_SAFETY_BOUNDS, dict)
        self.assertIsInstance(m.CORRECTION_RULES, dict)

    def test_hardware_detection(self):
        profile = m._detect_hardware()
        self.assertIn('cpu_count', profile)
        self.assertGreaterEqual(profile['cpu_count'], 1)


if __name__ == '__main__':
    unittest.main()
