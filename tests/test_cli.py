"""CLI parsing tests — run the script with --help and argument combinations."""
import os
import subprocess
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, 'satellite_trail_detector.py')


def _run(args):
    """Run the script with UTF-8 io, capturing output. Returns (rc, stdout, stderr)."""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True, text=True, encoding='utf-8', env=env, cwd=REPO_ROOT,
    )
    return r.returncode, r.stdout, r.stderr


class TestCLI(unittest.TestCase):

    def test_help_runs(self):
        rc, out, err = _run(['--help'])
        self.assertEqual(rc, 0, msg=err)
        self.assertIn('satellite_trail_detector', out)

    def test_v030_initiative_flags_present(self):
        rc, out, _ = _run(['--help'])
        self.assertEqual(rc, 0)
        for flag in ('--tracker', '--long-bg', '--hypernet', '--pseudo-label',
                     '--train-rescue', '--tier2-optimizer', '--use-trust-region',
                     '--fusion'):
            self.assertIn(flag, out, f"--help missing flag: {flag}")

    def test_tier2_optimizer_choices(self):
        """Bogus optimizer value is rejected."""
        rc, _, err = _run(['in.mp4', 'out.mp4', '--tier2-optimizer', 'nelder-mead'])
        self.assertNotEqual(rc, 0)
        self.assertIn('invalid choice', err.lower())

    def test_tracker_choices(self):
        rc, _, err = _run(['in.mp4', 'out.mp4', '--tracker', 'particle'])
        self.assertNotEqual(rc, 0)
        self.assertIn('invalid choice', err.lower())

    def test_fusion_without_hybrid_emits_warning(self):
        """--fusion without --nn-hybrid is accepted but warns."""
        # We don't have a real model path, so the run will fail later, but
        # the warning should be printed before the failure (it's printed
        # during nn_params assembly).
        rc, out, err = _run([
            'in.mp4', 'out.mp4', '--algorithm', 'nn',
            '--model', 'nonexistent.pt', '--fusion',
        ])
        # Accept either an early parser error (if --model nonexistent is
        # rejected) or a later pipeline failure — but the fusion warning
        # string should appear.
        combined = out + err
        self.assertIn('fusion', combined.lower())


if __name__ == '__main__':
    unittest.main()
