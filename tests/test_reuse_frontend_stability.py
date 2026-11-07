"""Model-free failure controls for the original-STFT diagnostic."""
import importlib.util
from pathlib import Path
import unittest
try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/frontend_stability.py'
spec = importlib.util.spec_from_file_location('reuse_frontend_stability', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FrontendStability(unittest.TestCase):
    def test_raw_branch_errors_are_counted_without_wrapping(self):
        phase = np.array([[np.pi, 0.5], [np.pi, 1.0]], np.float32)
        changed = phase.copy()
        changed[:, 0] *= -1
        self.assertEqual(module.branches(changed, phase), {'all': 2, 'first_frame': 1})
        self.assertGreater(module.metrics(changed, phase)['relative_rmse'], 1)

    def test_scale_and_nonfinite_arrays_are_visible(self):
        self.assertEqual(module.metrics([2., 4.], [1., 2.])['relative_rmse'], 1)
        for value in ([1.], [np.nan, 1.], []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                module.metrics(value, [1., 2.])

    def test_capture_rejects_changed_audio_invalid_shape_and_nonfinite(self):
        audio = np.ones(400, np.float32)
        capture = {'audio': audio.copy(), 'window': np.ones(640, np.float32),
                   'first_complex': np.ones((321, 2), np.float32),
                   **{name: np.ones((6, 321), np.float32) for name in ('phase', 'real', 'imaginary')}}
        module.validate(capture, audio)
        for name, invalid in (('audio', -audio), ('window', np.ones(639, np.float32)),
                              ('phase', np.full((6, 321), np.nan, np.float32))):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.validate({**capture, name: invalid}, audio)


if __name__ == '__main__':
    unittest.main()
