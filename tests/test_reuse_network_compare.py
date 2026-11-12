#!/usr/bin/env python3
"""Controls for the raw RE-USE stage and waveform acceptance gate."""
import importlib.util
from pathlib import Path
import unittest

try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/network_compare.py'
spec = importlib.util.spec_from_file_location('reuse_network_compare', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class CompareControls(unittest.TestCase):
    def test_equal_raw_signal_passes(self):
        x = np.array([0.2, -0.4, 0.7], dtype=np.float32)
        result = module.compare('equal', x, x.copy(), 0.002)
        self.assertEqual(result['relative_rmse'], 0)
        self.assertAlmostEqual(result['norm_ratio'], 1)

    def test_scale_blind_cosine_is_not_accepted(self):
        x = np.array([0.2, -0.4, 0.7], dtype=np.float32)
        with self.assertRaises(AssertionError):
            module.compare('scaled', x, 2 * x, 0.002)

    def test_phase_branch_cut_is_not_treated_as_equal_model_input(self):
        # These angles describe the same complex point, but the model reads
        # raw phase values, so +pi and -pi must not pass the input gate.
        reference = np.array([np.pi, 0.5], dtype=np.float32)
        native = np.array([-np.pi, 0.5], dtype=np.float32)
        with self.assertRaises(AssertionError):
            module.compare('raw_phase', reference, native, 0.002)

    def test_nonfinite_shape_and_silence_rejected(self):
        x = np.array([0.2, -0.4, 0.7], dtype=np.float32)
        for invalid in (np.array([np.nan, 0, 0]), np.zeros(3), np.zeros(2)):
            with self.subTest(invalid=invalid), self.assertRaises(AssertionError):
                module.compare('invalid', x, invalid, 0.002)

if __name__ == '__main__':
    unittest.main()
