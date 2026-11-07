"""Controls for the diagnostic RE-USE phase convention and compatibility bound."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/phase_stability.py'
spec = importlib.util.spec_from_file_location('reuse_phase_stability', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PhaseStability(unittest.TestCase):
    def test_canonical_candidate_only_changes_first_frame_near_cut(self):
        p = np.array([[-np.pi, np.pi, -1.0], [-np.pi, np.pi, 0.0]], dtype=np.float32)
        original = p.copy()
        result = mod.canonical_first_frame(p)
        self.assertEqual(result[0, 0], np.float32(np.pi))
        self.assertEqual(result[0, 2], -1.0)
        np.testing.assert_array_equal(result[1], original[1])
        np.testing.assert_array_equal(p, original)

    def test_invalid_phase_rejected(self):
        for p in (np.array([np.pi]), np.array([[np.nan]]), np.empty((0, 3))):
            with self.subTest(shape=p.shape), self.assertRaises(ValueError):
                mod.canonical_first_frame(p)
        with self.assertRaises(ValueError):
            mod.canonical_first_frame(np.ones((2, 3)), 0)

    def test_two_reference_compatibility_bound(self):
        self.assertTrue(mod.mutually_compatible(mod.compare([1., 2.], [1., 2.]), 0.005))
        self.assertFalse(mod.mutually_compatible(mod.compare([1., 0.], [1., 0.02]), 0.005))
        for bad in ([0., 0.], [np.nan, 1.], [1.]):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                mod.compare([1., 1.], bad)

if __name__ == '__main__':
    unittest.main()
