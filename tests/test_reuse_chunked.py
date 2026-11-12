"""Model-free transition, padding and input controls for RE-USE research chunks."""
import importlib.util
from pathlib import Path
import unittest
try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

path = Path(__file__).resolve().parents[1] / 'tools/ci-heavy/issue579_reuse/chunked.py'
spec = importlib.util.spec_from_file_location('reuse_chunked', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChunkedTests(unittest.TestCase):
    def test_identity_preserves_short_exact_and_irregular_long_audio(self):
        for n in (1, 15999, 16000, 16001, 24000, 48037, 160000):
            with self.subTest(n=n):
                x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
                out, starts = module.chunked(x, lambda block, i, start: block)
                np.testing.assert_allclose(out, x, atol=2e-7, rtol=1e-6)
                self.assertEqual(len(starts), len(set(starts)))
                self.assertEqual(starts[0], 0)
                self.assertTrue(all(s % 80 == 0 for s in starts))

    def test_boundary_impulses_and_padding_are_not_dropped(self):
        x = np.zeros(32127, np.float32)
        x[[0, 7999, 8000, 15999, 16000, len(x) - 1]] = 1
        out, starts = module.chunked(x, lambda block, i, start: block)
        np.testing.assert_array_equal(out, x)
        self.assertEqual(starts, [0, 8000, 16000, 16160])

    def test_invalid_input_plan_and_enhancer_fail_closed(self):
        for x in (np.array([]), np.array([[0.]]), np.array([np.nan])):
            with self.assertRaises(ValueError): module.chunked(x, lambda b, i, s: b)
        for kw in ({'hop_samples': 0}, {'hop_samples': 16080}, {'window_samples': 16080}, {'hop_samples': 81}):
            with self.assertRaises(ValueError): module.chunked(np.ones(400), lambda b, i, s: b, **kw)
        for result in (np.ones(1), np.full(16000, np.nan)):
            with self.assertRaises(ValueError): module.chunked(np.ones(400), lambda b, i, s: result)


if __name__ == '__main__':
    unittest.main()
