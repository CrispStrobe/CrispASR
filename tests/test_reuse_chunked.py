"""Model-free transition, padding and input controls for RE-USE research chunks."""
import importlib.util
import math
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

    def test_hop_aligned_long_plan_matches_pinned_upstream_end_anchoring(self):
        # NVIDIA's inference_chunk.py calculates ceil((n - chunk) / hop) + 1
        # and anchors the final window at n - chunk. Compare the diagnostic
        # plan at its smaller 1 s window size; this is not model-output parity.
        for n in (16000, 16080, 23920, 24000, 24080, 32000, 40160, 80000):
            with self.subTest(n=n):
                count = max(1, math.ceil((n - 16000) / 8000) + 1)
                expected = [i * 8000 for i in range(count - 1)] + [n - 16000]
                _, starts = module.chunked(np.ones(n, np.float32), lambda b, i, s: b)
                self.assertEqual(starts, expected)

    def test_boundary_impulses_and_padding_are_not_dropped(self):
        x = np.zeros(32127, np.float32)
        x[[0, 7999, 8000, 15999, 16000, len(x) - 1]] = 1
        out, starts = module.chunked(x, lambda block, i, start: block)
        np.testing.assert_array_equal(out, x)
        self.assertEqual(starts, [0, 8000, 16000, 16160])

    def test_explicit_five_second_identity_and_end_anchoring(self):
        for n in (1, 79999, 80000, 160000, 160001, 200037, 320000):
            with self.subTest(samples=n):
                x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
                calls = []
                def identity(block, index, start):
                    calls.append((index, start))
                    self.assertEqual(block.shape, (80000,))
                    valid = min(80000, n - start)
                    np.testing.assert_array_equal(block[:valid], x[start:start + valid])
                    np.testing.assert_array_equal(block[valid:], 0)
                    return block
                out, starts = module.chunked(x, identity, window_samples=80000,
                                             hop_samples=40000, max_window_samples=80000)
                padded_n = max(80000, ((n + 79) // 80) * 80)
                count = math.ceil((padded_n - 80000) / 40000) + 1
                expected = [i * 40000 for i in range(count - 1)] + [padded_n - 80000]
                self.assertEqual(starts, expected)
                self.assertEqual(calls, list(enumerate(expected)))
                self.assertEqual(len(starts), len(set(starts)))
                self.assertTrue(all(s % 80 == 0 for s in starts))
                self.assertEqual(out.shape, x.shape)
                np.testing.assert_allclose(out, x, atol=2e-7, rtol=1e-6)

    def test_five_second_overlap_blends_distinct_window_outputs(self):
        out, starts = module.chunked(np.zeros(160000, np.float32),
                                     lambda block, index, start: np.full(80000, index, np.float32),
                                     window_samples=80000, hop_samples=40000, max_window_samples=80000)
        self.assertEqual(starts, [0, 40000, 80000])
        self.assertEqual(out[0], 0)
        self.assertAlmostEqual(float(out[60000]), 0.5, places=6)
        self.assertAlmostEqual(float(out[100000]), 1.5, places=6)
        self.assertEqual(out[-1], 2)

    def test_aligned_overlap_matches_upstream_float32_weighting(self):
        # Pinned inference_chunk.py uses periodic Hann, clamping and float32
        # accumulation. Distinct waveforms exercise the complete overlaps.
        rng = np.random.default_rng(579)
        windows = [rng.standard_normal(16000).astype(np.float32) for _ in range(3)]
        actual, starts = module.chunked(np.zeros(32000, np.float32),
                                        lambda block, index, start: windows[index])
        self.assertEqual(starts, [0, 8000, 16000])
        phase = np.arange(16000, dtype=np.float32) * np.float32(2 * np.pi / 16000)
        weights = np.maximum(np.float32(0.5) * (np.float32(1) - np.cos(phase)), np.float32(1e-7))
        total = np.zeros(32000, np.float32)
        norm = np.zeros(32000, np.float32)
        for start, waveform in zip(starts, windows):
            total[start:start + 16000] += waveform * weights
            norm[start:start + 16000] += weights
        np.testing.assert_allclose(actual, total / norm, atol=1e-6, rtol=1e-6)

    def test_opt_in_limit_preserves_default_policy_and_large_window_needs_opt_in(self):
        x = np.random.default_rng(579).standard_normal(200037).astype(np.float32)
        def enhance(block, index, start):
            self.assertEqual(len(block), 16000)
            return block * (index + 1)
        default, starts = module.chunked(x, enhance)
        explicit, explicit_starts = module.chunked(x, enhance, max_window_samples=80000)
        self.assertEqual(starts, explicit_starts)
        self.assertEqual(starts[-1], 184080)
        np.testing.assert_array_equal(default, explicit)
        with self.assertRaises(ValueError):
            module.chunked(x, enhance, window_samples=80000, hop_samples=40000)

    def test_invalid_explicit_limits_fail_before_enhancement(self):
        calls = []
        for limit in (None, True, False, '80000', 80000.0, 0, -80, 320, 401, 16001, 80080):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                module.chunked(np.ones(400, np.float32), lambda *args: calls.append(args),
                               max_window_samples=limit)
        for window, hop, limit in ((80080, 40000, 80000), (80000, 40000, 40000),
                                   (80000, 40001, 80000), (80000, 80080, 80000)):
            with self.subTest(window=window, hop=hop, limit=limit), self.assertRaises(ValueError):
                module.chunked(np.ones(400, np.float32), lambda *args: calls.append(args),
                               window_samples=window, hop_samples=hop, max_window_samples=limit)
        self.assertEqual(calls, [])

    def test_invalid_input_plan_and_enhancer_fail_closed(self):
        for x in (np.array([]), np.array([[0.]]), np.array([np.nan])):
            with self.assertRaises(ValueError): module.chunked(x, lambda b, i, s: b)
        for kw in ({'hop_samples': 0}, {'hop_samples': 16080}, {'window_samples': 16080}, {'hop_samples': 81}):
            with self.assertRaises(ValueError): module.chunked(np.ones(400), lambda b, i, s: b, **kw)
        for result in (np.ones(1), np.full(16000, np.nan)):
            with self.assertRaises(ValueError): module.chunked(np.ones(400), lambda b, i, s: result)


if __name__ == '__main__':
    unittest.main()
