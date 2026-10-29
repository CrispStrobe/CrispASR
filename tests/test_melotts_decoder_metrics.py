"""The decoder gate must reject scale defects invisible to cosine."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/validation'))
spec = importlib.util.spec_from_file_location('decoder_validation', ROOT/'tools/validation/melotts_decoder.py')
decoder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(decoder)


class DecoderMetrics(unittest.TestCase):
    def test_identity_and_small_roundoff(self):
        self.assertTrue(decoder.comparison([-1, 2, -3, 4], [-1, 2, -3, 4])['accepted'])
        self.assertTrue(decoder.comparison([-1.00001, 2, -3, 4], [-1, 2, -3, 4])['accepted'])

    def test_cosine_one_does_not_hide_scale_error(self):
        result = decoder.comparison([-2, 4, -6, 8], [-1, 2, -3, 4])
        self.assertAlmostEqual(result['cosine'], 1)
        self.assertEqual(result['norm_ratio'], 2)
        self.assertFalse(result['accepted'])

    def test_shape_nonfinite_and_zero_cannot_qualify(self):
        for actual, reference in [([1], [1, 2]), ([float('nan')], [1]),
                                  ([float('inf')], [1]), ([0], [0]), ([], [])]:
            with self.subTest(actual=actual, reference=reference):
                self.assertFalse(decoder.comparison(actual, reference)['accepted'])

    def test_equal_magnitude_different_waveform_rejected(self):
        self.assertFalse(decoder.comparison([1, -2, 3, -4], [-1, 2, -3, 4])['accepted'])


if __name__ == '__main__':
    unittest.main()
