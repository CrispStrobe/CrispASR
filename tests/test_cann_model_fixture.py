"""Non-vacuous scale/input guards for the CPU fixture producer."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

path = Path(__file__).resolve().parents[1]/'tools/ci-heavy/cann_model_fixture.py'
spec = importlib.util.spec_from_file_location('fixture', path)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class FixtureGuardTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.array([[1., 2., 3.], [3., 2., 1.]], dtype=np.float32)

    def test_exact_encoder(self):
        metrics = fixture.compare(self.reference, self.reference.copy())
        self.assertEqual(metrics['relative_l2'], 0)
        self.assertEqual(metrics['max_row_norm_error'], 0)

    def test_scale_blind_cosine_is_rejected(self):
        with self.assertRaises(ValueError):
            fixture.compare(self.reference, self.reference*2)

    def test_bad_shape_and_nonfinite_are_rejected(self):
        for actual in (self.reference[:1], self.reference*np.nan, np.zeros_like(self.reference)):
            with self.subTest(shape=actual.shape), self.assertRaises(ValueError):
                fixture.compare(self.reference, actual)

    def test_wrong_direction_is_rejected(self):
        with self.assertRaises(ValueError):
            fixture.compare(self.reference, -self.reference)


if __name__ == '__main__':
    unittest.main()
