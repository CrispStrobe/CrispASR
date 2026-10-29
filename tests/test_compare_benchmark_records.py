"""Contract checks for comparing retained measurements; no inference."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('benchmark_compare', Path(__file__).resolve().parents[1] / 'tools/compare_benchmark_records.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def record(seconds=(1, 2, 3), steps=10):
    return dict(model_revision='revision', precision='f16', input_sha256='a' * 64,
                settings={'beam': 1}, device='T4',
                samples=[dict(seconds=x, steps=steps) if steps is not None else dict(seconds=x) for x in seconds])


class CompareTest(unittest.TestCase):
    def test_medians_and_per_step(self):
        result = module.compare(record(), record((2, 4, 6)))
        self.assertEqual(result['left']['median_ms_per_step'], 200)
        self.assertEqual(result['left_over_right_speed_ratio'], 2)

    def test_each_identity_mismatch_rejected(self):
        for key in module.IDENTITY:
            other = record()
            other[key] = 'different'
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.compare(record(), other)

    def test_missing_or_malformed_identity_rejected(self):
        for key, value in [('input_sha256', ''), ('model_revision', ''), ('settings', None)]:
            a = record(); b = record(); a[key] = b[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.compare(a, b)

    def test_invalid_measurements_and_asymmetric_steps_rejected(self):
        for seconds in [(1, 2), (1, float('nan'), 3), (1, 0, 3)]:
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                module.compare(record(seconds), record())
        with self.assertRaises(ValueError):
            module.compare(record(), record(steps=None))
        self.assertIsNone(module.compare(record(steps=None), record(steps=None))['left']['median_ms_per_step'])


if __name__ == '__main__':
    unittest.main()
