"""Check sidecar conversion rejects malformed weights and preserves F32 bytes."""
import importlib.util
import gc
from pathlib import Path
import tempfile
import subprocess
import unittest
import sys

try:
    import gguf
    import numpy as np
    import safetensors  # noqa: F401
except ModuleNotFoundError:
    sys.exit(77)

spec = importlib.util.spec_from_file_location('ultra_converter', Path(__file__).resolve().parents[1] /
                                              'models/convert-qwen3-ultra-head-to-gguf.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.weights = {name: np.full(shape, 0.25, dtype=np.float32) for name, shape in converter.SHAPES.items()}
        self.norm = {'mean': [0.0] * 91, 'std': [1.0] * 91}

    def test_round_trip_preserves_every_tensor(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'head.gguf'
            converter.write_gguf(self.weights, self.norm, path)
            actual = {t.name: t.data for t in gguf.GGUFReader(str(path)).tensors}
            self.assertEqual(set(actual), set(self.weights) | {'normalizer.mean', 'normalizer.std'})
            for name, expected in self.weights.items():
                np.testing.assert_array_equal(actual[name].reshape(-1), expected.reshape(-1))
            np.testing.assert_array_equal(actual['normalizer.std'], np.ones(91, dtype=np.float32))
            del actual
            gc.collect()

    def test_rejects_missing_wrong_shape_and_nonfinite(self):
        name = 'audio.0.bias'
        for value in (None, np.zeros((2,), dtype=np.float32), np.full((128,), np.nan, dtype=np.float32)):
            with self.subTest(value_shape=None if value is None else value.shape):
                original = self.weights.pop(name)
                if value is not None:
                    self.weights[name] = value
                with self.assertRaises(ValueError):
                    converter.validate(self.weights, self.norm)
                self.weights[name] = original

    def test_loader_negative_fixtures_are_constructed_as_intended(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run([sys.executable, str(root / 'tools/ci-heavy/issue579_ultra_sidecar_fixtures.py'), temp],
                           cwd=root, check=True)
            directory = Path(temp)
            for kind in ('valid', 'invalid', 'shape', 'nan', 'std'):
                self.assertTrue((directory / f'synthetic-{kind}.gguf').is_file())
            tensors = {t.name: t.data for t in gguf.GGUFReader(str(directory / 'synthetic-shape.gguf')).tensors}
            self.assertEqual(tensors['audio.0.bias'].size, 127)
            tensors = {t.name: t.data for t in gguf.GGUFReader(str(directory / 'synthetic-nan.gguf')).tensors}
            self.assertTrue(np.isnan(tensors['audio.0.bias']).all())
            tensors = {t.name: t.data for t in gguf.GGUFReader(str(directory / 'synthetic-std.gguf')).tensors}
            self.assertTrue((tensors['normalizer.std'] == 0).all())
            del tensors
            gc.collect()

    def test_rejects_zero_or_nonfinite_normalizer(self):
        for value in (0.0, float('nan')):
            self.norm['std'][3] = value
            with self.assertRaises(ValueError):
                converter.validate(self.weights, self.norm)


if __name__ == '__main__':
    unittest.main()
