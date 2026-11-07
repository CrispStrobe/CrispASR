"""Model-free conversion controls for the pinned RE-USE research artifact."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import gguf
    import numpy as np
    import safetensors  # noqa: F401
except ModuleNotFoundError:
    sys.exit(77)

spec = importlib.util.spec_from_file_location('reuse_converter', Path(__file__).resolve().parents[1] /
                                              'models/convert-reuse-to-gguf.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTest(unittest.TestCase):
    def setUp(self):
        self.config = {'model_cfg': dict(converter.EXPECTED_MODEL), 'stft_cfg': dict(converter.EXPECTED_STFT)}

    def test_round_trip_preserves_f32_tensor_values(self):
        tensors = {'a': np.arange(12, dtype=np.float32).reshape(3, 4), 'b': np.array([-1.0, 0.25], np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.gguf'
            converter.write_gguf(tensors, self.config, path)
            actual = {t.name: t.data for t in gguf.GGUFReader(str(path)).tensors}
            self.assertEqual(set(actual), set(tensors))
            for name, expected in tensors.items():
                np.testing.assert_array_equal(actual[name].reshape(-1), expected.reshape(-1))

    def test_rejects_incompatible_model_and_frontend(self):
        for section, key in (('model_cfg', 'd_state'), ('stft_cfg', 'n_fft')):
            original = self.config[section][key]
            self.config[section][key] = 1
            with self.assertRaises(ValueError):
                converter.validate_config(self.config)
            self.config[section][key] = original

    def test_rejects_nonfinite_wrong_dtype_and_bad_rank(self):
        for value in (np.array([np.nan], np.float32), np.ones(2, np.float16), np.ones((), np.float32),
                      np.ones((1,1,1,1,1), np.float32)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                converter.validate_tensor('bad', value)


if __name__ == '__main__':
    unittest.main()
