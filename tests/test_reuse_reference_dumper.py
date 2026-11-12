"""Model-free controls for the RE-USE crispasr-diff reference contract."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

path = Path(__file__).resolve().parents[1] / 'tools/reference_backends/reuse.py'
spec = importlib.util.spec_from_file_location('reuse_reference_dumper', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReferenceContract(unittest.TestCase):
    def test_frontend_observer_preserves_original_call_result_and_signed_zero(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)
            def __getitem__(self, item): return Tensor(self.value[item])
            @property
            def real(self): return Tensor(self.value.real)
            @property
            def imag(self): return Tensor(self.value.imag)
            @property
            def T(self): return Tensor(self.value.T)
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def contiguous(self): return self
            def numpy(self): return self.value

        z = np.empty((1, 2, 3), dtype=np.complex64)
        z.real[:] = -1
        z.imag[:] = -0.0
        window = Tensor(np.array([0., 0.5, 1.], dtype=np.float32))
        original_result = Tensor(z)
        original = Mock(return_value=original_result)
        torch = SimpleNamespace(stft=original, from_numpy=lambda x: x)
        expected = object()
        def upstream(*args, **kwargs):
            self.assertIs(torch.stft(args[0], window=window), original_result)
            return expected
        captures = {}
        actual = module.capture_frontend(torch, upstream, np.zeros(400, np.float32), captures)
        self.assertIs(actual, expected)
        self.assertIs(torch.stft, original)
        original.assert_called_once()
        np.testing.assert_array_equal(captures['frontend_window'], window.value)
        np.testing.assert_array_equal(captures['frontend_real'], z[0].real.T)
        self.assertTrue(np.signbit(captures['frontend_imaginary']).all())
        z[:] = 2 + 3j
        self.assertTrue(np.signbit(captures['frontend_imaginary']).all())

    def test_frontend_observer_restores_stft_after_failure(self):
        original = Mock(side_effect=RuntimeError('original STFT failure'))
        torch = SimpleNamespace(stft=original, from_numpy=lambda x: x)
        def upstream(*args, **kwargs):
            return torch.stft(args[0])
        with self.assertRaisesRegex(RuntimeError, 'original STFT failure'):
            module.capture_frontend(torch, upstream, np.zeros(400, np.float32), {})
        self.assertIs(torch.stft, original)
        original.assert_called_once()

    def test_all_model_boundaries_are_registered_once(self):
        stages = module.DEFAULT_STAGES
        self.assertEqual(len(stages), len(set(stages)))
        for index in range(6):
            self.assertIn(f'encoder_layer_{index}', stages)
        for index in range(30):
            self.assertIn(f'block_time_{index}', stages)
            self.assertIn(f'block_{index}', stages)
        for name in ('raw_audio', 'frontend_magnitude', 'frontend_phase', 'frontend_window', 'frontend_real',
                     'frontend_imaginary', 'amplitude', 'output_phase', 'waveform'):
            self.assertIn(name, stages)

    def test_bounded_audio_is_exact_and_oversized_input_is_rejected(self):
        audio = np.arange(24000, dtype=np.float32) / 24000
        with self.assertRaisesRegex(ValueError, "16000"):
            module.prepare_audio(audio)
        np.testing.assert_array_equal(module.prepare_audio(audio[:16000]), audio[:16000])
        np.testing.assert_array_equal(module.prepare_audio(audio[:1600]), audio[:1600])

    def test_explicit_five_second_bound_preserves_every_sample(self):
        audio = np.arange(80000, dtype=np.float32) / 80000
        np.testing.assert_array_equal(module.prepare_audio(audio, max_samples=80000), audio)
        with self.assertRaises(ValueError):
            module.prepare_audio(np.zeros(80080, np.float32), max_samples=80000)
        for bad in (0, 16001, 80080, -80):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                module.prepare_audio(audio[:400], max_samples=bad)

    def test_invalid_audio_and_unknown_stage_fail_before_loading_model(self):
        for audio in (np.zeros(320, np.float32), np.zeros(401, np.float32), np.full(400, np.nan, np.float32), np.zeros((400, 2), np.float32)):
            with self.subTest(length=len(audio)), self.assertRaises(ValueError):
                module.dump('/nonexistent', audio, {'waveform'})
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            module.dump('/nonexistent', np.zeros(400, np.float32), {'invented'})

if __name__ == '__main__':
    unittest.main()
