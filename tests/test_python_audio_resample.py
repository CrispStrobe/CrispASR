"""Model-free audio front-door and real native C ABI regression tests.

The native subset compiles only the resampler, without a model or full runtime.
"""
import ctypes as C
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from crispasr import _binding as binding


def tone(rate, frequency):
    return (.5 * np.sin(2 * np.pi * frequency * np.arange(rate) / rate)).astype(np.float32)


def rms(signal):
    return float(np.sqrt(np.mean(signal[128:-128].astype(np.float64) ** 2)))


class AudioResampleTests(unittest.TestCase):
    def test_anti_alias_and_passband(self):
        for rate in (24000, 44100, 48000, 96000):
            with self.subTest(rate=rate):
                out = binding._prepare_asr_pcm(object(), tone(rate, 10000), rate)
                self.assertLess(rms(out), 1e-4)
                out = binding._prepare_asr_pcm(object(), tone(rate, 1000), rate)
                self.assertAlmostEqual(rms(out), .5 / np.sqrt(2), delta=1e-4)
                dc = binding._prepare_asr_pcm(object(), np.full(rate, .25, np.float32), rate)
                self.assertLess(float(np.max(np.abs(dc[128:-128] - .25))), 1e-4)

    def test_contiguous_unchanged_and_ceil_length(self):
        source = np.arange(101, dtype=np.float32)[::2]
        out = binding._prepare_asr_pcm(object(), source, 16000)
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(out, source)
        self.assertEqual(len(binding._prepare_asr_pcm(object(), source, 24000)), 34)
        self.assertEqual(len(binding._prepare_asr_pcm(object(), [], 48000)), 0)
        self.assertEqual(len(binding._prepare_asr_pcm(object(), [1], 24000)), 1)

    def test_invalid_inputs(self):
        for rate in (0, -1, True, 16000.0, 384001, 1):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                binding._prepare_asr_pcm(object(), [0, 1], rate)
        with self.assertRaises(ValueError):
            binding._prepare_asr_pcm(object(), np.zeros((2, 3)), 16000)

    def test_both_front_doors_receive_filtered_contiguous_samples(self):
        source = tone(48000, 10000)[::2]  # Strided actual 24 kHz signal.
        expected = binding._prepare_asr_pcm(object(), source, 24000)
        for kind in ('session', 'chunked', 'vad', 'logits', 'whisper'):
            received = []
            def capture(handle, pointer, count, *args):
                received.append(np.ctypeslib.as_array(pointer, shape=(count,)).copy())
                return 1
            lib = Mock(spec=[])
            if kind != 'whisper':
                lib.crispasr_session_transcribe = capture
                lib.crispasr_session_result_n_segments = lambda _: 0
                lib.crispasr_session_result_free = lambda _: None
                session = object.__new__(binding.Session)
                session._lib, session._handle, session.backend = lib, 1, 'test'
                if kind == 'session':
                    result = session.transcribe(source, sample_rate=24000)
                elif kind == 'chunked':
                    lib.crispasr_session_transcribe_chunked_lang = capture
                    result = session.transcribe_chunked(source, sample_rate=24000)
                elif kind == 'vad':
                    lib.crispasr_session_transcribe_vad = capture
                    result = session.transcribe_vad(source, 'vad.gguf', sample_rate=24000)
                else:
                    lib.crispasr_session_result_logits = lambda _: None
                    lib.crispasr_session_result_n_logit_frames = lambda _: 0
                    lib.crispasr_session_result_n_logit_vocab = lambda _: 0
                    result, logits = session.transcribe_with_logits(source, sample_rate=24000)
                    self.assertIsNone(logits)
                self.assertEqual(result, [])
                session._handle = None
            else:
                lib.whisper_full_default_params_by_ref = lambda _: 1
                lib.whisper_free_params = lambda _: None
                lib.whisper_full_n_segments = lambda _: 0
                def full(handle, params, pointer, count):
                    capture(handle, pointer, count)
                    return 0
                lib.whisper_full = full
                whisper = object.__new__(binding.CrispASR)
                whisper._lib, whisper._ctx, whisper._helpers = lib, None, None
                self.assertEqual(whisper.transcribe_pcm(source, sample_rate=24000), [])
            np.testing.assert_array_equal(received[0], expected)


    def test_static_wav_loader_is_bandlimited(self):
        import tempfile
        import wave
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tone.wav'
            with wave.open(str(path), 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes((tone(48000, 10000) * 32767).astype('<i2').tobytes())
            self.assertLess(rms(binding.CrispASR._load_audio(str(path))), 1e-4)

    def test_native_failure_does_not_silently_use_fallback(self):
        class Fn:
            def __call__(self, pcm, count, src, dst, out, length):
                return -2
        lib = Mock(spec=[])
        lib.crispasr_audio_resample = Fn()
        lib.crispasr_audio_free = Mock()
        with self.assertRaisesRegex(RuntimeError, 'code -2'):
            binding._prepare_asr_pcm(lib, [0, 1, 2], 24000)
        lib.crispasr_audio_free.assert_not_called()


class NativeAudioResampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = None
        shipped = os.environ.get('CRISPASR_TEST_LIB')
        if shipped:
            cls.lib = C.CDLL(shipped)
            cls.pointer = C.POINTER(C.c_float)
            cls.lib.crispasr_audio_resample.argtypes = [cls.pointer, C.c_int, C.c_int,
                C.c_int, C.POINTER(cls.pointer), C.POINTER(C.c_int)]
            cls.lib.crispasr_audio_resample.restype = C.c_int
            cls.lib.crispasr_audio_free.argtypes = [cls.pointer]
            cls.lib.crispasr_audio_free.restype = None
            cls.allocator = type('Allocator', (), {'free': cls.lib.crispasr_audio_free})()
            return
        compiler = shutil.which('c++')
        if not compiler:
            raise RuntimeError('C++ compiler required for native resampling ABI tests')
        # Keep local scratch on the configured storage volume; CI defaults to /tmp.
        import tempfile
        cls.temp = tempfile.TemporaryDirectory(prefix='crispasr-resample-')
        cls.path = Path(cls.temp.name) / 'resample.so'
        subprocess.run([compiler, '-std=c++17', '-O2', '-shared', '-fPIC',
            '-DCRISPASR_SHARED', '-DCRISPASR_BUILD',
            '-I' + str(ROOT / 'include'), '-I' + str(ROOT / 'ggml/include'),
            '-I' + str(ROOT / 'src'), str(ROOT / 'src/core/audio_resample.cpp'),
            str(ROOT / 'src/crispasr_audio_resample.cpp'), '-o', str(cls.path)], check=True)
        cls.lib = C.CDLL(str(cls.path))
        cls.pointer = C.POINTER(C.c_float)
        cls.lib.crispasr_audio_resample.argtypes = [cls.pointer, C.c_int, C.c_int,
            C.c_int, C.POINTER(cls.pointer), C.POINTER(C.c_int)]
        cls.lib.crispasr_audio_resample.restype = C.c_int
        # Production crispasr_audio_free calls std::free; this minimal library
        # omits the file codecs and uses that same allocator directly.
        cls.allocator = C.CDLL(None)
        cls.allocator.free.argtypes = [cls.pointer]
        cls.allocator.free.restype = None
        cls.lib.crispasr_audio_free = cls.allocator.free

    @classmethod
    def tearDownClass(cls):
        if cls.temp is not None:
            cls.temp.cleanup()

    def test_native_fallback_agreement(self):
        rng = np.random.default_rng(123)
        for rate in (8000, 16000, 22050, 24000, 44100, 48000, 96000):
            source = rng.normal(0, .1, 1001).astype(np.float32)
            native = binding._prepare_asr_pcm(self.lib, source, rate)
            fallback = binding._prepare_asr_pcm(object(), source, rate)
            self.assertEqual(len(native), (1001 * 16000 + rate - 1) // rate)
            np.testing.assert_allclose(native, fallback, rtol=0, atol=1e-7)
        for rate in (24000, 48000):
            self.assertLess(rms(binding._prepare_asr_pcm(self.lib, tone(rate, 10000), rate)), 1e-4)

    def test_error_reset_empty_and_identity_ownership(self):
        source = np.arange(19, dtype=np.float32)
        pointer = source.ctypes.data_as(self.pointer)
        for rate, count in ((0, 19), (384001, 19), (1, 19), (16000, -1)):
            output, length = C.cast(source.ctypes.data, self.pointer), C.c_int(17)
            self.assertEqual(self.lib.crispasr_audio_resample(pointer, count, rate, 16000,
                C.byref(output), C.byref(length)), -1)
            self.assertFalse(output)
            self.assertEqual(length.value, 0)
        output, length = self.pointer(), C.c_int(17)
        self.assertEqual(self.lib.crispasr_audio_resample(None, 0, 24000, 16000,
            C.byref(output), C.byref(length)), 0)
        self.assertFalse(output)
        self.assertEqual(length.value, 0)
        self.assertEqual(self.lib.crispasr_audio_resample(pointer, 19, 16000, 16000,
            C.byref(output), C.byref(length)), 0)
        try:
            self.assertNotEqual(C.addressof(output.contents), source.ctypes.data)
            np.testing.assert_array_equal(np.ctypeslib.as_array(output, shape=(19,)), source)
        finally:
            self.allocator.free(output)


if __name__ == '__main__':
    unittest.main()
