#!/usr/bin/env python3
"""Integration tests for the CrispASR Python Session API.

Requires:
  - Built libwhisper.so/dylib (cmake --build build)
  - whisper-tiny model: models/ggml-tiny.en.bin
  - parakeet model: set PARAKEET_MODEL env var or skip
  - samples/jfk.wav (11s JFK speech)

Run:
  python tests/test_python_session.py
  # or with pytest:
  pytest tests/test_python_session.py -v
"""

import os
import sys
import unittest
import wave

import numpy as np

# Add the python dir to path so we can import crispasr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
JFK_WAV = os.path.join(REPO_ROOT, "samples", "jfk.wav")
CRISPASR_TINY = os.path.join(REPO_ROOT, "models", "ggml-tiny.en.bin")
PARAKEET_MODEL = os.environ.get(
    "PARAKEET_MODEL",
    os.path.join(os.path.dirname(__file__), "..", "..", "test_cohere", "parakeet-tdt-0.6b-v3.gguf"),
)

# Find the built shared library
LIB_PATH = os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for candidate in [
        "/tmp/build-shared/src/libwhisper.so",
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.so"),
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "Release", "whisper.dll"),
        os.path.join(REPO_ROOT, "build-shared", "src", "libwhisper.so"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break


def load_jfk_pcm():
    """Load jfk.wav as 16kHz mono float32 numpy array."""
    with wave.open(JFK_WAV, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return pcm


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestWhisperSession(unittest.TestCase):
    """Test the Session API with whisper-tiny.en."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CRISPASR_TINY):
            raise unittest.SkipTest(f"Model not found: {CRISPASR_TINY}")
        from crispasr import Session
        # Whisper GGML files are auto-detected via magic bytes fallback
        cls.session = Session(CRISPASR_TINY, lib_path=LIB_PATH, n_threads=2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def test_backend_name(self):
        self.assertEqual(self.session.backend, "whisper")

    def test_transcribe_jfk(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        full_text = " ".join(s.text for s in segs).lower()
        self.assertIn("fellow americans", full_text)
        self.assertIn("country", full_text)

    def test_segment_timestamps(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        for seg in segs:
            self.assertGreaterEqual(seg.start, 0.0)
            self.assertGreater(seg.end, seg.start)
            self.assertLess(seg.end, 15.0)  # audio is ~11s

    def test_empty_audio(self):
        """Empty or near-silent audio should not crash."""
        silence = np.zeros(16000, dtype=np.float32)  # 1s silence
        segs = self.session.transcribe(silence)
        # May produce empty or very short result — just shouldn't crash
        self.assertIsInstance(segs, list)

    def test_very_short_audio(self):
        """Very short audio (0.1s) should not crash."""
        short = np.zeros(1600, dtype=np.float32)
        segs = self.session.transcribe(short)
        self.assertIsInstance(segs, list)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
@unittest.skipUnless(os.path.exists(PARAKEET_MODEL), f"Parakeet model not found at {PARAKEET_MODEL}")
class TestParakeetSession(unittest.TestCase):
    """Test the Session API with parakeet."""

    @classmethod
    def setUpClass(cls):
        from crispasr import Session
        cls.session = Session(PARAKEET_MODEL, lib_path=LIB_PATH, n_threads=2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def test_backend_name(self):
        self.assertEqual(self.session.backend, "parakeet")

    def test_transcribe_jfk(self):
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        full_text = " ".join(s.text for s in segs).lower()
        self.assertIn("fellow americans", full_text)
        self.assertIn("country", full_text)

    def test_word_timestamps(self):
        """Parakeet should produce word-level timestamps."""
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        self.assertGreater(len(segs), 0)
        # Parakeet produces words natively
        words = segs[0].words
        self.assertGreater(len(words), 0)
        for w in words:
            self.assertGreaterEqual(w.start, 0.0)
            self.assertGreaterEqual(w.end, w.start)
            self.assertTrue(len(w.text) > 0)

    def test_timestamps_monotonic(self):
        """Word timestamps should be non-decreasing."""
        pcm = load_jfk_pcm()
        segs = self.session.transcribe(pcm)
        for seg in segs:
            prev_end = 0.0
            for w in seg.words:
                self.assertGreaterEqual(w.start, prev_end - 0.01,
                    f"Word '{w.text}' starts at {w.start} before prev end {prev_end}")
                prev_end = w.end


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestAvailableBackends(unittest.TestCase):
    """Test backend discovery."""

    def test_available_backends(self):
        from crispasr import Session
        backends = Session.available_backends(lib_path=LIB_PATH)
        self.assertIsInstance(backends, list)
        self.assertIn("whisper", backends)
        self.assertIn("parakeet", backends)

    def test_backend_autodetect(self):
        """Session should auto-detect backend from GGUF metadata."""
        if not os.path.exists(PARAKEET_MODEL):
            self.skipTest("Parakeet model not available")
        from crispasr import Session
        with Session(PARAKEET_MODEL, lib_path=LIB_PATH) as s:
            self.assertEqual(s.backend, "parakeet")


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestRegistryAndCache(unittest.TestCase):
    """Test model registry and cache helpers."""

    def test_cache_dir(self):
        from crispasr import cache_dir
        d = cache_dir(lib_path=LIB_PATH)
        self.assertIsInstance(d, str)
        self.assertGreater(len(d), 0)

    def test_registry_lookup(self):
        from crispasr import registry_lookup
        entry = registry_lookup("parakeet", lib_path=LIB_PATH)
        # May return None if registry not compiled in, or a RegistryEntry
        if entry is not None:
            self.assertIsInstance(entry.filename, str)
            self.assertGreater(len(entry.filename), 0)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestKokoroPhonemeCacheClear(unittest.TestCase):
    """PLAN #56 #5: kokoro_phoneme_cache_clear C ABI is exposed and callable.

    Doesn't require a kokoro model — only verifies the symbol export and
    the null-handle return path.
    """

    def test_symbol_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        self.assertTrue(hasattr(lib, "crispasr_session_kokoro_clear_phoneme_cache"))

    def test_null_handle_returns_neg_one(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        fn = lib.crispasr_session_kokoro_clear_phoneme_cache
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        self.assertEqual(fn(ctypes.c_void_p(0)), -1)


@unittest.skipUnless(LIB_PATH, "libwhisper not built")
class TestKokoroPhonemizerParityHarness(unittest.TestCase):
    """PLAN #56 #4: phonemizer-diff harness ABI is exposed.

    Both kokoro_phonemize_text_lib and kokoro_phonemize_text_popen
    must be present for `tools/check_kokoro_phonemizer_parity.py` to
    work. Doesn't require espeak-ng to be installed — only checks the
    symbols are exported.
    """

    def test_symbols_exported(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        self.assertTrue(hasattr(lib, "kokoro_phonemize_text_lib"))
        self.assertTrue(hasattr(lib, "kokoro_phonemize_text_popen"))

    def test_null_args_return_nullptr(self):
        import ctypes
        lib = ctypes.CDLL(LIB_PATH)
        for name in ("kokoro_phonemize_text_lib", "kokoro_phonemize_text_popen"):
            fn = getattr(lib, name)
            fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            fn.restype = ctypes.c_void_p
            self.assertIsNone(fn(None, None), f"{name}(None, None) should return nullptr")
            self.assertIsNone(fn(b"en-us", None), f"{name}('en-us', None) should return nullptr")
            self.assertIsNone(fn(None, b"hello"), f"{name}(None, 'hello') should return nullptr")


if __name__ == "__main__":
    unittest.main(verbosity=2)
