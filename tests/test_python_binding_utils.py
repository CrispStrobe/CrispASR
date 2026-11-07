"""Unit tests for the standalone utility functions added to the Python binding.

These tests validate the function signatures and error handling without
requiring a loaded model. They import _binding.py and check that the
new functions exist and have correct type annotations.
"""

import importlib
import inspect
import sys
from pathlib import Path

import pytest

# Import the binding module
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))
from crispasr import _binding


def test_ultra_alignment_preserves_precision_and_frees_result(monkeypatch):
    import numpy as np

    class Function:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    class Library:
        def __init__(self):
            self.freed = []
            self.crispasr_align_words_ultra_abi = Function(lambda *args: 123)
            self.crispasr_ultra_align_result_n_words = Function(lambda _: 1)
            self.crispasr_ultra_align_result_word_text = Function(lambda _, i: "你好".encode())
            self.crispasr_ultra_align_result_word_start_ms = Function(lambda _, i: 12.345)
            self.crispasr_ultra_align_result_word_end_ms = Function(lambda _, i: 67.891)
            self.crispasr_ultra_align_result_free = Function(self.freed.append)

    lib = Library()
    monkeypatch.setattr(_binding.ctypes, "CDLL", lambda _: lib)
    words = _binding.align_words_ultra("forced-aligner.gguf", "head.gguf", "你好", np.zeros(8, dtype=np.float32),
                                      lib_path="mock")
    assert len(words) == 1 and words[0].text == "你好"
    assert words[0].start == pytest.approx(0.012345)
    assert words[0].end == pytest.approx(0.067891)
    assert lib.freed == [123]


def test_ultra_alignment_invalid_and_missing_abi(monkeypatch):
    import numpy as np

    audio = np.zeros(8, dtype=np.float32)
    assert _binding.align_words_ultra("", "head", "hello", audio) == []
    assert _binding.align_words_ultra("forced-aligner", "", "hello", audio) == []
    monkeypatch.setattr(_binding.ctypes, "CDLL", lambda _: object())
    with pytest.raises(RuntimeError, match="does not provide Ultra"):
        _binding.align_words_ultra("forced-aligner", "head", "hello", audio, lib_path="mock")


class TestFunctionSignatures:
    """Verify new standalone functions exist with correct signatures."""

    def test_vad_segments_exists(self):
        assert hasattr(_binding, "vad_segments")
        sig = inspect.signature(_binding.vad_segments)
        assert "pcm" in sig.parameters
        assert "model_path" in sig.parameters
        assert "threshold" in sig.parameters

    def test_text_detect_language_exists(self):
        assert hasattr(_binding, "text_detect_language")
        sig = inspect.signature(_binding.text_detect_language)
        assert "text" in sig.parameters
        assert "model_path" in sig.parameters

    def test_enhance_audio_rnnoise_exists(self):
        assert hasattr(_binding, "enhance_audio_rnnoise")
        sig = inspect.signature(_binding.enhance_audio_rnnoise)
        assert "pcm" in sig.parameters

    def test_detect_backend_from_gguf_exists(self):
        assert hasattr(_binding, "detect_backend_from_gguf")
        sig = inspect.signature(_binding.detect_backend_from_gguf)
        assert "gguf_path" in sig.parameters

    def test_detect_backend_from_gguf_returns_name(self):
        """Regression: the ABI returns strlen(name) (>0) on a successful
        detection, so a `rc != 0`-means-failure check wrongly rejected every
        real detection (silently nulled / raised). Guard the correct contract
        against a checked-in fixture."""
        fixture = Path(__file__).parent / "fixtures" / "vibevoice-quant-carveout.gguf"
        if not fixture.exists():
            pytest.skip("fixture GGUF not present")
        try:
            name = _binding.detect_backend_from_gguf(str(fixture))
        except OSError:
            pytest.skip("native crispasr library not built/loadable")
        assert name == "vibevoice-tts", f"expected 'vibevoice-tts', got {name!r}"

    def test_vad_span_dataclass(self):
        span = _binding.VadSpan(start=1.0, end=2.5)
        assert span.start == 1.0
        assert span.end == 2.5


class TestSessionTranslateText:
    """Verify translate_text method exists on Session."""

    def test_translate_text_method_exists(self):
        assert hasattr(_binding.Session, "translate_text")
        sig = inspect.signature(_binding.Session.translate_text)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "src_lang" in params
        assert "tgt_lang" in params


class TestSegmentDataclass:
    """Verify Segment dataclass."""

    def test_segment_fields(self):
        seg = _binding.Segment(text="hello", start=0.0, end=1.0)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.no_speech_prob == 0.0

    def test_segment_with_no_speech(self):
        seg = _binding.Segment(text="", start=0.0, end=0.5, no_speech_prob=0.9)
        assert seg.no_speech_prob == 0.9


class TestModuleExports:
    """Verify the module exports the expected public API."""

    def test_core_classes(self):
        assert hasattr(_binding, "Session")
        assert hasattr(_binding, "Segment")
        assert hasattr(_binding, "PuncModel")

    def test_standalone_functions(self):
        # All functions added in PLAN #59
        for name in ["vad_segments", "text_detect_language",
                      "enhance_audio_rnnoise", "detect_backend_from_gguf"]:
            assert hasattr(_binding, name), f"missing: {name}"

    def test_detect_language_pcm_exists(self):
        """The module-level audio LID function should exist."""
        assert hasattr(_binding, "detect_language_pcm")
