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
sys.path.insert(0, str(Path(__file__).parent.parent / "python" / "crispasr"))
import _binding


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
