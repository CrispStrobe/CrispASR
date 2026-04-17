"""CrispASR Python wrapper via ctypes.

Provides speech-to-text transcription using ggml inference.
Wraps the whisper.h C API from whisper.cpp / CrispASR.
"""

import ctypes
import os
import platform
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass
class Segment:
    """A transcription segment with timing information."""
    text: str
    start: float  # seconds
    end: float    # seconds
    no_speech_prob: float = 0.0


def _find_lib():
    """Find the whisper shared library."""
    names = {
        "Linux": "libwhisper.so",
        "Darwin": "libwhisper.dylib",
        "Windows": "whisper.dll",
    }
    lib_name = names.get(platform.system(), "libwhisper.so")

    search = [
        Path(__file__).parent,
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "src",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path.cwd() / "build",
        Path.cwd() / "build" / "src",
    ]
    for d in search:
        p = d / lib_name
        if p.exists():
            return str(p)
    return lib_name


# Whisper sampling strategies
WHISPER_SAMPLING_GREEDY = 0
WHISPER_SAMPLING_BEAM_SEARCH = 1


class CrispASR:
    """Speech-to-text model using ggml inference.

    Usage:
        model = CrispASR("ggml-base.en.bin")
        segments = model.transcribe("audio.wav")
        for seg in segments:
            print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

        # Or from raw PCM data
        segments = model.transcribe_pcm(pcm_f32, sample_rate=16000)

        model.close()
    """

    def __init__(self, model_path: str, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._setup_signatures()

        # Create context params
        self._lib.whisper_context_default_params_by_ref.restype = ctypes.c_void_p
        params_ptr = self._lib.whisper_context_default_params_by_ref()

        # Init model
        self._ctx = self._lib.whisper_init_from_file_with_params(
            model_path.encode("utf-8"), params_ptr
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {model_path}")

    def _setup_signatures(self):
        lib = self._lib

        # Init
        lib.whisper_init_from_file_with_params.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        lib.whisper_init_from_file_with_params.restype = ctypes.c_void_p

        # Free
        lib.whisper_free.argtypes = [ctypes.c_void_p]
        lib.whisper_free.restype = None

        # Full inference
        lib.whisper_full.argtypes = [
            ctypes.c_void_p,  # ctx
            ctypes.c_void_p,  # params (by ref)
            ctypes.POINTER(ctypes.c_float),  # samples
            ctypes.c_int,     # n_samples
        ]
        lib.whisper_full.restype = ctypes.c_int

        # Default params
        lib.whisper_full_default_params_by_ref.argtypes = [ctypes.c_int]
        lib.whisper_full_default_params_by_ref.restype = ctypes.c_void_p

        # Results
        lib.whisper_full_n_segments.argtypes = [ctypes.c_void_p]
        lib.whisper_full_n_segments.restype = ctypes.c_int

        lib.whisper_full_get_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_text.restype = ctypes.c_char_p

        lib.whisper_full_get_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_t0.restype = ctypes.c_int64

        lib.whisper_full_get_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_t1.restype = ctypes.c_int64

        lib.whisper_full_get_segment_no_speech_prob.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.whisper_full_get_segment_no_speech_prob.restype = ctypes.c_float

        # Language
        lib.whisper_full_lang_id.argtypes = [ctypes.c_void_p]
        lib.whisper_full_lang_id.restype = ctypes.c_int

        lib.whisper_lang_str.argtypes = [ctypes.c_int]
        lib.whisper_lang_str.restype = ctypes.c_char_p

        # Free params
        lib.whisper_free_params.argtypes = [ctypes.c_void_p]
        lib.whisper_free_params.restype = None

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        strategy: int = WHISPER_SAMPLING_GREEDY,
    ) -> List[Segment]:
        """Transcribe an audio file (WAV, 16kHz mono recommended).

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g. "en", "de") or "auto" for detection.
            strategy: WHISPER_SAMPLING_GREEDY or WHISPER_SAMPLING_BEAM_SEARCH.

        Returns:
            List of Segment objects with text and timing.
        """
        pcm = self._load_audio(audio_path)
        return self.transcribe_pcm(pcm, language=language, strategy=strategy)

    def transcribe_pcm(
        self,
        pcm: np.ndarray,
        sample_rate: int = 16000,
        language: str = "auto",
        strategy: int = WHISPER_SAMPLING_GREEDY,
    ) -> List[Segment]:
        """Transcribe raw PCM audio data.

        Args:
            pcm: Float32 mono PCM samples.
            sample_rate: Sample rate (will be resampled to 16kHz if different).
            language: Language code or "auto".
            strategy: Sampling strategy.

        Returns:
            List of Segment objects.
        """
        if sample_rate != 16000:
            # Simple resampling via linear interpolation
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)

        pcm = pcm.astype(np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Get default params
        params_ptr = self._lib.whisper_full_default_params_by_ref(strategy)

        # Run inference
        ret = self._lib.whisper_full(self._ctx, params_ptr, samples_ptr, len(pcm))
        self._lib.whisper_free_params(params_ptr)

        if ret != 0:
            raise RuntimeError(f"Transcription failed (error code {ret})")

        # Collect segments
        n_segments = self._lib.whisper_full_n_segments(self._ctx)
        segments = []
        for i in range(n_segments):
            text_bytes = self._lib.whisper_full_get_segment_text(self._ctx, i)
            text = text_bytes.decode("utf-8") if text_bytes else ""
            t0 = self._lib.whisper_full_get_segment_t0(self._ctx, i) / 100.0
            t1 = self._lib.whisper_full_get_segment_t1(self._ctx, i) / 100.0
            nsp = float(self._lib.whisper_full_get_segment_no_speech_prob(self._ctx, i))
            segments.append(Segment(text=text, start=t0, end=t1, no_speech_prob=nsp))

        return segments

    @property
    def detected_language(self) -> str:
        """Language detected during the last transcription."""
        lang_id = self._lib.whisper_full_lang_id(self._ctx)
        lang_str = self._lib.whisper_lang_str(lang_id)
        return lang_str.decode("utf-8") if lang_str else "unknown"

    @staticmethod
    def _load_audio(path: str) -> np.ndarray:
        """Load audio file to float32 mono PCM."""
        if path.endswith(".wav"):
            with wave.open(path, "rb") as wf:
                assert wf.getsampwidth() in (1, 2, 4), "Unsupported sample width"
                assert wf.getnchannels() in (1, 2), "Unsupported channel count"
                frames = wf.readframes(wf.getnframes())
                if wf.getsampwidth() == 2:
                    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                elif wf.getsampwidth() == 4:
                    pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
                # Convert stereo to mono
                if wf.getnchannels() == 2:
                    pcm = pcm.reshape(-1, 2).mean(axis=1)
                # Resample if needed
                if wf.getframerate() != 16000:
                    ratio = 16000 / wf.getframerate()
                    new_len = int(len(pcm) * ratio)
                    indices = np.linspace(0, len(pcm) - 1, new_len)
                    pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
                return pcm
        else:
            raise ValueError(f"Unsupported audio format: {path}. Use .wav or pass raw PCM via transcribe_pcm().")

    def close(self):
        """Release all resources."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.whisper_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
