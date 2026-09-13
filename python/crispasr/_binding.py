"""CrispASR Python wrapper via ctypes.

Provides speech-to-text transcription using ggml inference.
Wraps the whisper.h C API from crispasr / CrispASR.
"""

import codecs
import contextlib
import ctypes
import os
import platform
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


@dataclass
class Segment:
    """A transcription segment with timing information."""
    text: str
    start: float  # seconds
    end: float    # seconds
    no_speech_prob: float = 0.0


def _find_lib():
    """Locate the crispasr / whisper shared library.

    Platform wheels (CPU on PyPI, GPU on the extra index) bundle the native
    library *next to this file*, so that copy is probed first and always wins.
    The pure-Python sdist does not bundle it; there the user supplies
    `libcrispasr.{so,dylib,dll}` via a package manager (Homebrew, apt), a
    source build, or `$CRISPASR_LIB_PATH`.

    Probe order:
      1. $CRISPASR_LIB_PATH (explicit override — full path to the .so/.dylib/.dll)
      2. This package directory (a lib bundled into the installed wheel)
      3. sys.prefix/lib (pip install --user, virtualenv, conda)
      4. Standard install prefixes (Homebrew arm64/x64, /usr/local, /usr)
      5. Repo-relative `build/` paths (for `pip install -e .` from a clone)
      6. The bare filename (lets the loader use $LD_LIBRARY_PATH /
         $DYLD_LIBRARY_PATH / PATH and the system loader cache)

    Both `libcrispasr.*` (preferred — all backends) and the legacy
    `libwhisper.*` alias are accepted at every step.
    """
    import sys

    system = platform.system()
    if system == "Darwin":
        candidates = ["libcrispasr.dylib", "libwhisper.dylib"]
    elif system == "Windows":
        candidates = ["crispasr.dll", "whisper.dll"]
    else:
        candidates = ["libcrispasr.so", "libwhisper.so"]

    override = os.environ.get("CRISPASR_LIB_PATH")
    if override and Path(override).exists():
        _register_dll_dir(Path(override).parent)
        return override

    search = [
        # A lib bundled into the installed wheel — probed first so a platform
        # wheel is self-contained and a stray system copy can't shadow it.
        Path(__file__).parent,
        Path(sys.prefix) / "lib",
        Path("/opt/homebrew/lib"),  # macOS arm64 Homebrew
        Path("/usr/local/lib"),     # macOS x64 Homebrew, /usr/local installs
        Path("/usr/lib"),           # apt, dnf
        Path("/usr/lib/x86_64-linux-gnu"),  # Debian/Ubuntu multiarch
        Path("/usr/lib/aarch64-linux-gnu"),
        # Repo-relative — for `pip install -e .` from a clone.
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "src",
        Path(__file__).parent.parent.parent / "build" / "lib",
        Path.cwd() / "build",
        Path.cwd() / "build" / "src",
    ]
    for d in search:
        for name in candidates:
            p = d / name
            if p.exists():
                _register_dll_dir(p.parent)
                return str(p)
    # Fall back to bare name; ctypes will use the system loader path.
    return candidates[0]


def _register_dll_dir(directory: Path) -> None:
    """Windows only: make `directory` searchable for DEPENDENT DLLs.

    `crispasr.dll` needs `ggml.dll`, `ggml-base.dll` and `ggml-cpu.dll`, which a
    platform wheel ships right beside it. Since Python 3.8 the directory of a DLL
    loaded by full path is NOT searched for that DLL's own dependencies, so
    `ctypes.CDLL(<path>/crispasr.dll)` raises "DLL load failed while importing"
    even though every dependency is present in the same folder.
    `os.add_dll_directory` is the documented remedy.

    No-op off Windows, and deliberately quiet: on a system install the loader
    already resolves these, so a failure here must not break a working setup.
    """
    if os.name != "nt":
        return
    add = getattr(os, "add_dll_directory", None)  # Python 3.8+
    if add is None:
        return
    key = str(directory)
    if key in _dll_dirs:
        return
    try:
        add(key)          # keep the cookie alive for the process lifetime
        _dll_dirs[key] = True
    except OSError:
        pass


# Directories already handed to os.add_dll_directory (Windows).
_dll_dirs: dict = {}


# Whisper sampling strategies
CRISPASR_SAMPLING_GREEDY = 0
CRISPASR_SAMPLING_BEAM_SEARCH = 1


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

    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 helpers_lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._setup_signatures()

        # Load helpers library (provides pointer-based wrappers for by-value struct APIs)
        helpers_search = [
            helpers_lib_path,
            str(Path(lib_path).parent / "libcrispasr_helpers.so") if lib_path else None,
            str(Path(__file__).parent.parent.parent / "build" / "libcrispasr_helpers.so"),
        ]
        self._helpers = None
        for hp in helpers_search:
            if hp and Path(hp).exists():
                self._helpers = ctypes.CDLL(hp)
                break

        if self._helpers:
            # Use pointer-based wrappers (avoids by-value struct issues)
            self._helpers.whisper_init_from_file_ptr.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
            self._helpers.whisper_init_from_file_ptr.restype = ctypes.c_void_p
            self._helpers.whisper_full_ptr.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ]
            self._helpers.whisper_full_ptr.restype = ctypes.c_int

            cparams = self._lib.whisper_context_default_params_by_ref()
            self._ctx = self._helpers.whisper_init_from_file_ptr(
                model_path.encode("utf-8"), cparams
            )
            self._lib.whisper_free_context_params(cparams)
        else:
            # Fallback: use deprecated simple init (no params)
            self._lib.whisper_init_from_file.argtypes = [ctypes.c_char_p]
            self._lib.whisper_init_from_file.restype = ctypes.c_void_p
            self._ctx = self._lib.whisper_init_from_file(model_path.encode("utf-8"))

        if not self._ctx:
            raise RuntimeError(f"Failed to load model: {model_path}")

    def _setup_signatures(self):
        lib = self._lib

        # Free
        lib.whisper_free.argtypes = [ctypes.c_void_p]
        lib.whisper_free.restype = None

        # Context params (by ref)
        lib.whisper_context_default_params_by_ref.argtypes = []
        lib.whisper_context_default_params_by_ref.restype = ctypes.c_void_p

        lib.whisper_free_context_params.argtypes = [ctypes.c_void_p]
        lib.whisper_free_context_params.restype = None

        # Full params (by ref)
        lib.whisper_full_default_params_by_ref.argtypes = [ctypes.c_int]
        lib.whisper_full_default_params_by_ref.restype = ctypes.c_void_p

        lib.whisper_free_params.argtypes = [ctypes.c_void_p]
        lib.whisper_free_params.restype = None

        # whisper_full (takes params by value — needs helpers lib for pointer variant)
        lib.whisper_full.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.whisper_full.restype = ctypes.c_int

        # Results (ctx-based variants)
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

        # Parameter setters on whisper_full_params — all void(ptr, val).
        for _sym, _argtypes in [
            ("crispasr_params_set_language", [ctypes.c_void_p, ctypes.c_char_p]),
            ("crispasr_params_set_translate", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_detect_language", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_token_timestamps", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_n_threads", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_max_len", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_best_of", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_split_on_word", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_no_context", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_single_segment", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_print_realtime", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_print_progress", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_print_timestamps", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_print_special", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_suppress_blank", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_temperature", [ctypes.c_void_p, ctypes.c_float]),
            ("crispasr_params_set_max_tokens", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_initial_prompt", [ctypes.c_void_p, ctypes.c_char_p]),
            ("crispasr_params_set_alt_n", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_vad", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_vad_model_path", [ctypes.c_void_p, ctypes.c_char_p]),
            ("crispasr_params_set_vad_threshold", [ctypes.c_void_p, ctypes.c_float]),
            ("crispasr_params_set_vad_min_speech_ms", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_vad_min_silence_ms", [ctypes.c_void_p, ctypes.c_int]),
            ("crispasr_params_set_tdrz", [ctypes.c_void_p, ctypes.c_int]),
        ]:
            if hasattr(lib, _sym):
                getattr(lib, _sym).argtypes = _argtypes
                getattr(lib, _sym).restype = None

        # Token-level accessors (0.5.x).
        for _sym, _args, _ret in [
            ("crispasr_token_t0", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int64),
            ("crispasr_token_t1", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int64),
            ("crispasr_token_p", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_float),
            ("crispasr_token_n_alts", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], ctypes.c_int),
            ("crispasr_token_alt_id", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int32),
            ("crispasr_token_alt_p", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_float),
            ("crispasr_token_alt_text", [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                         ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
        ]:
            if hasattr(lib, _sym):
                getattr(lib, _sym).argtypes = _args
                getattr(lib, _sym).restype = _ret

        # Language detection (whisper context).
        if hasattr(lib, "crispasr_detect_language"):
            lib.crispasr_detect_language.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ]
            lib.crispasr_detect_language.restype = ctypes.c_float

        # VAD free + slices.
        if hasattr(lib, "crispasr_vad_free"):
            lib.crispasr_vad_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
            lib.crispasr_vad_free.restype = None
        if hasattr(lib, "crispasr_vad_slices"):
            lib.crispasr_vad_slices.argtypes = [
                ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_float, ctypes.c_int,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ]
            lib.crispasr_vad_slices.restype = ctypes.c_int

        # Streaming (whisper context).
        if hasattr(lib, "crispasr_stream_open"):
            lib.crispasr_stream_open.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
            ]
            lib.crispasr_stream_open.restype = ctypes.c_void_p

    def transcribe(
        self,
        audio_path: str,
        language: str = "auto",
        strategy: int = CRISPASR_SAMPLING_GREEDY,
    ) -> List[Segment]:
        """Transcribe an audio file (WAV, 16kHz mono recommended).

        Args:
            audio_path: Path to audio file.
            language: Language code (e.g. "en", "de") or "auto" for detection.
            strategy: CRISPASR_SAMPLING_GREEDY or CRISPASR_SAMPLING_BEAM_SEARCH.

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
        strategy: int = CRISPASR_SAMPLING_GREEDY,
        vad: bool = False,
        vad_model_path: Optional[str] = None,
        vad_threshold: float = 0.5,
        vad_min_speech_ms: int = 250,
        vad_min_silence_ms: int = 100,
        tdrz: bool = False,
    ) -> List[Segment]:
        """Transcribe raw PCM audio data.

        Args:
            pcm: Float32 mono PCM samples.
            sample_rate: Sample rate (will be resampled to 16kHz if different).
            language: Language code or "auto".
            strategy: Sampling strategy.
            vad: Enable Silero VAD to skip silent regions (0.4.2+ dylibs).
            vad_model_path: Path to Silero VAD GGML model. Required when vad=True.
            vad_threshold: Speech detection threshold (0.0-1.0, default 0.5).
            vad_min_speech_ms: Minimum speech span to keep (default 250ms).
            vad_min_silence_ms: Minimum silence span to split on (default 100ms).
            tdrz: Enable tinydiarize speaker-turn markers (requires .en.tdrz model).

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

        # 0.4.2: VAD + tdrz. Setters are optional — older dylibs don't
        # have them, the lookup-time hasattr() guard skipped the argtypes
        # declaration so these calls no-op silently.
        if vad and hasattr(self._lib, "crispasr_params_set_vad"):
            self._lib.crispasr_params_set_vad(params_ptr, 1)
            if hasattr(self._lib, "crispasr_params_set_vad_threshold"):
                self._lib.crispasr_params_set_vad_threshold(params_ptr, vad_threshold)
            if hasattr(self._lib, "crispasr_params_set_vad_min_speech_ms"):
                self._lib.crispasr_params_set_vad_min_speech_ms(params_ptr, vad_min_speech_ms)
            if hasattr(self._lib, "crispasr_params_set_vad_min_silence_ms"):
                self._lib.crispasr_params_set_vad_min_silence_ms(params_ptr, vad_min_silence_ms)
            if vad_model_path and hasattr(self._lib, "crispasr_params_set_vad_model_path"):
                self._lib.crispasr_params_set_vad_model_path(
                    params_ptr, vad_model_path.encode("utf-8")
                )
        if tdrz and hasattr(self._lib, "crispasr_params_set_tdrz"):
            self._lib.crispasr_params_set_tdrz(params_ptr, 1)

        # Run inference
        if self._helpers:
            ret = self._helpers.whisper_full_ptr(self._ctx, params_ptr, samples_ptr, len(pcm))
        else:
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


# =========================================================================
# Unified session — works for every backend libcrispasr was built with
# =========================================================================

@dataclass
class SessionWord:
    """Word-level timing from a session transcribe (backends that produce it)."""
    text: str
    start: float  # seconds
    end: float    # seconds
    confidence: float = 1.0  # softmax probability in [0, 1]; 1.0 if backend doesn't emit one


@dataclass
class SessionSegment:
    """A transcription segment from Session.transcribe."""
    text: str
    start: float  # seconds
    end: float    # seconds
    words: List[SessionWord]
    # Whisper's per-segment no-speech probability (the <|nospeech|> posterior)
    # in [0, 1]. Whisper-only; other backends (and older libcrispasr builds
    # without the accessor) leave the -1.0 "no data" sentinel.
    no_speech_prob: float = -1.0
    # Native per-segment speaker label from a backend that diarizes on its own,
    # in the "(Speaker N) " form the CLI prefixes into text/srt/vtt output, or
    # "" when the backend produced none (and on older libcrispasr builds without
    # the accessor). Populated today by vibevoice, whose model answers with a
    # Start/End/Speaker/Content array. The ordinals are CHUNK-LOCAL: "Speaker 1"
    # in one transcribe call is not necessarily the same voice as "Speaker 1" in
    # the next — use the diarize_* helpers for cross-recording clustering.
    speaker: str = ""


# =========================================================================
# Diarization (shared C-ABI, 0.4.5+)
# =========================================================================

class DiarizeMethod:
    """Diarization method identifiers matching the C-ABI enum."""
    ENERGY = 0      # stereo only
    XCORR = 1       # stereo only
    VAD_TURNS = 2   # mono-friendly, timing-based
    PYANNOTE = 3    # mono-friendly, GGUF pyannote seg model
    FOXNOSE = 4     # mono-friendly, WeSpeaker + spectral clustering


@dataclass
class DiarizeSegment:
    """One ASR segment passed in to :func:`diarize_segments`.

    The caller fills ``t0`` / ``t1`` (seconds) from the upstream
    transcribe result; the diarizer writes the zero-based speaker index
    into ``speaker`` (``-1`` means the method had no info to pick).
    """
    t0: float
    t1: float
    speaker: int = -1


@dataclass
class DiarizeTurn:
    """One audio-derived speaker turn returned by FoxNose.

    Times are seconds on the same absolute timeline as
    :class:`DiarizeSegment`. Other diarization methods return no turns.
    """
    t0: float
    t1: float
    speaker: int


class _DiarizeSegAbi(ctypes.Structure):
    _fields_ = [
        ("t0_cs", ctypes.c_int64),
        ("t1_cs", ctypes.c_int64),
        ("speaker", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


class _DiarizeOptsAbi(ctypes.Structure):
    # APPEND-ONLY mirror of crispasr_diarize_opts_abi. Keep the explicit
    # padding field: native code reads all 48 bytes on 64-bit platforms.
    _fields_ = [
        ("method", ctypes.c_int32),
        ("n_threads", ctypes.c_int32),
        ("slice_t0_cs", ctypes.c_int64),
        ("pyannote_model_path", ctypes.c_char_p),
        ("foxnose_embedder_path", ctypes.c_char_p),
        ("min_speakers", ctypes.c_int32),
        ("max_speakers", ctypes.c_int32),
        ("num_speakers", ctypes.c_int32),
        ("_pad2", ctypes.c_int32),
    ]


class _DiarizeTurnAbi(ctypes.Structure):
    _fields_ = [
        ("t0_cs", ctypes.c_int64),
        ("t1_cs", ctypes.c_int64),
        ("speaker", ctypes.c_int32),
        ("_pad", ctypes.c_int32),
    ]


# =========================================================================
# Language identification (shared C-ABI, 0.4.6+)
# =========================================================================

class LidMethod:
    """LID method identifiers matching the C-ABI enum."""
    WHISPER = 0
    SILERO = 1


@dataclass
class LidResult:
    """Result from :func:`detect_language_pcm`. ``lang_code`` is ISO 639-1."""
    lang_code: str
    confidence: float


# =========================================================================
# CTC / forced-aligner word timings (shared C-ABI, 0.4.7+)
# =========================================================================

@dataclass
class AlignedWord:
    """Per-word output of :func:`align_words`."""
    text: str
    start: float  # seconds (centiseconds / 100 on the C side)
    end: float


# =========================================================================
# Cache + model registry (shared C-ABI, 0.4.8+)
# =========================================================================

@dataclass
class RegistryEntry:
    """Known-model registry entry."""
    filename: str
    url: str
    approx_size: str


@dataclass
class RegistryArtifact:
    """One file in a backend's canonical default download bundle."""
    kind: str
    filename: str
    url: str
    approx_size: str


@dataclass
class RegistryBundle:
    """The exact artifact bundle downloaded by ``-m auto``."""
    backend: str
    license: str
    requires_acceptance: bool
    artifacts: List[RegistryArtifact]


def registry_lookup(backend: str, *, lib_path: Optional[str] = None) -> Optional[RegistryEntry]:
    """Look up the canonical GGUF for a backend. Returns ``None`` on miss."""
    return _registry_call("crispasr_registry_lookup_abi", backend, lib_path)


def registry_lookup_by_filename(filename: str, *, lib_path: Optional[str] = None) -> Optional[RegistryEntry]:
    """Look up the canonical GGUF by filename (exact, then fuzzy substring)."""
    return _registry_call("crispasr_registry_lookup_by_filename_abi", filename, lib_path)


def registry_default_bundle(
    backend: str, *, lib_path: Optional[str] = None
) -> Optional[RegistryBundle]:
    """Return the backend's exact canonical ``-m auto`` artifact bundle.

    Artifacts are ordered as downloaded: primary model, inline companion,
    then any extra companions. This API does not rewrite quant suffixes or
    infer a recommendation. Returns ``None`` for an unknown backend.
    """
    if not backend:
        return None
    lib = ctypes.CDLL(lib_path or _find_lib())
    info_symbol = "crispasr_registry_default_bundle_info_abi"
    artifact_symbol = "crispasr_registry_default_bundle_artifact_abi"
    if not hasattr(lib, info_symbol) or not hasattr(lib, artifact_symbol):
        raise RuntimeError(
            "default-bundle registry API not in loaded library — rebuild CrispASR."
        )

    info = getattr(lib, info_symbol)
    info.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    info.restype = ctypes.c_int
    backend_buf = ctypes.create_string_buffer(256)
    license_buf = ctypes.create_string_buffer(1024)
    requires_acceptance = ctypes.c_int32()
    count = info(
        backend.encode("utf-8"),
        backend_buf, len(backend_buf),
        license_buf, len(license_buf),
        ctypes.byref(requires_acceptance),
    )
    if count == 0:
        return None
    if count < 0:
        raise RuntimeError(f"default-bundle registry lookup failed (rc={count}).")

    artifact_fn = getattr(lib, artifact_symbol)
    artifact_fn.argtypes = [
        ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
    ]
    artifact_fn.restype = ctypes.c_int
    kinds = {0: "primary", 1: "companion", 2: "extra"}
    artifacts = []
    for index in range(count):
        kind = ctypes.c_int32()
        filename_buf = ctypes.create_string_buffer(256)
        url_buf = ctypes.create_string_buffer(2048)
        size_buf = ctypes.create_string_buffer(64)
        rc = artifact_fn(
            backend.encode("utf-8"), index, ctypes.byref(kind),
            filename_buf, len(filename_buf),
            url_buf, len(url_buf),
            size_buf, len(size_buf),
        )
        if rc != 0:
            raise RuntimeError(
                f"default-bundle artifact {index} lookup failed (rc={rc})."
            )
        if kind.value not in kinds:
            raise RuntimeError(
                f"default-bundle artifact {index} has unknown kind {kind.value}."
            )
        artifacts.append(RegistryArtifact(
            kind=kinds[kind.value],
            filename=filename_buf.value.decode("utf-8"),
            url=url_buf.value.decode("utf-8"),
            approx_size=size_buf.value.decode("utf-8"),
        ))

    return RegistryBundle(
        backend=backend_buf.value.decode("utf-8"),
        license=license_buf.value.decode("utf-8"),
        requires_acceptance=requires_acceptance.value != 0,
        artifacts=artifacts,
    )


def list_known_models(*, lib_path: Optional[str] = None) -> list:
    """Return every backend name in the registry, in declaration order.

    Useful for wrappers building UIs (model picker, "what can I download?").
    Each name can be passed back to :func:`registry_lookup` for full
    details (filename, URL, approximate size).
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_registry_list_backends_abi"):
        return []
    fn = lib.crispasr_registry_list_backends_abi
    fn.argtypes = [ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(8192)
    n = fn(buf, 8192)
    if n < 0:
        return []
    csv = buf.value.decode("utf-8")
    return [s for s in csv.split(",") if s]


def _registry_call(sym: str, key: str, lib_path: Optional[str]) -> Optional[RegistryEntry]:
    if not key:
        return None
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, sym):
        raise RuntimeError(f"{sym} not in loaded library — rebuild CrispASR 0.4.8+.")
    fn = getattr(lib, sym)
    fn.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32,
    ]
    fn.restype = ctypes.c_int
    fn_buf = ctypes.create_string_buffer(256)
    url_buf = ctypes.create_string_buffer(512)
    size_buf = ctypes.create_string_buffer(32)
    rc = fn(
        key.encode("utf-8"),
        fn_buf, 256,
        url_buf, 512,
        size_buf, 32,
    )
    if rc != 0:
        return None
    return RegistryEntry(
        filename=fn_buf.value.decode("utf-8"),
        url=url_buf.value.decode("utf-8"),
        approx_size=size_buf.value.decode("utf-8"),
    )


@dataclass
class KokoroResolved:
    """Result of :func:`kokoro_resolve_for_lang` — see that function."""
    model_path: str
    voice_path: Optional[str]
    voice_name: Optional[str]
    backbone_swapped: bool


def kokoro_resolve_for_lang(
    model_path: str,
    lang: str,
    *,
    lib_path: Optional[str] = None,
) -> KokoroResolved:
    """Resolve the kokoro model + fallback voice for ``lang``.

    Mirrors what the CLI does for ``--backend kokoro -l <lang>`` — see
    PLAN #56 opt 2b. Returns:

    - ``model_path``: the path to actually load (may differ from input
      when a German backbone sibling, ``kokoro-de-hui-base-f16.gguf``,
      sits next to the official Kokoro-82M baseline).
    - ``voice_path``/``voice_name``: the per-language fallback voice
      path + basename. ``None`` if the language has a native Kokoro-82M
      voice or no candidate exists in the model directory.
    - ``backbone_swapped``: True iff the model path was rewritten.

    Wrappers should call this *before* opening the Session so the
    routing kicks in even outside the CLI entry point.
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    out_model = ctypes.create_string_buffer(1024)
    out_voice = ctypes.create_string_buffer(1024)
    out_picked = ctypes.create_string_buffer(64)

    swapped = False
    if hasattr(lib, "crispasr_kokoro_resolve_model_for_lang_abi"):
        lib.crispasr_kokoro_resolve_model_for_lang_abi.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ]
        lib.crispasr_kokoro_resolve_model_for_lang_abi.restype = ctypes.c_int
        rc = lib.crispasr_kokoro_resolve_model_for_lang_abi(
            model_path.encode("utf-8"), (lang or "").encode("utf-8"),
            out_model, 1024,
        )
        if rc == 0:
            swapped = True
        elif rc < 0:
            raise RuntimeError("kokoro_resolve_model_for_lang: buffer too small")
        # rc == 1 => no swap; out_model is the original path

    voice_path: Optional[str] = None
    voice_name: Optional[str] = None
    if hasattr(lib, "crispasr_kokoro_resolve_fallback_voice_abi"):
        lib.crispasr_kokoro_resolve_fallback_voice_abi.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        lib.crispasr_kokoro_resolve_fallback_voice_abi.restype = ctypes.c_int
        rc = lib.crispasr_kokoro_resolve_fallback_voice_abi(
            model_path.encode("utf-8"), (lang or "").encode("utf-8"),
            out_voice, 1024, out_picked, 64,
        )
        if rc == 0:
            voice_path = out_voice.value.decode("utf-8")
            voice_name = out_picked.value.decode("utf-8")
        elif rc < 0:
            raise RuntimeError("kokoro_resolve_fallback_voice: buffer too small")
        # rc == 1 (native voice) or 2 (no candidate) => leave voice_* as None

    return KokoroResolved(
        model_path=out_model.value.decode("utf-8") or model_path,
        voice_path=voice_path,
        voice_name=voice_name,
        backbone_swapped=swapped,
    )


def cache_ensure_file(
    filename: str,
    url: str,
    *,
    quiet: bool = False,
    cache_dir_override: Optional[str] = None,
    lib_path: Optional[str] = None,
) -> Optional[str]:
    """Return the path to a cached copy of ``filename``, downloading
    from ``url`` if missing. ``None`` on failure.
    """
    if not filename or not url:
        return None
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_cache_ensure_file_abi"):
        raise RuntimeError(
            "crispasr_cache_ensure_file_abi not in loaded library — rebuild CrispASR 0.4.8+."
        )
    lib.crispasr_cache_ensure_file_abi.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
    ]
    lib.crispasr_cache_ensure_file_abi.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(2048)
    rc = lib.crispasr_cache_ensure_file_abi(
        filename.encode("utf-8"),
        url.encode("utf-8"),
        1 if quiet else 0,
        (cache_dir_override or "").encode("utf-8"),
        buf, 2048,
    )
    return buf.value.decode("utf-8") if rc == 0 else None


def cache_dir(*, override: Optional[str] = None, lib_path: Optional[str] = None) -> Optional[str]:
    """Return the CrispASR cache directory (creating it if missing)."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_cache_dir_abi"):
        raise RuntimeError(
            "crispasr_cache_dir_abi not in loaded library — rebuild CrispASR 0.4.8+."
        )
    lib.crispasr_cache_dir_abi.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
    lib.crispasr_cache_dir_abi.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(2048)
    rc = lib.crispasr_cache_dir_abi((override or "").encode("utf-8"), buf, 2048)
    return buf.value.decode("utf-8") if rc == 0 else None


def align_words(
    aligner_model: str,
    transcript: str,
    pcm: np.ndarray,
    *,
    t_offset: float = 0.0,
    n_threads: int = 4,
    lib_path: Optional[str] = None,
) -> List[AlignedWord]:
    """Run CTC forced alignment for a transcript + audio pair.

    ``aligner_model`` filename picks the backend: paths containing
    "forced-aligner" / "qwen3-fa" / "qwen3-forced" route to the
    Qwen3-ForcedAligner path; everything else goes through
    canary-ctc-aligner.

    ``t_offset`` (seconds) is added to every word start/end so the
    returned timings are absolute against the original audio.
    Returns an empty list on failure.
    """
    if not aligner_model or not transcript or pcm is None or len(pcm) == 0:
        return []

    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_align_words_abi"):
        raise RuntimeError(
            "crispasr_align_words_abi not in loaded library — rebuild "
            "CrispASR 0.4.7+ to use forced alignment from the Python binding."
        )
    # (argtypes/restype wired in Session._setup_session_signatures when a
    # session is open; do it defensively here too so standalone use works.)
    lib.crispasr_align_words_abi.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.c_int64, ctypes.c_int32,
    ]
    lib.crispasr_align_words_abi.restype = ctypes.c_void_p
    lib.crispasr_align_result_n_words.argtypes = [ctypes.c_void_p]
    lib.crispasr_align_result_n_words.restype = ctypes.c_int
    lib.crispasr_align_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.crispasr_align_result_word_text.restype = ctypes.c_char_p
    lib.crispasr_align_result_word_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.crispasr_align_result_word_t0.restype = ctypes.c_int64
    lib.crispasr_align_result_word_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.crispasr_align_result_word_t1.restype = ctypes.c_int64
    lib.crispasr_align_result_free.argtypes = [ctypes.c_void_p]
    lib.crispasr_align_result_free.restype = None

    pcm_np = np.asarray(pcm, dtype=np.float32)
    res = lib.crispasr_align_words_abi(
        aligner_model.encode("utf-8"),
        transcript.encode("utf-8"),
        pcm_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(len(pcm_np)),
        int(round(t_offset * 100)),
        int(n_threads),
    )
    if not res:
        return []
    try:
        n = lib.crispasr_align_result_n_words(res)
        out: List[AlignedWord] = []
        for i in range(n):
            t = lib.crispasr_align_result_word_text(res, i)
            text = t.decode("utf-8") if t else ""
            out.append(AlignedWord(
                text=text,
                start=lib.crispasr_align_result_word_t0(res, i) / 100.0,
                end=lib.crispasr_align_result_word_t1(res, i) / 100.0,
            ))
        return out
    finally:
        lib.crispasr_align_result_free(res)


# ===========================================================================
# Microphone capture (PLAN #62d) — cross-platform live PCM via miniaudio.
# ===========================================================================

class Mic:
    """Library-level microphone capture handle.

    Wraps the C-ABI ``crispasr_mic_*`` functions which delegate to
    miniaudio's ``ma_device`` (Core Audio on macOS, ALSA/PulseAudio on
    Linux, WASAPI on Windows). The callback runs on miniaudio's audio
    thread — keep it short and non-blocking. To do streaming ASR,
    open a :meth:`Session.stream_open` handle and call
    ``stream.feed(pcm)`` from inside the callback (or queue and feed
    from another thread to avoid blocking the audio thread).

    Use as a context manager::

        with Mic(sample_rate=16000, callback=my_callback) as mic:
            mic.start()
            time.sleep(10)
            mic.stop()

    The :meth:`start_dictation` helper combines mic + stream + per-call
    feed for the common dictation use case.
    """

    # Holds the ctypes callback wrapper so it isn't garbage-collected
    # while the device is running. miniaudio's data callback is fired
    # from the audio thread; if the wrapper goes away we crash.
    _CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p)

    def __init__(self, *, sample_rate: int = 16000, channels: int = 1,
                 callback=None, lib_path: Optional[str] = None):
        if callback is None:
            raise ValueError("callback is required")
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        if not hasattr(self._lib, "crispasr_mic_open"):
            raise RuntimeError("mic API not present in this libcrispasr build")
        self._py_callback = callback

        def _trampoline(pcm_ptr, n_samples, _userdata):
            # Copy the audio thread's buffer into a numpy view; the
            # original miniaudio buffer is reused after we return.
            import numpy as np
            arr = np.ctypeslib.as_array(pcm_ptr, shape=(n_samples,)).copy()
            try:
                callback(arr)
            except Exception as e:
                import sys
                sys.stderr.write(f"crispasr.Mic callback raised: {e}\n")

        self._cb_holder = Mic._CB_TYPE(_trampoline)

        self._lib.crispasr_mic_open.argtypes = [
            ctypes.c_int, ctypes.c_int, Mic._CB_TYPE, ctypes.c_void_p,
        ]
        self._lib.crispasr_mic_open.restype = ctypes.c_void_p
        self._handle = self._lib.crispasr_mic_open(
            int(sample_rate), int(channels), self._cb_holder, None,
        )
        if not self._handle:
            raise RuntimeError("crispasr_mic_open failed")
        self._started = False

    def start(self) -> None:
        self._lib.crispasr_mic_start.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_mic_start.restype = ctypes.c_int
        rc = self._lib.crispasr_mic_start(self._handle)
        if rc != 0:
            raise RuntimeError(f"mic_start failed (rc={rc})")
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._lib.crispasr_mic_stop.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_mic_stop.restype = ctypes.c_int
        self._lib.crispasr_mic_stop(self._handle)
        self._started = False

    def close(self) -> None:
        if not self._handle:
            return
        self.stop()
        self._lib.crispasr_mic_close.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_mic_close.restype = None
        self._lib.crispasr_mic_close(self._handle)
        self._handle = None
        self._cb_holder = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def mic_default_device_name(*, lib_path: Optional[str] = None) -> str:
    """Return the human-readable name of the default capture device,
    or empty string if no input device is available."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_mic_default_device_name"):
        return ""
    fn = lib.crispasr_mic_default_device_name
    fn.argtypes = []
    fn.restype = ctypes.c_char_p
    s = fn()
    return s.decode("utf-8") if s else ""


def detect_language_pcm(
    pcm: np.ndarray,
    *,
    method: int = LidMethod.WHISPER,
    model_path: str,
    n_threads: int = 4,
    use_gpu: bool = False,
    gpu_device: int = 0,
    flash_attn: bool = True,
    lib_path: Optional[str] = None,
) -> LidResult:
    """Run language identification on a 16 kHz mono float PCM buffer.

    ``model_path`` must point to a concrete file (auto-download is the
    caller's responsibility — Python users typically cache the model
    themselves). Returns an empty :class:`LidResult` (``lang_code == ""``)
    on failure.
    """
    if pcm is None or len(pcm) == 0 or not model_path:
        return LidResult(lang_code="", confidence=-1.0)

    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_detect_language_pcm"):
        raise RuntimeError(
            "crispasr_detect_language_pcm not in loaded library — rebuild "
            "CrispASR 0.4.6+ to use LID from the Python binding."
        )
    lib.crispasr_detect_language_pcm.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.crispasr_detect_language_pcm.restype = ctypes.c_int

    pcm_np = np.asarray(pcm, dtype=np.float32)
    buf = ctypes.create_string_buffer(16)
    conf = ctypes.c_float(-1.0)
    rc = lib.crispasr_detect_language_pcm(
        pcm_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(len(pcm_np)), int(method),
        model_path.encode("utf-8"),
        int(n_threads), 1 if use_gpu else 0, int(gpu_device),
        1 if flash_attn else 0,
        buf, 16, ctypes.byref(conf),
    )
    if rc == 0:
        return LidResult(lang_code=buf.value.decode("utf-8"), confidence=conf.value)
    return LidResult(lang_code="", confidence=-1.0)


def diarize_segments(
    segs: List[DiarizeSegment],
    left: np.ndarray,
    *,
    right: Optional[np.ndarray] = None,
    is_stereo: bool = False,
    method: int = DiarizeMethod.VAD_TURNS,
    pyannote_model_path: Optional[str] = None,
    n_threads: int = 4,
    slice_t0: float = 0.0,
    foxnose_embedder_path: Optional[str] = None,
    min_speakers: int = 0,
    max_speakers: int = 0,
    num_speakers: int = 0,
    lib_path: Optional[str] = None,
) -> bool:
    """Assign a speaker index to each of ``segs``, mutating in place.

    Five methods — see :class:`DiarizeMethod`. ``left`` is mono PCM for
    mono-only methods, otherwise the left channel of a stereo pair.
    All PCM is 16 kHz float32. Returns ``True`` on success; only
    Model-backed methods can fail when their model cannot be loaded.
    """
    ok, _ = _diarize_segments_impl(
        segs, left, right=right, is_stereo=is_stereo, method=method,
        pyannote_model_path=pyannote_model_path, n_threads=n_threads,
        slice_t0=slice_t0, foxnose_embedder_path=foxnose_embedder_path,
        min_speakers=min_speakers, max_speakers=max_speakers,
        num_speakers=num_speakers, lib_path=lib_path, want_turns=False,
    )
    return ok


def diarize_segments_with_turns(
    segs: List[DiarizeSegment],
    left: np.ndarray,
    *,
    right: Optional[np.ndarray] = None,
    is_stereo: bool = False,
    method: int = DiarizeMethod.VAD_TURNS,
    pyannote_model_path: Optional[str] = None,
    n_threads: int = 4,
    slice_t0: float = 0.0,
    foxnose_embedder_path: Optional[str] = None,
    min_speakers: int = 0,
    max_speakers: int = 0,
    num_speakers: int = 0,
    lib_path: Optional[str] = None,
) -> Tuple[bool, List[DiarizeTurn]]:
    """Label ``segs`` and return audio-derived FoxNose speaker turns.

    The returned timestamps are seconds on the same absolute timeline as the
    input segments. Non-FoxNose methods return an empty turn list.
    """
    return _diarize_segments_impl(
        segs, left, right=right, is_stereo=is_stereo, method=method,
        pyannote_model_path=pyannote_model_path, n_threads=n_threads,
        slice_t0=slice_t0, foxnose_embedder_path=foxnose_embedder_path,
        min_speakers=min_speakers, max_speakers=max_speakers,
        num_speakers=num_speakers, lib_path=lib_path, want_turns=True,
    )


def _diarize_segments_impl(
    segs: List[DiarizeSegment],
    left: np.ndarray,
    *,
    right: Optional[np.ndarray],
    is_stereo: bool,
    method: int,
    pyannote_model_path: Optional[str],
    n_threads: int,
    slice_t0: float,
    foxnose_embedder_path: Optional[str],
    min_speakers: int,
    max_speakers: int,
    num_speakers: int,
    lib_path: Optional[str],
    want_turns: bool,
) -> Tuple[bool, List[DiarizeTurn]]:
    if not segs or left is None or len(left) == 0:
        return True, []

    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_diarize_segments_abi"):
        raise RuntimeError(
            "crispasr_diarize_segments_abi not in loaded library — rebuild "
            "CrispASR 0.4.5+ to use diarization from the Python binding."
        )
    lib.crispasr_diarize_segments_abi.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_void_p,
    ]
    lib.crispasr_diarize_segments_abi.restype = ctypes.c_int

    left_np = np.asarray(left, dtype=np.float32)
    left_ptr = left_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if is_stereo and right is not None:
        right_np = np.asarray(right, dtype=np.float32)
        right_ptr = right_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        right_ptr = left_ptr

    seg_array = (_DiarizeSegAbi * len(segs))()
    for i, s in enumerate(segs):
        seg_array[i].t0_cs = int(round(s.t0 * 100))
        seg_array[i].t1_cs = int(round(s.t1 * 100))
        seg_array[i].speaker = s.speaker
        seg_array[i]._pad = 0

    opts = _DiarizeOptsAbi(
        method=int(method),
        n_threads=int(n_threads),
        slice_t0_cs=int(round(slice_t0 * 100)),
        pyannote_model_path=(pyannote_model_path.encode("utf-8")
                             if pyannote_model_path else None),
        foxnose_embedder_path=(foxnose_embedder_path.encode("utf-8")
                               if foxnose_embedder_path else None),
        min_speakers=int(min_speakers),
        max_speakers=int(max_speakers),
        num_speakers=int(num_speakers),
        _pad2=0,
    )

    turns: List[DiarizeTurn] = []
    if want_turns:
        if not hasattr(lib, "crispasr_diarize_segments_turns_abi"):
            raise RuntimeError(
                "crispasr_diarize_segments_turns_abi not in loaded library — "
                "rebuild CrispASR 0.8.30+ to receive FoxNose turns."
            )
        fn = lib.crispasr_diarize_segments_turns_abi
        fn.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
        ]
        fn.restype = ctypes.c_int
        turn_cap = max(1, (len(left_np) + 7999) // 8000 + len(segs) + 16)
        for attempt in range(2):
            turn_array = (_DiarizeTurnAbi * turn_cap)()
            n_turns = ctypes.c_int32()
            rc = fn(
                left_ptr, right_ptr, int(len(left_np)), 1 if is_stereo else 0,
                ctypes.byref(seg_array), len(segs), ctypes.byref(opts),
                ctypes.byref(turn_array), turn_cap, ctypes.byref(n_turns),
            )
            if rc == 2 and attempt == 0 and n_turns.value > turn_cap:
                turn_cap = n_turns.value
                continue
            if rc == 0:
                turns = [
                    DiarizeTurn(
                        t0=turn_array[i].t0_cs / 100.0,
                        t1=turn_array[i].t1_cs / 100.0,
                        speaker=int(turn_array[i].speaker),
                    )
                    for i in range(min(n_turns.value, turn_cap))
                ]
            break
    else:
        rc = lib.crispasr_diarize_segments_abi(
            left_ptr, right_ptr, int(len(left_np)), 1 if is_stereo else 0,
            ctypes.byref(seg_array), len(segs), ctypes.byref(opts),
        )
    if rc == 0:
        for i, s in enumerate(segs):
            s.speaker = int(seg_array[i].speaker)
    return rc == 0, turns


class Session:
    """Backend-agnostic transcription session over any CrispASR-supported GGUF.

    The backend is auto-detected from the file's `general.architecture`
    metadata. `Session.available_backends()` lists which backends the
    bundled libcrispasr was actually compiled with — a model whose
    backend isn't in that list will fail to open.

    Usage:
        with crispasr.Session("model.gguf") as s:
            print(f"backend: {s.backend}")
            for seg in s.transcribe(pcm_f32):
                print(f"[{seg.start:.1f}-{seg.end:.1f}s] {seg.text}")
    """

    # Progress callback signature: void(int processed, int total, void* ud).
    # Held on the instance while a chunked call runs so it isn't GC'd.
    _PROGRESS_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)

    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 n_threads: int = 4, backend: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._progress_cb_holder = None
        self._setup_session_signatures()

        path_bytes = model_path.encode("utf-8")
        if backend:
            self._handle = self._lib.crispasr_session_open_explicit(
                path_bytes, backend.encode("utf-8"), n_threads
            )
        else:
            self._handle = self._lib.crispasr_session_open(path_bytes, n_threads)

        if not self._handle:
            avail = Session.available_backends(lib_path=lib_path)
            raise RuntimeError(
                f"Failed to open {model_path!r} — backend not supported. "
                f"libcrispasr was built with: {avail}"
            )
        be = self._lib.crispasr_session_backend(self._handle)
        self.backend = be.decode("utf-8") if be else ""
        self._n_threads = int(n_threads)

    def _setup_session_signatures(self):
        lib = self._lib
        # Missing symbol ⇒ pre-0.4.0 dylib.
        for name in (
            "crispasr_session_open", "crispasr_session_transcribe",
            "crispasr_session_available_backends", "crispasr_session_close",
        ):
            if not hasattr(lib, name):
                raise RuntimeError(
                    "Unified session API not found in loaded library — "
                    "rebuild CrispASR with 0.4.0+ helpers."
                )

        lib.crispasr_session_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_open.restype = ctypes.c_void_p
        lib.crispasr_session_open_explicit.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
        ]
        lib.crispasr_session_open_explicit.restype = ctypes.c_void_p
        lib.crispasr_session_backend.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_backend.restype = ctypes.c_char_p
        # Acoustic detected language (Whisper). Probe with hasattr — older
        # libcrispasr builds don't export it (detected_language() -> "unknown").
        if hasattr(lib, "crispasr_session_detected_language"):
            lib.crispasr_session_detected_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
            lib.crispasr_session_detected_language.restype = ctypes.c_int
        lib.crispasr_session_available_backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_available_backends.restype = ctypes.c_int
        # 2026-07-08: CTC vocabulary access (Omni CTC backend). n_vocab is the
        # piece count; token_text maps an id to its raw piece. hasattr-guarded
        # so a binding loaded against an older dylib still works.
        if hasattr(lib, "crispasr_session_token_text"):
            lib.crispasr_session_n_vocab.argtypes = [ctypes.c_void_p]
            lib.crispasr_session_n_vocab.restype = ctypes.c_int
            lib.crispasr_session_token_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_session_token_text.restype = ctypes.c_char_p
        lib.crispasr_session_transcribe.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.crispasr_session_transcribe.restype = ctypes.c_void_p
        # 0.4.9+: language-aware session transcribe. Backends that
        # accept a source-language hint (whisper / canary / cohere /
        # voxtral / voxtral4b) honour it; others ignore.
        if hasattr(lib, "crispasr_session_transcribe_lang"):
            lib.crispasr_session_transcribe_lang.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_char_p,
            ]
            lib.crispasr_session_transcribe_lang.restype = ctypes.c_void_p
        # 0.10.3+ (issue #208): chunked-encode transcribe — forces the
        # Parakeet backend through its bounded overlapping-window long-form
        # path (inert on other backends). hasattr-guarded for old dylibs.
        if hasattr(lib, "crispasr_session_transcribe_chunked_lang"):
            lib.crispasr_session_transcribe_chunked_lang.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
            ]
            lib.crispasr_session_transcribe_chunked_lang.restype = ctypes.c_void_p
        # 0.10.3+ (issue #208): long-form progress. crispasr_get_progress()
        # polls 0..100 (-1 idle) and now tracks chunked windows; the callback
        # setter fires cb(processed, total, ud) once per finished window.
        if hasattr(lib, "crispasr_get_progress"):
            lib.crispasr_get_progress.argtypes = []
            lib.crispasr_get_progress.restype = ctypes.c_int
        if hasattr(lib, "crispasr_session_set_progress_callback"):
            lib.crispasr_session_set_progress_callback.argtypes = [
                ctypes.c_void_p, Session._PROGRESS_CB_TYPE, ctypes.c_void_p,
            ]
            lib.crispasr_session_set_progress_callback.restype = None
        # 0.4.3+: VAD-driven session transcribe. hasattr-guarded so a
        # binding loaded against an older dylib still works for non-VAD
        # calls.
        if hasattr(lib, "crispasr_session_transcribe_vad"):
            lib.crispasr_session_transcribe_vad.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p,
            ]
            lib.crispasr_session_transcribe_vad.restype = ctypes.c_void_p
        # 0.4.9+: language-aware VAD transcribe (same hint semantics
        # as _lang above).
        if hasattr(lib, "crispasr_session_transcribe_vad_lang"):
            lib.crispasr_session_transcribe_vad_lang.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p,
                ctypes.c_char_p,
            ]
            lib.crispasr_session_transcribe_vad_lang.restype = ctypes.c_void_p
        # 0.4.5+: shared speaker diarization. Same hasattr guard.
        if hasattr(lib, "crispasr_diarize_segments_abi"):
            lib.crispasr_diarize_segments_abi.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p,
                ctypes.c_int32, ctypes.c_void_p,
            ]
            lib.crispasr_diarize_segments_abi.restype = ctypes.c_int
        # 0.4.6+: shared language identification.
        if hasattr(lib, "crispasr_detect_language_pcm"):
            lib.crispasr_detect_language_pcm.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
                ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.crispasr_detect_language_pcm.restype = ctypes.c_int
        # 0.4.7+: shared CTC / forced-aligner word timings.
        if hasattr(lib, "crispasr_align_words_abi"):
            lib.crispasr_align_words_abi.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                ctypes.c_int64, ctypes.c_int32,
            ]
            lib.crispasr_align_words_abi.restype = ctypes.c_void_p
            lib.crispasr_align_result_n_words.argtypes = [ctypes.c_void_p]
            lib.crispasr_align_result_n_words.restype = ctypes.c_int
            lib.crispasr_align_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_align_result_word_text.restype = ctypes.c_char_p
            lib.crispasr_align_result_word_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_align_result_word_t0.restype = ctypes.c_int64
            lib.crispasr_align_result_word_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_align_result_word_t1.restype = ctypes.c_int64
            lib.crispasr_align_result_free.argtypes = [ctypes.c_void_p]
            lib.crispasr_align_result_free.restype = None
        lib.crispasr_session_result_n_segments.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_result_n_segments.restype = ctypes.c_int
        lib.crispasr_session_result_segment_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_text.restype = ctypes.c_char_p
        lib.crispasr_session_result_segment_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_t0.restype = ctypes.c_int64
        lib.crispasr_session_result_segment_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_segment_t1.restype = ctypes.c_int64
        # segment_speaker was added 2026-07-27 (#300). Older libcrispasr builds
        # don't export it — probe with hasattr at the call site and fall back to
        # "" so a new wheel keeps working against an older system library.
        if hasattr(lib, "crispasr_session_result_segment_speaker"):
            lib.crispasr_session_result_segment_speaker.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_session_result_segment_speaker.restype = ctypes.c_char_p
        lib.crispasr_session_result_n_words.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_result_n_words.restype = ctypes.c_int
        lib.crispasr_session_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_text.restype = ctypes.c_char_p
        lib.crispasr_session_result_word_t0.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_t0.restype = ctypes.c_int64
        lib.crispasr_session_result_word_t1.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.crispasr_session_result_word_t1.restype = ctypes.c_int64
        # word_p was added 2026-05-02. Older libcrispasr builds don't export it
        # — probe with hasattr below and fall back to 1.0 when missing.
        if hasattr(lib, "crispasr_session_result_word_p"):
            lib.crispasr_session_result_word_p.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            lib.crispasr_session_result_word_p.restype = ctypes.c_float
        # Per-segment no_speech_prob (Whisper). Probe with hasattr like word_p —
        # older libcrispasr builds don't export it (fall back to the -1.0 sentinel).
        if hasattr(lib, "crispasr_session_result_segment_no_speech_prob"):
            lib.crispasr_session_result_segment_no_speech_prob.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_session_result_segment_no_speech_prob.restype = ctypes.c_float
        # 0.5.13: per-word top-N alternative candidates.
        if hasattr(lib, "crispasr_session_result_word_n_alts"):
            lib.crispasr_session_result_word_n_alts.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            lib.crispasr_session_result_word_n_alts.restype = ctypes.c_int
        if hasattr(lib, "crispasr_session_result_word_alt_p"):
            lib.crispasr_session_result_word_alt_p.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.crispasr_session_result_word_alt_p.restype = ctypes.c_float
        # 2026-07-07: per-frame CTC logits for backends with a dense CTC grid
        # (Omni CTC, wav2vec2/hubert/data2vec, canary-ctc), opted in via
        # crispasr_session_set_return_logits. Frame-major (raw pre-softmax for
        # Omni & wav2vec2, log-probabilities for canary-ctc);
        # crispasr_session_result_logits returns NULL when none were captured.
        # hasattr-guarded so a binding loaded against an older dylib still works.
        if hasattr(lib, "crispasr_session_result_logits"):
            lib.crispasr_session_result_n_logit_frames.argtypes = [ctypes.c_void_p]
            lib.crispasr_session_result_n_logit_frames.restype = ctypes.c_int
            lib.crispasr_session_result_n_logit_vocab.argtypes = [ctypes.c_void_p]
            lib.crispasr_session_result_n_logit_vocab.restype = ctypes.c_int
            lib.crispasr_session_result_logits.argtypes = [ctypes.c_void_p]
            lib.crispasr_session_result_logits.restype = ctypes.POINTER(ctypes.c_float)
        lib.crispasr_session_result_free.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_result_free.restype = None
        lib.crispasr_session_close.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_close.restype = None
        # 0.6.1: session_open_with_params.
        if hasattr(lib, "crispasr_session_open_with_params"):
            lib.crispasr_session_open_with_params.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p,
            ]
            lib.crispasr_session_open_with_params.restype = ctypes.c_void_p

    @staticmethod
    def available_backends(lib_path: Optional[str] = None) -> List[str]:
        """List the backend names the loaded CrispASR library was built with."""
        lib = ctypes.CDLL(lib_path or _find_lib())
        if not hasattr(lib, "crispasr_session_available_backends"):
            return []
        lib.crispasr_session_available_backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_available_backends.restype = ctypes.c_int
        buf = ctypes.create_string_buffer(256)
        needed = lib.crispasr_session_available_backends(buf, len(buf))
        if needed >= len(buf):
            buf = ctypes.create_string_buffer(needed + 1)
            lib.crispasr_session_available_backends(buf, len(buf))
        csv = buf.value.decode("utf-8")
        return [s.strip() for s in csv.split(",") if s.strip()]

    def detected_language(self) -> str:
        """The acoustic language Whisper detected on the last transcribe, as an
        ISO-639-1 code (e.g. "en"). Whisper-only; other backends return the
        session's source-language hint, or "unknown" (also on libcrispasr
        builds predating the accessor)."""
        if not hasattr(self._lib, "crispasr_session_detected_language"):
            return "unknown"
        buf = ctypes.create_string_buffer(32)
        self._lib.crispasr_session_detected_language(self._handle, buf, len(buf))
        return buf.value.decode("utf-8") or "unknown"

    def transcribe(
        self, pcm: np.ndarray, sample_rate: int = 16000,
        *,
        language: Optional[str] = None,
    ) -> List[SessionSegment]:
        """Transcribe 16 kHz mono float32 PCM. Dispatches via crispasr_session.

        ``language`` is an optional ISO 639-1 code ("en", "de", "ja", ...).
        Backends that accept a source-language hint (whisper, canary,
        cohere, voxtral, voxtral4b) honour it; others ignore silently.
        ``None`` preserves each backend's historical default.
        """
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
        pcm = np.asarray(pcm, dtype=np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if language and hasattr(self._lib, "crispasr_session_transcribe_lang"):
            res = self._lib.crispasr_session_transcribe_lang(
                self._handle, samples_ptr, len(pcm), language.encode("utf-8"))
        else:
            res = self._lib.crispasr_session_transcribe(self._handle, samples_ptr, len(pcm))
        if not res:
            self.set_return_logits(False)
            raise RuntimeError(f"crispasr_session_transcribe failed for backend {self.backend!r}")

        try:
            n_seg = self._lib.crispasr_session_result_n_segments(res)
            out: List[SessionSegment] = []
            for i in range(n_seg):
                t = self._lib.crispasr_session_result_segment_text(res, i)
                text = t.decode("utf-8") if t else ""
                t0 = self._lib.crispasr_session_result_segment_t0(res, i) / 100.0
                t1 = self._lib.crispasr_session_result_segment_t1(res, i) / 100.0
                wn = self._lib.crispasr_session_result_n_words(res, i)
                has_nsp = hasattr(self._lib, "crispasr_session_result_segment_no_speech_prob")
                nsp = self._lib.crispasr_session_result_segment_no_speech_prob(res, i) if has_nsp else -1.0
                spk_b = (self._lib.crispasr_session_result_segment_speaker(res, i)
                         if hasattr(self._lib, "crispasr_session_result_segment_speaker") else None)
                spk = spk_b.decode("utf-8") if spk_b else ""
                words: List[SessionWord] = []
                has_word_p = hasattr(self._lib, "crispasr_session_result_word_p")
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    raw_p = self._lib.crispasr_session_result_word_p(res, i, j) if has_word_p else 1.0
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                        # -1.0 from C means "no per-word p for this backend";
                        # surface 1.0 so callers can render uniformly.
                        confidence=1.0 if raw_p < 0 else raw_p,
                    ))
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words, no_speech_prob=nsp, speaker=spk))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

    def transcribe_chunked(
        self, pcm: np.ndarray,
        chunk_seconds: int = 0, overlap_seconds: int = -1,
        *,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> List[SessionSegment]:
        """Chunked-encode transcribe (issue #208).

        Forces the Parakeet backend through its bounded overlapping-window
        long-form path regardless of length, so long files transcribe in
        bounded time and don't drop sections. ``chunk_seconds <= 0`` keeps the
        per-model default window; ``overlap_seconds < 0`` keeps the default
        overlap. Inert (== :meth:`transcribe`) on non-Parakeet backends.

        ``progress(processed_samples, total_samples)`` — if given — fires once
        per finished window on the calling thread. You can also poll
        :meth:`get_progress` (0..100) from another thread.
        """
        if not hasattr(self._lib, "crispasr_session_transcribe_chunked_lang"):
            return self.transcribe(pcm, sample_rate, language=language)  # old dylib
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
        pcm = np.asarray(pcm, dtype=np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lang_c = language.encode("utf-8") if language else None

        registered = False
        if progress is not None and hasattr(self._lib, "crispasr_session_set_progress_callback"):
            def _trampoline(processed, total, _ud):
                try:
                    progress(processed, total)
                except Exception as e:  # never let a Python exception cross the FFI boundary
                    import sys
                    sys.stderr.write(f"crispasr progress callback raised: {e}\n")
            self._progress_cb_holder = Session._PROGRESS_CB_TYPE(_trampoline)
            self._lib.crispasr_session_set_progress_callback(self._handle, self._progress_cb_holder, None)
            registered = True

        try:
            res = self._lib.crispasr_session_transcribe_chunked_lang(
                self._handle, samples_ptr, len(pcm), int(chunk_seconds), int(overlap_seconds), lang_c)
        finally:
            if registered:
                self._lib.crispasr_session_set_progress_callback(self._handle, None, None)
                self._progress_cb_holder = None
        if not res:
            raise RuntimeError(f"crispasr_session_transcribe_chunked failed for backend {self.backend!r}")

        try:
            n_seg = self._lib.crispasr_session_result_n_segments(res)
            out: List[SessionSegment] = []
            has_word_p = hasattr(self._lib, "crispasr_session_result_word_p")
            for i in range(n_seg):
                t = self._lib.crispasr_session_result_segment_text(res, i)
                text = t.decode("utf-8") if t else ""
                t0 = self._lib.crispasr_session_result_segment_t0(res, i) / 100.0
                t1 = self._lib.crispasr_session_result_segment_t1(res, i) / 100.0
                wn = self._lib.crispasr_session_result_n_words(res, i)
                has_nsp = hasattr(self._lib, "crispasr_session_result_segment_no_speech_prob")
                nsp = self._lib.crispasr_session_result_segment_no_speech_prob(res, i) if has_nsp else -1.0
                spk_b = (self._lib.crispasr_session_result_segment_speaker(res, i)
                         if hasattr(self._lib, "crispasr_session_result_segment_speaker") else None)
                spk = spk_b.decode("utf-8") if spk_b else ""
                words: List[SessionWord] = []
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    raw_p = self._lib.crispasr_session_result_word_p(res, i, j) if has_word_p else 1.0
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                        confidence=1.0 if raw_p < 0 else raw_p,
                    ))
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words, no_speech_prob=nsp, speaker=spk))
            return out
        finally:
            self.set_return_logits(False)
            self._lib.crispasr_session_result_free(res)

    def get_progress(self) -> int:
        """Poll long-form transcription progress: 0..100, or -1 when idle.

        Updated in lockstep with :meth:`transcribe_chunked` windows (issue
        #208), so a UI thread can render a progress bar without a callback.
        Returns -1 if the loaded libcrispasr predates the poll API.
        """
        if not hasattr(self._lib, "crispasr_get_progress"):
            return -1
        return int(self._lib.crispasr_get_progress())

    def transcribe_vad(
        self,
        pcm: np.ndarray,
        vad_model_path: str,
        *,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        chunk_seconds: int = 30,
        n_threads: int = 4,
        language: Optional[str] = None,
    ) -> List[SessionSegment]:
        """Transcribe with Silero VAD segmentation + crispasr-style stitching.

        Runs VAD on ``pcm``, merges short / overlong speech slices into usable
        chunks, stitches them into a single buffer with 0.1s silence gaps,
        calls the backend once, then remaps segment + word timestamps back to
        original-audio positions.

        ``vad_model_path`` must point to a Silero GGUF on disk. If it fails
        to load, this falls back to a plain :meth:`transcribe` call.

        Compared to the fixed-chunk CLI loop, one stitched call preserves
        cross-segment context (no boundary artefacts like words split across
        chunks), which matters for O(T²) backends such as parakeet /
        cohere / canary.
        """
        if not hasattr(self._lib, "crispasr_session_transcribe_vad"):
            raise RuntimeError(
                "crispasr_session_transcribe_vad not in loaded library — "
                "rebuild CrispASR 0.4.3+ or call transcribe() instead."
            )
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
        pcm = np.asarray(pcm, dtype=np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # ABI struct layout must match crispasr_vad_abi_opts (crispasr_c_api.cpp):
        # float + 5 x int32.
        class _VadAbiOpts(ctypes.Structure):
            _fields_ = [
                ("threshold", ctypes.c_float),
                ("min_speech_duration_ms", ctypes.c_int32),
                ("min_silence_duration_ms", ctypes.c_int32),
                ("speech_pad_ms", ctypes.c_int32),
                ("chunk_seconds", ctypes.c_int32),
                ("n_threads", ctypes.c_int32),
            ]
        opts = _VadAbiOpts(
            float(threshold),
            int(min_speech_duration_ms),
            int(min_silence_duration_ms),
            int(speech_pad_ms),
            int(chunk_seconds),
            int(n_threads),
        )

        if language and hasattr(self._lib, "crispasr_session_transcribe_vad_lang"):
            res = self._lib.crispasr_session_transcribe_vad_lang(
                self._handle,
                samples_ptr,
                len(pcm),
                16000,
                vad_model_path.encode("utf-8"),
                ctypes.byref(opts),
                language.encode("utf-8"),
            )
        else:
            res = self._lib.crispasr_session_transcribe_vad(
                self._handle,
                samples_ptr,
                len(pcm),
                16000,
                vad_model_path.encode("utf-8"),
                ctypes.byref(opts),
            )
        if not res:
            raise RuntimeError(
                f"crispasr_session_transcribe_vad failed for backend {self.backend!r}"
            )

        try:
            n_seg = self._lib.crispasr_session_result_n_segments(res)
            out: List[SessionSegment] = []
            for i in range(n_seg):
                t = self._lib.crispasr_session_result_segment_text(res, i)
                text = t.decode("utf-8") if t else ""
                t0 = self._lib.crispasr_session_result_segment_t0(res, i) / 100.0
                t1 = self._lib.crispasr_session_result_segment_t1(res, i) / 100.0
                wn = self._lib.crispasr_session_result_n_words(res, i)
                has_nsp = hasattr(self._lib, "crispasr_session_result_segment_no_speech_prob")
                nsp = self._lib.crispasr_session_result_segment_no_speech_prob(res, i) if has_nsp else -1.0
                spk_b = (self._lib.crispasr_session_result_segment_speaker(res, i)
                         if hasattr(self._lib, "crispasr_session_result_segment_speaker") else None)
                spk = spk_b.decode("utf-8") if spk_b else ""
                words: List[SessionWord] = []
                has_word_p = hasattr(self._lib, "crispasr_session_result_word_p")
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    raw_p = self._lib.crispasr_session_result_word_p(res, i, j) if has_word_p else 1.0
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                        # -1.0 from C means "no per-word p for this backend";
                        # surface 1.0 so callers can render uniformly.
                        confidence=1.0 if raw_p < 0 else raw_p,
                    ))
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words, no_speech_prob=nsp, speaker=spk))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

    def transcribe_with_logits(
        self, pcm: np.ndarray, sample_rate: int = 16000,
        *,
        language: Optional[str] = None,
    ) -> Tuple[List[SessionSegment], Optional[np.ndarray]]:
        """Transcribe and also return the per-frame CTC logits.

        Enables logit capture for this call (no prior :meth:`set_return_logits`
        needed), then returns ``(segments, logits)``. ``logits`` is a 2D
        float32 numpy array of shape ``(n_frames, n_vocab)`` — frame-major, i.e.
        ``logits[t, v]`` is the score for vocabulary entry ``v`` at encoder
        frame ``t``. The Omni CTC and wav2vec2/hubert/data2vec grids are raw
        logits (pre-softmax); the canary-ctc grid is log-probabilities. It is
        ``None`` for backends that don't produce a dense CTC grid, when the
        transcript is empty, or on dylibs predating the accessor.

        ``sample_rate`` / ``language`` behave exactly as in :meth:`transcribe`.
        """
        segs = []  # type: List[SessionSegment]
        if not hasattr(self._lib, "crispasr_session_result_logits"):
            return self.transcribe(pcm, sample_rate, language=language), None
        if len(pcm) == 0:
            return segs, None

        self.set_return_logits(True)
        if sample_rate != 16000:
            ratio = 16000 / sample_rate
            new_len = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, new_len)
            pcm = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.float32)
        pcm = np.asarray(pcm, dtype=np.float32)
        samples_ptr = pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if language and hasattr(self._lib, "crispasr_session_transcribe_lang"):
            res = self._lib.crispasr_session_transcribe_lang(
                self._handle, samples_ptr, len(pcm), language.encode("utf-8"))
        else:
            res = self._lib.crispasr_session_transcribe(self._handle, samples_ptr, len(pcm))
        if not res:
            raise RuntimeError(f"crispasr_session_transcribe failed for backend {self.backend!r}")

        try:
            n_seg = self._lib.crispasr_session_result_n_segments(res)
            has_word_p = hasattr(self._lib, "crispasr_session_result_word_p")
            for i in range(n_seg):
                t = self._lib.crispasr_session_result_segment_text(res, i)
                text = t.decode("utf-8") if t else ""
                t0 = self._lib.crispasr_session_result_segment_t0(res, i) / 100.0
                t1 = self._lib.crispasr_session_result_segment_t1(res, i) / 100.0
                wn = self._lib.crispasr_session_result_n_words(res, i)
                has_nsp = hasattr(self._lib, "crispasr_session_result_segment_no_speech_prob")
                nsp = self._lib.crispasr_session_result_segment_no_speech_prob(res, i) if has_nsp else -1.0
                spk_b = (self._lib.crispasr_session_result_segment_speaker(res, i)
                         if hasattr(self._lib, "crispasr_session_result_segment_speaker") else None)
                spk = spk_b.decode("utf-8") if spk_b else ""
                words: List[SessionWord] = []
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    raw_p = self._lib.crispasr_session_result_word_p(res, i, j) if has_word_p else 1.0
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                        confidence=1.0 if raw_p < 0 else raw_p,
                    ))
                segs.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words, no_speech_prob=nsp, speaker=spk))

            # Lift out the CTC logits (if any) before the handle is freed.
            n_frames = self._lib.crispasr_session_result_n_logit_frames(res)
            n_vocab = self._lib.crispasr_session_result_n_logit_vocab(res)
            lp = self._lib.crispasr_session_result_logits(res)
            logits: Optional[np.ndarray] = None
            if n_frames > 0 and n_vocab > 0 and lp:
                # Buffer is owned by the result — copy before the free below.
                logits = np.ctypeslib.as_array(
                    lp, shape=(n_frames, n_vocab)).copy()
            return segs, logits
        finally:
            self._lib.crispasr_session_result_free(res)

    def ctc_vocab(self) -> Optional[List[str]]:
        """Return the Omni CTC vocabulary as raw pieces, indexed by token id.

        ``vocab[id]`` is the raw piece for token ``id`` — word-boundary marker
        intact (the v2 Omni vocab uses a literal space, v1 uses U+2581) — so a
        consumer can detokenize a greedy CTC decode over the grid from
        :meth:`transcribe_with_logits`. Returns ``None`` for backends that
        don't expose a CTC vocab or on dylibs predating the accessor.
        """
        if not hasattr(self._lib, "crispasr_session_token_text"):
            return None
        n = self._lib.crispasr_session_n_vocab(self._handle)
        if n <= 0:
            return None
        vocab: List[str] = []
        for i in range(n):
            p = self._lib.crispasr_session_token_text(self._handle, i)
            vocab.append(p.decode("utf-8") if p else "")
        return vocab

    # ---------------------------------------------------------------------
    # TTS synthesis (vibevoice, qwen3-tts, miotts, moss-tts, moss-tts-local, confucius4-tts, omnivoice, kokoro, orpheus, chatterbox, outetts, indextts, voxcpm2, csm, dia, zonos-tts, bark, speecht5, parler-tts, pocket-tts, kugelaudio, tada, lfm2-audio, dots-tts, fireredtts3)
    # ---------------------------------------------------------------------

    def set_codec_path(self, path: str) -> None:
        """Load a separate codec GGUF.

        Required for qwen3-tts (12 Hz tokenizer), moss-tts (1.6B transformer
        RVQ codec) and orpheus (SNAC codec); no-op for other backends.
        """
        if not hasattr(self._lib, "crispasr_session_set_codec_path"):
            raise RuntimeError("TTS API not present in this libcrispasr build")
        self._lib.crispasr_session_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_codec_path.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_codec_path(self._handle, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_codec_path failed (rc={rc}) for backend {self.backend!r}")

    def set_pcm_sample_rate(self, sample_rate: int) -> None:
        """Declare the sample rate of PCM passed to the next audio call."""
        if not hasattr(self._lib, "crispasr_session_set_pcm_sample_rate"):
            raise RuntimeError("PCM sample-rate API not present in this libcrispasr build")
        self._lib.crispasr_session_set_pcm_sample_rate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_pcm_sample_rate.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_pcm_sample_rate(self._handle, int(sample_rate))
        if rc != 0:
            raise RuntimeError(f"set_pcm_sample_rate failed (rc={rc})")

    def set_parakeet_att_context(self, left: int, right: int) -> None:
        """Set parakeet/canary local-attention window (issue #257).

        Encoder frames (~80 ms each) — the equivalent of NeMo's
        ``model.change_attention_model("rel_pos_local_attn", [left, right])``.
        Bounds long-audio encoder memory to ``O(T * window)`` instead of
        ``O(T^2)``, so long clips fit in limited VRAM. Negative values select
        full (global) attention; the default (unset) keeps the model's own
        window. No-op for non-parakeet backends.
        """
        if not hasattr(self._lib, "crispasr_session_set_parakeet_att_context"):
            raise RuntimeError("crispasr_session_set_parakeet_att_context not present in this libcrispasr build")
        self._lib.crispasr_session_set_parakeet_att_context.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        self._lib.crispasr_session_set_parakeet_att_context.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_parakeet_att_context(self._handle, int(left), int(right))
        if rc != 0:
            raise RuntimeError(f"set_parakeet_att_context failed (rc={rc}) for backend {self.backend!r}")

    def set_voice(self, path: str, ref_text: Optional[str] = None) -> None:
        """Load a voice prompt: a baked GGUF voice pack OR a *.wav reference.

        For qwen3-tts a *.wav reference requires ``ref_text`` (the
        transcription of the reference audio).

        For orpheus voice selection is BY NAME — use
        :meth:`set_speaker_name` instead of this method.

        For speecht5, pass a raw float32 binary file containing a 512-d
        x-vector (e.g. from Matthijs/cmu-arctic-xvectors).
        """
        if not hasattr(self._lib, "crispasr_session_set_voice"):
            raise RuntimeError("TTS API not present in this libcrispasr build")
        self._lib.crispasr_session_set_voice.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_voice.restype = ctypes.c_int
        rt = ref_text.encode("utf-8") if ref_text else None
        rc = self._lib.crispasr_session_set_voice(self._handle, path.encode("utf-8"), rt)
        if rc != 0:
            raise RuntimeError(f"set_voice failed (rc={rc}) for backend {self.backend!r}")

    def set_speaker_name(self, name: str) -> None:
        """Select a fixed/preset speaker by NAME (orpheus).

        Orpheus bakes speaker names into the LM training data as the
        literal ``f"{name}: {text}"`` prompt prefix — there is no
        embedding-table dispatch. Names are e.g. ``"tara"``, ``"leo"``,
        ``"leah"`` for the canopylabs English finetune; the
        Kartoffel_Orpheus DE finetunes use ``"Anton"``, ``"Sophie"``,
        etc. Enumerate available names with :meth:`speakers`.

        Raises if the active backend has no preset-speaker contract or
        the name is not in the GGUF metadata.
        """
        if not hasattr(self._lib, "crispasr_session_set_speaker_name"):
            raise RuntimeError("set_speaker_name API not present in this libcrispasr build")
        self._lib.crispasr_session_set_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_speaker_name.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_speaker_name(self._handle, name.encode("utf-8"))
        if rc == -2:
            raise ValueError(f"unknown speaker {name!r} for backend {self.backend!r}; "
                             f"call .speakers() to enumerate")
        if rc == -3:
            raise RuntimeError(f"backend {self.backend!r} has no preset speakers; "
                               f"use set_voice() instead")
        if rc != 0:
            raise RuntimeError(f"set_speaker_name failed (rc={rc}) for backend {self.backend!r}")

    def set_speaker_id(self, speaker_id: int) -> None:
        """Select a speaker by integer index (melotts, piper, fastpitch).

        Multi-speaker TTS backends use 0-based integer speaker IDs.
        For melotts: 0=EN-US, 1=EN-BR, etc. Valid range is
        ``[0, n_speakers - 1]`` where ``n_speakers`` comes from
        :meth:`speakers` (which now also returns counts for
        integer-indexed backends).

        Raises :exc:`ValueError` if the id is out of range, or
        :exc:`RuntimeError` if the backend has no integer-speaker
        contract (use :meth:`set_speaker_name` for name-based backends
        like orpheus).
        """
        if not hasattr(self._lib, "crispasr_session_set_speaker_id"):
            raise RuntimeError("set_speaker_id API not present in this libcrispasr build")
        self._lib.crispasr_session_set_speaker_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_speaker_id.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_speaker_id(self._handle, speaker_id)
        if rc == -2:
            raise ValueError(f"speaker id {speaker_id} out of range for backend {self.backend!r}")
        if rc == -3:
            raise RuntimeError(f"backend {self.backend!r} has no integer-speaker contract; "
                               f"use set_speaker_name() instead")
        if rc != 0:
            raise RuntimeError(f"set_speaker_id failed (rc={rc}) for backend {self.backend!r}")

    def set_instruct(self, instruct: str) -> None:
        """Set the voice description / style instruct (qwen3-tts, parler, omnivoice).

        VoiceDesign generates speech in a voice **described by a
        natural-language instruction** — no reference WAV, no preset
        speaker. The instruct text is wrapped as
        ``"<|im_start|>user\\n{instruct}<|im_end|>\\n"`` and prepended
        to the talker prefill; the codec bridge omits the speaker
        frame entirely.

        Required for qwen3-tts VoiceDesign before
        :meth:`synthesize`. Re-callable; latest call wins. Raises if
        the active backend has no instruct contract.

        Detect VoiceDesign via :meth:`is_voice_design`.

        .. warning::
           **omnivoice does not take free prose.** It was trained on a
           closed 48-item vocabulary — a gender, age, pitch, style,
           accent or Chinese dialect, comma-separated, at most one per
           category — and the string reaches its prompt literally, so
           anything else is rejected rather than ignored::

               s.set_instruct("female, elderly, british accent")   # ok
               s.set_instruct("a gruff pirate")                    # raises

           Casing and separator width are normalised for you, and the
           whole instruct is unified to the language of the text being
           spoken (``"male, elderly"`` becomes ``男，老年`` for Chinese
           text). See docs/tts.md for the full vocabulary.
        """
        if not hasattr(self._lib, "crispasr_session_set_instruct"):
            raise RuntimeError("set_instruct API not present in this libcrispasr build")
        self._lib.crispasr_session_set_instruct.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_instruct.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_instruct(self._handle, instruct.encode("utf-8"))
        if rc == -3:
            raise RuntimeError(f"backend {self.backend!r} is not a VoiceDesign variant; "
                               f"set_instruct only applies to qwen3-tts VoiceDesign models")
        if rc != 0:
            raise RuntimeError(f"set_instruct failed (rc={rc}) for backend {self.backend!r}")

    def set_tts_phonemes(self, phonemes: str) -> None:
        """Synthesize these phonemes verbatim instead of phonemizing the text — the seam between text processing and the acoustic model. Use it to reproduce another implementation's pronunciation exactly, or to tell a G2P bug from a model bug (#316). Empty clears. Honoured by kokoro and piper; other backends soft no-op (rc=-2).

        Args:
            phonemes: IPA string in the backend's own alphabet, or "" to clear.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_phonemes"):
            raise RuntimeError("set_tts_phonemes API not present in this libcrispasr build")
        self._lib.crispasr_session_set_tts_phonemes.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_tts_phonemes.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_phonemes(self._session, phonemes.encode("utf-8"))
        if rc == -2:
            raise RuntimeError(
                f"backend {self.backend!r} has no phonemes-in entry point (kokoro and piper do)"
            )
        if rc != 0:
            raise RuntimeError(f"set_tts_phonemes failed (rc={rc})")

    def set_tts_pad_silence_ms(self, ms: int) -> None:
        """Pad N ms of silence at the beginning of TTS output. Useful to bypass VLC playback bugs where it drops the first ~1.5s of audio while parsing a large C2PA chunk.

        Args:
            ms: Milliseconds of silence to pad (e.g. 1500).
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_pad_silence_ms"):
            raise RuntimeError("set_tts_pad_silence_ms API not present in this libcrispasr build")
        self._lib.crispasr_session_set_tts_pad_silence_ms.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_tts_pad_silence_ms.restype = None
        self._lib.crispasr_session_set_tts_pad_silence_ms(self._session, int(ms))

    def clear_phoneme_cache(self) -> None:
        """Drop the kokoro per-session phoneme cache.

        No-op for non-kokoro backends. Useful for long-running daemons
        that resynthesize across many speakers and want bounded memory.
        """
        if not hasattr(self._lib, "crispasr_session_kokoro_clear_phoneme_cache"):
            return
        self._lib.crispasr_session_kokoro_clear_phoneme_cache.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_kokoro_clear_phoneme_cache.restype = ctypes.c_int
        rc = self._lib.crispasr_session_kokoro_clear_phoneme_cache(self._handle)
        if rc != 0:
            raise RuntimeError(f"clear_phoneme_cache failed (rc={rc})")

    # ------------------------------------------------------------------
    # Sticky session-state setters (PLAN #59 partial unblock).
    # ------------------------------------------------------------------

    def set_source_language(self, lang: str) -> None:
        """Sticky source-language hint (canary, cohere, voxtral, whisper).

        Empty string clears. Per-call ``language`` arg passed to
        :meth:`transcribe` still wins.
        """
        if not hasattr(self._lib, "crispasr_session_set_source_language"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_source_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_source_language.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_source_language(self._handle, lang.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_source_language failed (rc={rc})")

    def set_target_language(self, lang: str) -> None:
        """Sticky target-language. When set ≠ source on canary/cohere, the
        backend emits a translation. For whisper, pair with
        :meth:`set_translate` ``(True)``.
        """
        if not hasattr(self._lib, "crispasr_session_set_target_language"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_target_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_target_language.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_target_language(self._handle, lang.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_target_language failed (rc={rc})")

    def set_tts_reference_language(self, lang: str) -> None:
        """Language a voice-cloning reference clip is spoken in (issue #329).

        Cross-lingual TTS backends (cosyvoice3) compare it to the requested
        output language — :meth:`set_target_language`, falling back to
        :meth:`set_source_language` — and drop the reference transcript when
        they differ, so the clone speaks the target language rather than
        carrying the reference's accent.

        Optional: the backend otherwise infers the reference language from the
        voice-bank entry or the reference transcript. That inference cannot
        answer for a short transcript, and when it cannot, the requested target
        language has no effect — set this to make it explicit. Empty string
        clears.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_reference_language"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_tts_reference_language.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_tts_reference_language.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_reference_language(self._handle, lang.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_tts_reference_language failed (rc={rc})")

    def set_punctuation(self, enable: bool) -> None:
        """Toggle punctuation + capitalisation in the output (canary/cohere
        natively; LLM backends via post-process strip). Default True."""
        if not hasattr(self._lib, "crispasr_session_set_punctuation"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_punctuation.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_punctuation.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_punctuation(self._handle, 1 if enable else 0)
        if rc != 0:
            raise RuntimeError(f"set_punctuation failed (rc={rc})")

    def set_punc_model(self, punc_model: str) -> None:
        """Select + load a punctuation-restoration model on the session.

        ``punc_model`` is an alias (``auto`` / ``firered`` / ``fullstop`` /
        ``punctuate-all`` / ``pcs``) or a path to a ``.gguf``; ``"none"`` or
        ``""`` unloads. The model auto-downloads on first use. This restores
        punctuation on backends that emit none (parakeet RNNT/CTC, etc.) —
        the same post-processor the CLI ``--punc-model`` and server apply.
        """
        if not hasattr(self._lib, "crispasr_session_set_punc_model"):
            raise RuntimeError("punc-model API not present in this libcrispasr build")
        self._lib.crispasr_session_set_punc_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_punc_model.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_punc_model(self._handle, (punc_model or "").encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_punc_model failed (rc={rc})")

    def set_hotwords(self, hotwords: str, boost: float = 2.0) -> None:
        """Contextual biasing: comma-separated words/phrases to boost during
        decoding. Parakeet CTC/TDT use an Aho-Corasick trie; LLM backends inject
        them into the prompt (vibevoice splices the raw string into its
        "with extra info:" prompt slot, same as the CLI's --context). Empty
        string clears."""
        if not hasattr(self._lib, "crispasr_session_set_hotwords"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_hotwords.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_float]
        self._lib.crispasr_session_set_hotwords.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_hotwords(self._handle, (hotwords or "").encode("utf-8"), float(boost))
        if rc != 0:
            raise RuntimeError(f"set_hotwords failed (rc={rc})")

    def set_g2p_dict(self, source: str) -> None:
        """Select the G2P pronunciation dictionary for TTS phonemization
        (``olaph`` / ``open-dict`` or a path). Empty string keeps the default."""
        if not hasattr(self._lib, "crispasr_session_set_g2p_dict"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_g2p_dict.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_g2p_dict.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_g2p_dict(self._handle, (source or "").encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_g2p_dict failed (rc={rc})")

    def set_translate(self, enable: bool) -> None:
        """Whisper sticky ``--translate``. For canary/cohere/voxtral the
        equivalent is :meth:`set_target_language` ≠ source."""
        if not hasattr(self._lib, "crispasr_session_set_translate"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_translate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_translate.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_translate(self._handle, 1 if enable else 0)
        if rc != 0:
            raise RuntimeError(f"set_translate failed (rc={rc})")

    def set_temperature(self, temperature: float, seed: int = 0) -> None:
        """Set decoder temperature on backends that support runtime control
        (canary, cohere, parakeet, nemotron, moonshine). Other backends silently no-op.

        ``seed`` is the RNG seed for sampling; pass 0 for time-based.
        Returns silently when no backend in the session honours the
        setter (so it's safe to call for any session)."""
        if not hasattr(self._lib, "crispasr_session_set_temperature"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_uint64]
        self._lib.crispasr_session_set_temperature.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_temperature(self._handle, float(temperature), int(seed))
        # rc == -2 means no backend in this session supports temperature
        # — treat as a soft no-op so it's safe to call for any session.
        if rc not in (0, -2):
            raise RuntimeError(f"set_temperature failed (rc={rc})")

    def set_tts_seed(self, seed: int) -> None:
        """Reseed TTS backends that support runtime seed control.

        This is a soft no-op for sessions whose loaded backend does not
        expose a reseed hook.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_seed"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_tts_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self._lib.crispasr_session_set_tts_seed.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_seed(self._handle, int(seed))
        if rc not in (0, -2):
            raise RuntimeError(f"set_tts_seed failed (rc={rc})")

    def set_max_new_tokens(self, max_new_tokens: int) -> None:
        """Set a generated-token cap for autoregressive session backends.

        Pass ``<= 0`` to clear the override and use the backend default."""
        if not hasattr(self._lib, "crispasr_session_set_max_new_tokens"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_max_new_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_max_new_tokens.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_max_new_tokens(self._handle, int(max_new_tokens))
        if rc != 0:
            raise RuntimeError(f"set_max_new_tokens failed (rc={rc})")

    def set_frequency_penalty(self, penalty: float) -> None:
        """Set an opt-in repeated generated-token penalty for AR backends.

        Pass ``<= 0`` to disable it."""
        if not hasattr(self._lib, "crispasr_session_set_frequency_penalty"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        self._lib.crispasr_session_set_frequency_penalty.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_frequency_penalty.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_frequency_penalty(self._handle, float(penalty))
        if rc != 0:
            raise RuntimeError(f"set_frequency_penalty failed (rc={rc})")

    def set_tts_steps(self, steps: int) -> None:
        """Set the diffusion / CFM / masked-iterative step count for step-based
        TTS backends (chatterbox, vibevoice, kugelaudio, tada, irodori, omnivoice).

        Higher = better fidelity, slower. Soft no-op (rc=-2) when the active
        backend has no step-based stage.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_steps"):
            return
        self._lib.crispasr_session_set_tts_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_tts_steps.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_steps(self._handle, int(steps))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_tts_steps failed (rc={rc})")

    def set_tts_num_candidates(self, n: int) -> None:
        """Set the number of flow-matching timing candidates ranked per token.

        Honoured by TADA, where more candidates give more reliable
        multilingual timing at higher cost. Soft no-op (rc=-2) on backends
        that don't rank timing candidates.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_num_candidates"):
            return
        self._lib.crispasr_session_set_tts_num_candidates.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_tts_num_candidates.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_num_candidates(self._handle, int(n))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_tts_num_candidates failed (rc={rc})")

    def set_top_p(self, top_p: float) -> None:
        """Set the top-p nucleus-sampling threshold. Honoured by chatterbox."""
        if not hasattr(self._lib, "crispasr_session_set_top_p"):
            return
        self._lib.crispasr_session_set_top_p.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_top_p.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_top_p(self._handle, float(top_p))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_top_p failed (rc={rc})")

    def set_top_k(self, top_k: int) -> None:
        """Set the top-k sampling cutoff (0 = disabled). Honoured by TADA."""
        if not hasattr(self._lib, "crispasr_session_set_top_k"):
            return
        self._lib.crispasr_session_set_top_k.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_top_k.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_top_k(self._handle, int(top_k))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_top_k failed (rc={rc})")

    def set_do_sample(self, enable: bool) -> None:
        """Enable/disable sampling (False = greedy). Honoured by TADA."""
        if not hasattr(self._lib, "crispasr_session_set_do_sample"):
            return
        self._lib.crispasr_session_set_do_sample.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_do_sample.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_do_sample(self._handle, 1 if enable else 0)
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_do_sample failed (rc={rc})")

    def set_min_p(self, min_p: float) -> None:
        """Set the min-p sampling threshold. Honoured by chatterbox."""
        if not hasattr(self._lib, "crispasr_session_set_min_p"):
            return
        self._lib.crispasr_session_set_min_p.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_min_p.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_min_p(self._handle, float(min_p))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_min_p failed (rc={rc})")

    def set_repetition_penalty(self, r: float) -> None:
        """Set the repetition penalty (1.0 = no penalty). Honoured by chatterbox."""
        if not hasattr(self._lib, "crispasr_session_set_repetition_penalty"):
            return
        self._lib.crispasr_session_set_repetition_penalty.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_repetition_penalty.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_repetition_penalty(self._handle, float(r))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_repetition_penalty failed (rc={rc})")

    def set_cfg_weight(self, cfg_weight: float) -> None:
        """Set the classifier-free-guidance weight (chatterbox). 0 disables CFG."""
        if not hasattr(self._lib, "crispasr_session_set_cfg_weight"):
            return
        self._lib.crispasr_session_set_cfg_weight.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_cfg_weight.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_cfg_weight(self._handle, float(cfg_weight))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_cfg_weight failed (rc={rc})")

    def set_tts_noise_temp(self, noise_temp: float) -> None:
        """TADA flow-matching noise temperature (Python noise_temp, default 0.9)."""
        if not hasattr(self._lib, "crispasr_session_set_tts_noise_temp"):
            return
        self._lib.crispasr_session_set_tts_noise_temp.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_tts_noise_temp.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_noise_temp(self._handle, float(noise_temp))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_tts_noise_temp failed (rc={rc})")

    def set_exaggeration(self, exaggeration: float) -> None:
        """Set the emotion-exaggeration scalar (chatterbox). 0.5 is the upstream default."""
        if not hasattr(self._lib, "crispasr_session_set_exaggeration"):
            return
        self._lib.crispasr_session_set_exaggeration.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_exaggeration.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_exaggeration(self._handle, float(exaggeration))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_exaggeration failed (rc={rc})")

    def set_max_speech_tokens(self, n: int) -> None:
        """Set the upper bound on speech tokens per synthesize call (chatterbox). Default 1000 ≈ 20 s."""
        if not hasattr(self._lib, "crispasr_session_set_max_speech_tokens"):
            return
        self._lib.crispasr_session_set_max_speech_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_max_speech_tokens.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_max_speech_tokens(self._handle, int(n))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_max_speech_tokens failed (rc={rc})")

    def set_min_speech_tokens(self, n: int) -> None:
        """Set the floor on generated audio length (MOSS TTS). Units are codec frames at 12.5 Hz (80 ms each), so n=25 floors at ~2 s; other backends no-op (rc=-2)."""
        if not hasattr(self._lib, "crispasr_session_set_min_speech_tokens"):
            return
        self._lib.crispasr_session_set_min_speech_tokens.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_min_speech_tokens.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_min_speech_tokens(self._handle, int(n))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_min_speech_tokens failed (rc={rc})")

    def set_length_scale(self, scale: float) -> None:
        """Set the per-phoneme length-scale / speaking-rate scalar. Honoured by kokoro."""
        if not hasattr(self._lib, "crispasr_session_set_length_scale"):
            return
        self._lib.crispasr_session_set_length_scale.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_length_scale.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_length_scale(self._handle, float(scale))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_length_scale failed (rc={rc})")

    def set_best_of(self, n: int) -> None:
        """Set the best-of-N sampling count for ASR backends."""
        if not hasattr(self._lib, "crispasr_session_set_best_of"):
            return
        self._lib.crispasr_session_set_best_of.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_best_of.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_best_of(self._handle, int(n))
        if rc != 0:
            raise RuntimeError(f"set_best_of failed (rc={rc})")

    def set_beam_size(self, n: int) -> None:
        """Set the beam-search width for ASR backends that support it."""
        if not hasattr(self._lib, "crispasr_session_set_beam_size"):
            return
        self._lib.crispasr_session_set_beam_size.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_beam_size.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_beam_size(self._handle, int(n))
        if rc != 0:
            raise RuntimeError(f"set_beam_size failed (rc={rc})")

    def set_return_logits(self, on: bool) -> None:
        """Opt in to capturing per-frame CTC logits (backends with a dense CTC
        grid: Omni CTC, wav2vec2/hubert/data2vec, canary-ctc).

        Off by default: capture copies an ``n_frames × n_vocab`` float grid per
        transcribe, so leave it off unless a consumer (e.g. forced alignment)
        needs the logits. Retrieve them with :meth:`transcribe_with_logits`.
        No-op on dylibs predating the accessor.
        """
        if not hasattr(self._lib, "crispasr_session_set_return_logits"):
            return
        self._lib.crispasr_session_set_return_logits.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_return_logits.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_return_logits(self._handle, 1 if on else 0)
        if rc != 0:
            raise RuntimeError(f"set_return_logits failed (rc={rc})")

    def set_grammar_text(self, gbnf_text: str, root_rule: str = "root", penalty: float = 100.0) -> None:
        """Set a GBNF grammar for constrained whisper decoding. Pass "" to clear."""
        if not hasattr(self._lib, "crispasr_session_set_grammar_text"):
            return
        self._lib.crispasr_session_set_grammar_text.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float,
        ]
        self._lib.crispasr_session_set_grammar_text.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_grammar_text(
            self._handle,
            gbnf_text.encode() if gbnf_text else None,
            root_rule.encode() if root_rule else None,
            float(penalty),
        )
        if rc == -2:
            raise ValueError("set_grammar_text: invalid GBNF or root rule not found")
        if rc != 0:
            raise RuntimeError(f"set_grammar_text failed (rc={rc})")

    def set_fallback_thresholds(
        self,
        entropy_thold: float,
        logprob_thold: float,
        no_speech_thold: float,
        temperature_inc: float,
    ) -> None:
        """Set whisper decoder fallback thresholds. temperature_inc=0.0 disables fallback."""
        if not hasattr(self._lib, "crispasr_session_set_fallback_thresholds"):
            return
        self._lib.crispasr_session_set_fallback_thresholds.argtypes = [
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        self._lib.crispasr_session_set_fallback_thresholds.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_fallback_thresholds(
            self._handle,
            float(entropy_thold), float(logprob_thold),
            float(no_speech_thold), float(temperature_inc),
        )
        if rc != 0:
            raise RuntimeError(f"set_fallback_thresholds failed (rc={rc})")

    def set_sensitivity(self, preset: str) -> None:
        """Apply a named bundle of the four decoder fallback thresholds.

        One of "conservative", "balanced" (the shipped defaults, always a
        no-op) or "aggressive"; "strict"/"default"/"loose" are aliases.
        Mirrors the CLI's --sensitivity.

        conservative tightens the entropy and logprob bars and LOWERS
        no_speech_thold, so borderline audio is discarded rather than guessed
        at -- fewer hallucinations, some marginal speech lost. aggressive does
        the opposite: quiet or whispered audio still produces text.

        The four thresholds interact (a decode is only retried when the
        logprob AND no-speech bars are both crossed), which is why they move
        as a set. A later set_fallback_thresholds() overrides this.

        Raises ValueError for an unrecognised preset -- a typo is never
        silently treated as "balanced".
        """
        if not hasattr(self._lib, "crispasr_session_set_sensitivity"):
            return
        self._lib.crispasr_session_set_sensitivity.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_sensitivity.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_sensitivity(self._handle, str(preset).encode("utf-8"))
        if rc == -2:
            raise ValueError(
                f"unknown sensitivity preset {preset!r} "
                "(expected: conservative, balanced, aggressive)"
            )
        if rc != 0:
            raise RuntimeError(f"set_sensitivity failed (rc={rc})")

    def set_alt_n(self, n: int) -> None:
        """Set per-token top-N alternative-candidate capture for whisper greedy decode. 0 = off."""
        if not hasattr(self._lib, "crispasr_session_set_alt_n"):
            return
        self._lib.crispasr_session_set_alt_n.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_alt_n.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_alt_n(self._handle, int(n))
        if rc != 0:
            raise RuntimeError(f"set_alt_n failed (rc={rc})")

    def set_whisper_decode_extras(
        self,
        suppress_nst: bool = False,
        suppress_regex: str = "",
        carry_initial_prompt: bool = False,
    ) -> None:
        """Set whisper-only text-suppression and prompt-carry extras."""
        if not hasattr(self._lib, "crispasr_session_set_whisper_decode_extras"):
            return
        self._lib.crispasr_session_set_whisper_decode_extras.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.crispasr_session_set_whisper_decode_extras.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_whisper_decode_extras(
            self._handle,
            int(suppress_nst),
            suppress_regex.encode() if suppress_regex else b"",
            int(carry_initial_prompt),
        )
        if rc != 0:
            raise RuntimeError(f"set_whisper_decode_extras failed (rc={rc})")

    def set_ask(self, prompt: str) -> None:
        """Set a free-form prompt passed to the backend on the next transcribe/synthesize call.

        Supported by: granite, voxtral, qwen3-asr, glm-asr, gemma4-e2b,
        mimo-asr, moss-audio, moss-diarize, lfm2-audio, mini-omni2, ark-asr. For moss-audio
        this enables audio understanding beyond ASR (e.g. "Describe the sounds
        in this clip." or "What language is spoken?"). For ark-asr it is a
        best-effort language hint (the model is promptless / not instruction-
        trained).
        """
        if not hasattr(self._lib, "crispasr_session_set_ask"):
            return
        self._lib.crispasr_session_set_ask.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_ask.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_ask(self._handle, prompt.encode())
        if rc != 0:
            raise RuntimeError(f"set_ask failed (rc={rc})")

    def detect_language(self, pcm, lid_model_path: str, method: int = 1) -> tuple:
        """Auto-detect spoken language on raw 16 kHz mono PCM.

        ``method``: 0=Whisper, 1=Silero (default), 2=Firered, 3=Ecapa.
        Returns ``(lang_iso2, confidence_in_0_to_1)``. Raises
        :class:`RuntimeError` on failure.
        """
        if not hasattr(self._lib, "crispasr_session_detect_language"):
            raise RuntimeError("session-state API not present in this libcrispasr build")
        import numpy as np
        pcm_arr = np.ascontiguousarray(pcm, dtype=np.float32)
        self._lib.crispasr_session_detect_language.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.crispasr_session_detect_language.restype = ctypes.c_int
        out_buf = ctypes.create_string_buffer(16)
        out_prob = ctypes.c_float(0.0)
        rc = self._lib.crispasr_session_detect_language(
            self._handle,
            pcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(pcm_arr.size),
            lid_model_path.encode("utf-8"),
            int(method),
            out_buf,
            16,
            ctypes.byref(out_prob),
        )
        if rc != 0:
            raise RuntimeError(f"detect_language failed (rc={rc})")
        return out_buf.value.decode("utf-8"), float(out_prob.value)

    # ------------------------------------------------------------------
    # Text translation (PLAN #59 binding parity).
    # ------------------------------------------------------------------

    def translate_text(self, text: str, src_lang: str, tgt_lang: str,
                       max_tokens: int = 512) -> str:
        """Translate text between languages using the loaded translation
        model (m2m100 or similar).  Returns the translated string.

        Requires a backend that supports text translation (currently
        only ``m2m100``).  Raises :class:`RuntimeError` on failure.
        """
        fn = "crispasr_session_translate_text"
        if not hasattr(self._lib, fn):
            raise RuntimeError("translate_text not present in this libcrispasr build")
        func = getattr(self._lib, fn)
        func.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p,
                         ctypes.c_char_p, ctypes.c_int]
        func.restype = ctypes.c_char_p
        result = func(self._handle, text.encode("utf-8"),
                      src_lang.encode("utf-8"), tgt_lang.encode("utf-8"),
                      int(max_tokens))
        if not result:
            raise RuntimeError("translate_text returned null — check model and language pair")
        out = result.decode("utf-8")
        # Free the malloc'd string
        free_fn = "crispasr_session_translate_text_free"
        if hasattr(self._lib, free_fn):
            getattr(self._lib, free_fn).argtypes = [ctypes.c_char_p]
            getattr(self._lib, free_fn)(result)
        return out

    # ------------------------------------------------------------------
    # Streaming API (PLAN #62a — Python wrapper for crispasr_stream_*).
    # ------------------------------------------------------------------

    def stream_open(self, *, step_ms: int = 3000, length_ms: int = 10000, keep_ms: int = 200,
                    language: str = "", translate: bool = False, live: bool = False) -> "Session._Stream":
        """Open a rolling-window streaming decoder for this session.

        Backends with native streaming today: whisper, kyutai-stt,
        moonshine-streaming, voxtral4b. Other backends raise
        :class:`RuntimeError`.

        ``step_ms``: how often to commit a partial transcript (default 3s).
        ``length_ms``: rolling-window size (default 10s).
        ``keep_ms``: trailing audio carried over between windows (200ms).

        ``live``: voxtral4b-only — when True, decode runs during ``feed()``
        so ``get_text()`` returns progressive transcript as audio arrives.
        Default False (PTT semantics: decode happens in ``flush()``).
        Sequential live decode is ~1.5× realtime on M1 Q4_K voxtral4b;
        falls behind realtime audio without parallel encoder/decoder
        threads. Useful for: faster-than-realtime offline streaming,
        post-utterance live captions, manual stop-and-resume capture.

        Returns a :class:`_Stream` handle. Feed PCM with
        :meth:`_Stream.feed` and pull text with :meth:`_Stream.get_text`.
        """
        if not hasattr(self._lib, "crispasr_session_stream_open"):
            raise RuntimeError("streaming API not present in this libcrispasr build")
        self._lib.crispasr_session_stream_open.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        self._lib.crispasr_session_stream_open.restype = ctypes.c_void_p
        h = self._lib.crispasr_session_stream_open(
            self._handle, self._n_threads, step_ms, length_ms, keep_ms,
            language.encode("utf-8") if language else b"", 1 if translate else 0,
        )
        if not h:
            raise RuntimeError(f"stream_open failed for backend {self.backend!r}")
        # Voxtral4b live-decode toggle. No-op on other backends. Idempotent.
        if live and hasattr(self._lib, "crispasr_stream_set_live_decode"):
            self._lib.crispasr_stream_set_live_decode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self._lib.crispasr_stream_set_live_decode.restype = None
            self._lib.crispasr_stream_set_live_decode(h, 1)
        return Session._Stream(self._lib, h)

    class _Stream:
        """Rolling-window streaming decoder handle returned by
        :meth:`Session.stream_open`. Feed PCM, pull text."""

        def __init__(self, lib, handle):
            self._lib = lib
            self._handle = handle
            self._closed = False

        def feed(self, pcm) -> int:
            """Push 16 kHz mono float32 PCM. Returns 0 if still buffering,
            1 if a new partial transcript is ready (call :meth:`get_text`).
            Raises on error."""
            import numpy as np
            arr = np.ascontiguousarray(pcm, dtype=np.float32)
            self._lib.crispasr_stream_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self._lib.crispasr_stream_feed.restype = ctypes.c_int
            rc = self._lib.crispasr_stream_feed(
                self._handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(arr.size),
            )
            if rc < 0:
                raise RuntimeError(f"stream_feed failed (rc={rc})")
            return rc

        def get_text(self) -> dict:
            """Return latest committed transcript as
            ``{"text": str, "t0": float, "t1": float, "counter": int}``.
            ``counter`` increments per commit; same value means no new text."""
            self._lib.crispasr_stream_get_text.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_int64),
            ]
            self._lib.crispasr_stream_get_text.restype = ctypes.c_int
            buf = ctypes.create_string_buffer(8192)
            t0 = ctypes.c_double(0.0)
            t1 = ctypes.c_double(0.0)
            counter = ctypes.c_int64(0)
            rc = self._lib.crispasr_stream_get_text(
                self._handle, buf, 8192, ctypes.byref(t0), ctypes.byref(t1), ctypes.byref(counter),
            )
            if rc < 0:
                raise RuntimeError(f"stream_get_text failed (rc={rc})")
            return {"text": buf.value.decode("utf-8"), "t0": t0.value, "t1": t1.value, "counter": counter.value}

        def flush(self) -> None:
            """Finalise any remaining buffered audio."""
            self._lib.crispasr_stream_flush.argtypes = [ctypes.c_void_p]
            self._lib.crispasr_stream_flush.restype = ctypes.c_int
            rc = self._lib.crispasr_stream_flush(self._handle)
            if rc < 0:
                raise RuntimeError(f"stream_flush failed (rc={rc})")

        def close(self) -> None:
            if self._closed or not self._handle:
                return
            self._lib.crispasr_stream_close.argtypes = [ctypes.c_void_p]
            self._lib.crispasr_stream_close.restype = None
            self._lib.crispasr_stream_close(self._handle)
            self._closed = True

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.close()

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

    def is_voice_design(self) -> bool:
        """Return True iff the loaded model is a qwen3-tts VoiceDesign variant.

        Lets callers branch on the voice-prompt API: VoiceDesign needs
        :meth:`set_instruct`, CustomVoice needs :meth:`set_speaker_name`,
        Base needs :meth:`set_voice`.
        """
        if not hasattr(self._lib, "crispasr_session_is_voice_design"):
            return False
        self._lib.crispasr_session_is_voice_design.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_is_voice_design.restype = ctypes.c_int
        return bool(self._lib.crispasr_session_is_voice_design(self._handle))

    def is_custom_voice(self) -> bool:
        """Return True iff the loaded model is a qwen3-tts CustomVoice variant."""
        if not hasattr(self._lib, "crispasr_session_is_custom_voice"):
            return False
        self._lib.crispasr_session_is_custom_voice.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_is_custom_voice.restype = ctypes.c_int
        return bool(self._lib.crispasr_session_is_custom_voice(self._handle))

    def speakers(self) -> list:
        """Return the list of preset speaker names for the active backend.

        Empty list if the backend has no preset-speaker contract
        (e.g. vibevoice, kokoro, qwen3-tts ICL/Base, qwen3-tts
        VoiceDesign). Orpheus returns the speakers baked into the GGUF
        metadata.
        """
        if not hasattr(self._lib, "crispasr_session_n_speakers"):
            return []
        self._lib.crispasr_session_n_speakers.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_n_speakers.restype = ctypes.c_int
        self._lib.crispasr_session_get_speaker_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_get_speaker_name.restype = ctypes.c_char_p
        n = self._lib.crispasr_session_n_speakers(self._handle)
        out = []
        for i in range(n):
            ptr = self._lib.crispasr_session_get_speaker_name(self._handle, i)
            if ptr:
                out.append(ptr.decode("utf-8", errors="replace"))
        return out

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesise ``text`` to mono float32 PCM as a numpy array.

        Output sample rate is backend-dependent (24 kHz for most engines;
        ``voxcpm2-tts`` returns 48 kHz).

        Works with any TTS-capable backend — ``vibevoice``, ``qwen3-tts``,
        ``kokoro``, ``orpheus``, ``chatterbox``, ``indextts``, ``voxcpm2-tts``,
        ``csm``, ``dia``, ``fastpitch``, ``bananamind-tts``, ``speecht5``,
        ``melotts``, ``piper``, ``parler-tts``, ``outetts``, ``cosyvoice3-tts``,
        ``pocket-tts``, ``f5-tts``, ``irodori-tts``, ``bark``, ``kugelaudio``, ``tada``,
        ``lfm2-audio``, ``voxtral-tts``, ``dots-tts``, ``fireredtts3``, ``omnivoice``.
        For qwen3-tts call :meth:`set_codec_path` and one of:

        * :meth:`set_voice` — Base variants (WAV + ref_text, or voice-pack GGUF)
        * :meth:`set_speaker_name` — CustomVoice variants (fixed speaker name)
        * :meth:`set_instruct` — VoiceDesign variants (natural-language description)

        Branch via :meth:`is_voice_design` / :meth:`is_custom_voice` —
        Base if neither returns True. For orpheus call
        :meth:`set_codec_path` (SNAC GGUF) and :meth:`set_speaker_name`.
        voxcpm2-tts runs zero-shot today — the CLI ``--voice`` flag is
        accepted but ignored (the adapter prints a warning and falls back
        to the default voice; cloning hookup is still pending).
        """
        if not hasattr(self._lib, "crispasr_session_synthesize"):
            raise RuntimeError("TTS API not present in this libcrispasr build")
        self._lib.crispasr_session_synthesize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.crispasr_session_synthesize.restype = ctypes.POINTER(ctypes.c_float)
        self._lib.crispasr_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.crispasr_pcm_free.restype = None
        n = ctypes.c_int(0)
        ptr = self._lib.crispasr_session_synthesize(self._handle, text.encode("utf-8"), ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError(f"synthesize returned no audio for backend {self.backend!r}")
        try:
            arr = np.ctypeslib.as_array(ptr, shape=(n.value,)).copy()
        finally:
            self._lib.crispasr_pcm_free(ptr)
        return arr

    def accept_marking_responsibility(self, attestation: str = "") -> None:
        """Attest acceptance of AI-content marking/disclosure responsibility
        (EU AI Act Art. 50). REQUIRED before :meth:`synthesize_raw` will return
        unmarked audio; the default :meth:`synthesize` is watermarked and needs
        no attestation. ``attestation`` is recorded for audit."""
        if not hasattr(self._lib, "crispasr_session_accept_marking_responsibility"):
            raise RuntimeError("marking-attestation API not present in this libcrispasr build")
        self._lib.crispasr_session_accept_marking_responsibility.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
        ]
        self._lib.crispasr_session_accept_marking_responsibility.restype = ctypes.c_int
        self._lib.crispasr_session_accept_marking_responsibility(self._handle, attestation.encode("utf-8"))

    def set_speaker_identity(self, identity: str) -> None:
        """Declare whose voice a PRESET voice is: "real_person", "synthetic" or
        "unknown". A preset shipped inside a model can be an identifiable
        individual, which makes its output a deep fake under EU AI Act
        Art. 3(60) even though nothing was cloned — so "not a clone" is not
        the same as "nothing to disclose". Setting real_person makes the
        Art. 50(4) reminder fire for a non-cloned voice; it does NOT require a
        consent attestation, because the donor's agreement to the training is
        settled upstream and you cannot attest to it.

        Raises :class:`ValueError` on an unrecognised value rather than
        silently downgrading it to "unknown" — a typo must not quietly remove
        a duty you meant to declare.
        """
        if not hasattr(self._lib, "crispasr_session_set_speaker_identity"):
            raise RuntimeError("speaker-identity API not present in this libcrispasr build")
        self._lib.crispasr_session_set_speaker_identity.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_speaker_identity.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_speaker_identity(self._handle, identity.encode("utf-8"))
        if rc == -2:
            raise ValueError(
                f"unrecognised speaker_identity {identity!r}; "
                "expected 'real_person', 'synthetic' or 'unknown'"
            )
        if rc != 0:
            raise RuntimeError(f"set_speaker_identity failed (rc={rc})")

    def synthesize_raw(self, text: str) -> "np.ndarray":
        """UNMARKED synthesis (no watermark), for callers that post-process
        before embedding the mark themselves. Hard-refused unless
        :meth:`accept_marking_responsibility` was called first. Prefer
        :meth:`synthesize` for the default watermarked output."""
        if not hasattr(self._lib, "crispasr_session_synthesize_raw"):
            raise RuntimeError("TTS raw API not present in this libcrispasr build")
        self._lib.crispasr_session_synthesize_raw.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.crispasr_session_synthesize_raw.restype = ctypes.POINTER(ctypes.c_float)
        self._lib.crispasr_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.crispasr_pcm_free.restype = None
        n = ctypes.c_int(0)
        ptr = self._lib.crispasr_session_synthesize_raw(self._handle, text.encode("utf-8"), ctypes.byref(n))
        if not ptr or n.value <= 0:
            raise RuntimeError(
                "synthesize_raw returned no audio (attestation required? "
                "call accept_marking_responsibility() first)"
            )
        try:
            arr = np.ctypeslib.as_array(ptr, shape=(n.value,)).copy()
        finally:
            self._lib.crispasr_pcm_free(ptr)
        return arr

    def separate(self, pcm_stereo: "np.ndarray") -> dict:
        """Source separation: split audio into named stems.

        Input is stereo interleaved float32 PCM at the model's native
        sample rate (44100 Hz for ``htdemucs``).  Returns a dict mapping
        stem names (``"drums"``, ``"bass"``, ``"other"``, ``"vocals"``)
        to stereo interleaved float32 numpy arrays.

        Works with separation-capable backends — ``htdemucs``.
        """
        lib = self._lib
        lib.crispasr_session_separate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        lib.crispasr_session_separate.restype = ctypes.c_int
        lib.crispasr_session_separate_n_stems.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_separate_n_stems.restype = ctypes.c_int
        lib.crispasr_session_separate_stem_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.crispasr_session_separate_stem_name.restype = ctypes.c_char_p
        lib.crispasr_session_separate_stem.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
        ]
        lib.crispasr_session_separate_stem.restype = ctypes.POINTER(ctypes.c_float)
        lib.crispasr_session_separate_sample_rate.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_separate_sample_rate.restype = ctypes.c_int

        data = pcm_stereo.astype(np.float32)
        n_samples = len(data) // 2  # stereo interleaved
        n_stems = lib.crispasr_session_separate(
            self._handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n_samples,
        )
        if n_stems <= 0:
            raise RuntimeError(f"separate failed for backend {self.backend!r}")

        result = {}
        for i in range(n_stems):
            name_ptr = lib.crispasr_session_separate_stem_name(self._handle, i)
            name = name_ptr.decode("utf-8") if name_ptr else f"stem{i}"
            n_out = ctypes.c_int(0)
            ptr = lib.crispasr_session_separate_stem(self._handle, i, ctypes.byref(n_out))
            if ptr and n_out.value > 0:
                sr = lib.crispasr_session_separate_sample_rate(self._handle)
                n_ch = 2  # stereo
                arr = np.ctypeslib.as_array(ptr, shape=(n_out.value * n_ch,)).copy()
                result[name] = arr
        return result

    def pitch(self, pcm_mono: "np.ndarray", hop_ms: float = 10.0) -> "np.ndarray":
        """Pitch (F0) estimation: mono audio in, a pitch track out.

        Input is mono float32 PCM at the model's native sample rate
        (16000 Hz for ``crepe``).  Returns an ``(n_frames, 3)`` float32
        array whose columns are ``time_ms``, ``f0_hz`` and
        ``voiced_prob``.

        Works with pitch-capable backends — ``crepe``.
        """
        lib = self._lib
        lib.crispasr_session_pitch.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float,
        ]
        lib.crispasr_session_pitch.restype = ctypes.c_int
        lib.crispasr_session_pitch_frames.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
        lib.crispasr_session_pitch_frames.restype = ctypes.POINTER(ctypes.c_float)

        data = pcm_mono.astype(np.float32)
        n = lib.crispasr_session_pitch(
            self._handle, data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(data), float(hop_ms),
        )
        if n <= 0:
            raise RuntimeError(f"pitch failed for backend {self.backend!r}")

        n_out = ctypes.c_int(0)
        ptr = lib.crispasr_session_pitch_frames(self._handle, ctypes.byref(n_out))
        if not ptr or n_out.value <= 0:
            raise RuntimeError("pitch returned no frames")
        return np.ctypeslib.as_array(ptr, shape=(n_out.value * 3,)).copy().reshape(-1, 3)

    def speech_to_speech(self, input_pcm: "np.ndarray", language: str = None) -> tuple:
        """Speech-to-speech: audio in → audio out via a single model pass.

        Supported on backends with S2S capability (``lfm2-audio``,
        ``mini-omni2``, ``sidon``, ``voxcpm2-vae``). Input is mono float32 PCM; call
        :meth:`set_pcm_sample_rate` first when it is not 16 kHz. Returns a
        tuple ``(output_pcm, transcript)`` where *output_pcm* is a
        float32 numpy array at the backend's output sample rate (24 kHz
        for conversational S2S, 48 kHz for Sidon and VoxCPM2 AudioVAE) and *transcript* is the
        intermediate ASR text (may be
        empty if the backend doesn't produce one).

        Raises :class:`RuntimeError` if the C ABI lacks the symbol or
        if the backend doesn't support S2S.
        """
        if not hasattr(self._lib, "crispasr_session_speech_to_speech"):
            raise RuntimeError("S2S API not present in this libcrispasr build")
        self._lib.crispasr_session_speech_to_speech.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.crispasr_session_speech_to_speech.restype = ctypes.POINTER(ctypes.c_float)
        self._lib.crispasr_pcm_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
        self._lib.crispasr_pcm_free.restype = None
        import numpy as np
        in_arr = np.ascontiguousarray(input_pcm, dtype=np.float32)
        in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        text_out = ctypes.c_char_p(None)
        n_out = ctypes.c_int(0)
        ptr = self._lib.crispasr_session_speech_to_speech(
            self._handle, in_ptr, len(in_arr),
            ctypes.byref(text_out), ctypes.byref(n_out))
        if not ptr or n_out.value <= 0:
            raise RuntimeError(f"speech_to_speech returned no audio for backend {self.backend!r}")
        try:
            arr = np.ctypeslib.as_array(ptr, shape=(n_out.value,)).copy()
        finally:
            self._lib.crispasr_pcm_free(ptr)
        transcript = text_out.value.decode("utf-8") if text_out.value else ""
        if text_out.value and hasattr(self._lib, "crispasr_session_translate_text_free"):
            self._lib.crispasr_session_translate_text_free.argtypes = [ctypes.c_char_p]
            self._lib.crispasr_session_translate_text_free.restype = None
            self._lib.crispasr_session_translate_text_free(text_out)
        return arr, transcript

    def input_sample_rate(self) -> int:
        """Sample rate this backend expects for input PCM, in Hz.

        :meth:`speech_to_speech` and the other PCM entry points want audio at
        the backend's native rate, and it varies by backend — so telling
        callers to resample to it, without giving them a way to ask what it
        is, is not actionable (issue #321).
        """
        if not hasattr(self._lib, "crispasr_session_input_sample_rate"):
            raise RuntimeError("input_sample_rate not present in this libcrispasr build")
        self._lib.crispasr_session_input_sample_rate.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_input_sample_rate.restype = ctypes.c_int
        return int(self._lib.crispasr_session_input_sample_rate(self._handle))

    def output_sample_rate(self) -> int:
        """Sample rate of PCM this backend returns, in Hz.

        The counterpart to :meth:`input_sample_rate`: what
        :meth:`speech_to_speech` and :meth:`synthesize` hand back. 24 kHz for
        conversational S2S, 48 kHz for Sidon and VoxCPM2 AudioVAE.
        """
        if not hasattr(self._lib, "crispasr_session_output_sample_rate"):
            raise RuntimeError("output_sample_rate not present in this libcrispasr build")
        self._lib.crispasr_session_output_sample_rate.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_session_output_sample_rate.restype = ctypes.c_int
        return int(self._lib.crispasr_session_output_sample_rate(self._handle))

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.crispasr_session_close(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =========================================================================
# FireRedPunc — punctuation restoration post-processor
# =========================================================================

class PuncModel:
    """BERT-based punctuation restoration (FireRedPunc).

    Adds punctuation and capitalization to unpunctuated ASR output.
    Particularly useful for CTC-based backends (wav2vec2, omniasr,
    fastconformer-ctc, firered-asr) that output lowercase text without
    punctuation.

    Usage::

        punc = crispasr.PuncModel("fireredpunc-q8_0.gguf")
        text = punc.process("and so my fellow americans ask not")
        punc.close()

    Or as context manager::

        with crispasr.PuncModel("fireredpunc.gguf") as punc:
            for seg in segments:
                seg.text = punc.process(seg.text)
    """

    def __init__(self, model_path: str, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._setup_punc_signatures()
        self._handle = self._lib.crispasr_punc_init(model_path.encode("utf-8"))
        if not self._handle:
            raise RuntimeError(f"Failed to load punctuation model: {model_path}")

    def _setup_punc_signatures(self):
        lib = self._lib
        for name in ("crispasr_punc_init", "crispasr_punc_process",
                      "crispasr_punc_free_text", "crispasr_punc_free"):
            if not hasattr(lib, name):
                raise RuntimeError(
                    "FireRedPunc API not found — rebuild CrispASR 0.5.0+")
        lib.crispasr_punc_init.argtypes = [ctypes.c_char_p]
        lib.crispasr_punc_init.restype = ctypes.c_void_p
        lib.crispasr_punc_process.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.crispasr_punc_process.restype = ctypes.c_char_p
        lib.crispasr_punc_free_text.argtypes = [ctypes.c_char_p]
        lib.crispasr_punc_free_text.restype = None
        lib.crispasr_punc_free.argtypes = [ctypes.c_void_p]
        lib.crispasr_punc_free.restype = None

    def process(self, text: str) -> str:
        """Add punctuation to unpunctuated text."""
        result = self._lib.crispasr_punc_process(
            self._handle, text.encode("utf-8"))
        if not result:
            return text
        out = result.decode("utf-8")
        self._lib.crispasr_punc_free_text(result)
        return out

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.crispasr_punc_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# =========================================================================
# TitaNet speaker verification + speaker profile DB
# =========================================================================

class TitaNet:
    """TitaNet-Large speaker embedding extractor (192-d, L2-normalized)."""

    def __init__(self, model_path: str, n_threads: int = 4, lib_path: str = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._lib.crispasr_titanet_init.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        self._lib.crispasr_titanet_init.restype = ctypes.c_void_p
        self._lib.crispasr_titanet_free.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_titanet_free.restype = None
        self._lib.crispasr_titanet_embed.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.crispasr_titanet_embed.restype = ctypes.c_int32
        self._lib.crispasr_titanet_cosine_sim.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ]
        self._lib.crispasr_titanet_cosine_sim.restype = ctypes.c_float
        self._ctx = self._lib.crispasr_titanet_init(model_path.encode(), n_threads)
        if not self._ctx:
            raise RuntimeError(f"Failed to load TitaNet model: {model_path}")

    def embed(self, pcm_16k):
        """Extract 192-d speaker embedding from 16 kHz mono float32 PCM."""
        import numpy as np
        pcm = np.ascontiguousarray(pcm_16k, dtype=np.float32)
        out = np.zeros(192, dtype=np.float32)
        dim = self._lib.crispasr_titanet_embed(
            self._ctx,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(pcm),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if dim <= 0:
            raise RuntimeError("TitaNet embedding extraction failed")
        return out[:dim]

    @staticmethod
    def cosine_sim(a, b, lib_path=None):
        """Cosine similarity between two embeddings (dot product for L2-normed)."""
        import numpy as np
        return float(np.dot(a, b))

    def close(self):
        if self._ctx:
            self._lib.crispasr_titanet_free(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SpeakerDB:
    """Closed-roster speaker profile database (issue #266).

    Named identification is a claimed-participant confirmation, never an
    open 1:N search: ``expected_names`` (comma-separated, e.g.
    ``"Alice,Bob"``) is the roster of enrolled participants you assert are
    present in the audio, and the db is narrowed to exactly those
    profiles. ``consent`` affirms a lawful basis + explicit consent from
    every enrolled person (GDPR Art. 9); opening and enrolling refuse
    without it.
    """

    def __init__(self, dir_path: str, expected_names: str = "", consent: bool = False, lib_path: str = None):
        # Set _db before any check that can raise, so __del__ -> close()
        # (which reads self._db) never sees a half-constructed instance —
        # otherwise a no-consent ValueError during __init__ leaves _db
        # unset and garbage collection prints a spurious "Exception
        # ignored in __del__: AttributeError" for every refused instance.
        self._db = None
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._lib.crispasr_speaker_db_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32]
        self._lib.crispasr_speaker_db_open.restype = ctypes.c_void_p
        self._lib.crispasr_speaker_db_free.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_db_free.restype = None
        self._lib.crispasr_speaker_db_count.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_db_count.restype = ctypes.c_int32
        self._lib.crispasr_speaker_db_match.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_float, ctypes.c_char_p, ctypes.c_int32,
        ]
        self._lib.crispasr_speaker_db_match.restype = ctypes.c_float
        self._lib.crispasr_speaker_db_enroll2.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
        ]
        self._lib.crispasr_speaker_db_enroll2.restype = ctypes.c_int32
        self._consent = bool(consent)
        if not self._consent:
            raise ValueError(
                "SpeakerDB requires consent=True: matching named voiceprints is biometric "
                "identification (GDPR Art. 9); affirm a lawful basis + explicit consent "
                "from every enrolled person"
            )
        if expected_names:
            self._db = self._lib.crispasr_speaker_db_open(dir_path.encode(), expected_names.encode(), 1)
        self._dir = dir_path

    @property
    def count(self):
        return self._lib.crispasr_speaker_db_count(self._db) if self._db else 0

    def match(self, embedding, threshold=0.7):
        """Match embedding against DB. Returns (name, score) or (None, score)."""
        import numpy as np
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        name_buf = ctypes.create_string_buffer(256)
        score = self._lib.crispasr_speaker_db_match(
            self._db, emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(emb), threshold, name_buf, 256,
        )
        name = name_buf.value.decode() if score >= threshold else None
        return name, float(score)

    def enroll(self, name, embedding):
        """Enroll a speaker with the given name and embedding.

        The consent attestation given at construction is recorded in the
        v2 .spkr profile (audit trail).
        """
        import numpy as np
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        rc = self._lib.crispasr_speaker_db_enroll2(
            self._dir.encode(), name.encode(),
            emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(emb), 1,
        )
        return rc == 0

    def close(self):
        if self._db:
            self._lib.crispasr_speaker_db_free(self._db)
            self._db = None

    def __del__(self):
        self.close()


# =====================================================================
# Diarization pipeline primitives (issue #107 P6)
# =====================================================================
# Pluggable speaker embedder, agglomerative cosine clustering, and a
# pyannote-seg cache for cross-slice consistency. Same building blocks
# the CLI's --diarize-embedder path uses; exposed here so Python
# pipelines can compose their own diarization without shelling out.


class SpeakerEmbedder:
    """Pluggable speaker-embedding model (TitaNet or IndexTTS today).

    ``model_spec`` accepts (case-insensitive):
      - ``"auto"`` / ``"titanet"`` -> TitaNet-Large (192-d, auto-DL)
      - ``"indextts"`` / ``"indextts-bigvgan"`` / ``"ecapa"`` ->
        IndexTTS-BigVGAN ECAPA-TDNN (512-d, auto-DL)
      - a ``.gguf`` path containing ``"indextts"`` -> IndexTTS adapter
      - any other path -> TitaNet adapter

    The factory follows the same dispatch rules as the CLI's
    ``--diarize-embedder`` flag (#107 P5).
    """

    def __init__(self, model_spec: str, n_threads: int = 4,
                 cache_dir: str = "", lib_path: str = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._lib.crispasr_speaker_embedder_make_abi.argtypes = [
            ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p,
        ]
        self._lib.crispasr_speaker_embedder_make_abi.restype = ctypes.c_void_p
        self._lib.crispasr_speaker_embedder_free_abi.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_embedder_free_abi.restype = None
        self._lib.crispasr_speaker_embedder_dim_abi.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_embedder_dim_abi.restype = ctypes.c_int32
        self._lib.crispasr_speaker_embedder_embed_abi.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.crispasr_speaker_embedder_embed_abi.restype = ctypes.c_int32
        self._lib.crispasr_speaker_embedder_name_abi.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_embedder_name_abi.restype = ctypes.c_char_p
        self._ctx = self._lib.crispasr_speaker_embedder_make_abi(
            model_spec.encode("utf-8"), int(n_threads), cache_dir.encode("utf-8"),
        )
        if not self._ctx:
            raise RuntimeError(f"Failed to build speaker embedder '{model_spec}'")

    @property
    def dim(self) -> int:
        return int(self._lib.crispasr_speaker_embedder_dim_abi(self._ctx))

    @property
    def name(self) -> str:
        name = self._lib.crispasr_speaker_embedder_name_abi(self._ctx)
        return name.decode("utf-8") if name else ""

    def embed(self, pcm_16k):
        """Extract one embedding from 16 kHz mono float32 PCM.

        Returns an ndarray of length ``dim`` on success. Returns ``None``
        when the embedder rejected the input (e.g. too short for the
        underlying model's mel pipeline).
        """
        import numpy as np
        pcm = np.ascontiguousarray(pcm_16k, dtype=np.float32)
        out = np.zeros(self.dim, dtype=np.float32)
        ok = self._lib.crispasr_speaker_embedder_embed_abi(
            self._ctx,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(pcm),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return out if ok else None

    def close(self):
        if self._ctx:
            self._lib.crispasr_speaker_embedder_free_abi(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def agglomerative_cluster(embeddings, *, merge_threshold: float = 0.5,
                          max_speakers: int = 32, lib_path: str = None):
    """Cluster a list/array of (ideally L2-normalized) speaker embeddings.

    ``embeddings`` is either an ``(n, dim)`` ndarray or a 2-D list. Uses
    agglomerative single-linkage clustering on cosine similarity with
    both a similarity-threshold stop and a hard ``max_speakers`` cap.
    Returns a 1-D numpy int array of length ``n`` with cluster IDs in
    ``[0, k)`` assigned in first-appearance order.
    """
    import numpy as np
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return np.zeros(0, dtype=np.int32)
    n, dim = arr.shape

    lib = ctypes.CDLL(lib_path or _find_lib())
    lib.crispasr_speaker_cluster_abi.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
    ]
    lib.crispasr_speaker_cluster_abi.restype = ctypes.c_int32
    out = np.zeros(n, dtype=np.int32)
    rc = lib.crispasr_speaker_cluster_abi(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, dim,
        ctypes.c_float(merge_threshold), int(max_speakers),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )
    if rc < 0:
        raise RuntimeError("agglomerative_cluster: invalid arguments")
    return out


class PyannoteCache:
    """Pre-computed pyannote-seg posteriors over a full audio buffer.

    Build once at the start of a diarize pipeline, then call
    :meth:`apply` for each set of segment ranges. Gives cross-slice
    consistency for pyannote-method diarization (#107 P2a) without
    re-running the segmentation net per slice.
    """

    def __init__(self, pcm_16k, model_path: str, n_threads: int = 4,
                 lib_path: str = None):
        import numpy as np
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._lib.crispasr_pyannote_cache_compute_abi.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_char_p,
            ctypes.c_int32,
        ]
        self._lib.crispasr_pyannote_cache_compute_abi.restype = ctypes.c_void_p
        self._lib.crispasr_pyannote_cache_free_abi.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_pyannote_cache_free_abi.restype = None
        self._lib.crispasr_pyannote_cache_apply_abi.argtypes = [
            ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int32,
        ]
        self._lib.crispasr_pyannote_cache_apply_abi.restype = ctypes.c_int32

        pcm = np.ascontiguousarray(pcm_16k, dtype=np.float32)
        self._ctx = self._lib.crispasr_pyannote_cache_compute_abi(
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(len(pcm)), model_path.encode("utf-8"), int(n_threads),
        )
        if not self._ctx:
            raise RuntimeError(
                f"Failed to compute pyannote cache from model '{model_path}'"
            )

    def apply(self, segs: List[DiarizeSegment], slice_t0: float = 0.0) -> None:
        """Score ``segs`` against the cached posteriors, mutating ``speaker``."""
        if not segs:
            return

        class _SegAbi(ctypes.Structure):
            _fields_ = [
                ("t0_cs", ctypes.c_int64),
                ("t1_cs", ctypes.c_int64),
                ("speaker", ctypes.c_int32),
                ("_pad", ctypes.c_int32),
            ]

        seg_array = (_SegAbi * len(segs))()
        for i, s in enumerate(segs):
            seg_array[i].t0_cs = int(round(s.t0 * 100))
            seg_array[i].t1_cs = int(round(s.t1 * 100))
            seg_array[i].speaker = s.speaker
        rc = self._lib.crispasr_pyannote_cache_apply_abi(
            self._ctx, int(round(slice_t0 * 100)),
            ctypes.byref(seg_array), len(segs),
        )
        if rc != 0:
            raise RuntimeError("PyannoteCache.apply: invalid arguments")
        for i, s in enumerate(segs):
            s.speaker = int(seg_array[i].speaker)

    def close(self):
        if self._ctx:
            self._lib.crispasr_pyannote_cache_free_abi(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ======================================================================
# Standalone utilities (no Session / model load required)
# ======================================================================

@dataclass
class VadSpan:
    """One speech span from standalone VAD."""
    start: float  # seconds
    end: float    # seconds


def vad_segments(
    pcm: "np.ndarray",
    model_path: str,
    *,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    n_threads: int = 4,
    lib_path: Optional[str] = None,
) -> List[VadSpan]:
    """Run standalone VAD on raw PCM without a full ASR session.

    Returns a list of :class:`VadSpan` with start/end in seconds.

    Args:
        pcm: 16 kHz mono float32 PCM array.
        model_path: path to a Silero/FireRed VAD GGUF.
        threshold: speech probability threshold (0-1, default 0.5).
        min_speech_ms: minimum speech duration to keep (ms).
        min_silence_ms: minimum silence to split on (ms).
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_vad_segments
    fn.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_bool, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    fn.restype = ctypes.c_int

    pcm_arr = np.ascontiguousarray(pcm, dtype=np.float32)
    out_spans = ctypes.POINTER(ctypes.c_float)()
    n = fn(
        model_path.encode("utf-8"),
        pcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(pcm_arr.size), sample_rate, threshold,
        min_speech_ms, min_silence_ms, n_threads, False,
        ctypes.byref(out_spans),
    )
    if n < 0:
        raise RuntimeError(f"crispasr_vad_segments failed (rc={n})")
    spans = []
    for i in range(n):
        spans.append(VadSpan(start=float(out_spans[2 * i]),
                             end=float(out_spans[2 * i + 1])))
    if n > 0:
        lib.free(out_spans)
    return spans


def text_detect_language(
    text: str,
    model_path: str,
    *,
    n_threads: int = 4,
    lib_path: Optional[str] = None,
) -> tuple:
    """Detect the language of a text string using a text-LID model.

    Returns ``(lang_code, confidence)`` where ``lang_code`` is an
    ISO 639-1/3 code and ``confidence`` is in [0, 1].

    Args:
        text: UTF-8 text to classify.
        model_path: path to a text-LID GGUF (GlotLID, LID-176, etc.).
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_text_detect_language
    fn.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32,
        ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float),
    ]
    fn.restype = ctypes.c_int

    out_label = ctypes.create_string_buffer(32)
    out_conf = ctypes.c_float(0.0)
    rc = fn(text.encode("utf-8"), model_path.encode("utf-8"), n_threads,
            out_label, 32, ctypes.byref(out_conf))
    if rc != 0:
        raise RuntimeError(f"text_detect_language failed (rc={rc})")
    return out_label.value.decode("utf-8"), float(out_conf.value)


def enhance_audio_rnnoise(
    pcm: "np.ndarray",
    *,
    lib_path: Optional[str] = None,
) -> "np.ndarray":
    """Apply RNNoise audio denoising to raw 48 kHz mono PCM.

    Returns a float32 array of the same length with noise reduced.
    Note: RNNoise operates at 48 kHz internally. If your audio is
    16 kHz, resample to 48 kHz first, denoise, then resample back.
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_enhance_audio_rnnoise
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
    ]
    fn.restype = ctypes.c_int

    pcm_arr = np.ascontiguousarray(pcm, dtype=np.float32)
    out = np.zeros_like(pcm_arr)
    rc = fn(
        pcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(pcm_arr.size),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(out.size),
    )
    if rc != 0:
        raise RuntimeError(f"enhance_audio_rnnoise failed (rc={rc})")
    return out


def detect_backend_from_gguf(
    gguf_path: str,
    *,
    lib_path: Optional[str] = None,
) -> str:
    """Detect which CrispASR backend a GGUF file belongs to.

    Returns the backend name (e.g. "parakeet", "cohere", "whisper").
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_detect_backend_from_gguf
    fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int

    out = ctypes.create_string_buffer(64)
    rc = fn(gguf_path.encode("utf-8"), out, 64)
    # ABI contract: rc > 0 = detected (strlen of the name), rc == 0 = valid
    # GGUF but its architecture maps to no backend (name is ""), rc < 0 = error.
    # A prior `rc != 0` check treated every successful detection as a failure.
    if rc < 0:
        raise RuntimeError(f"detect_backend_from_gguf failed (rc={rc})")
    return out.value.decode("utf-8")


# =========================================================================
# Direct Parakeet API (bypasses unified session)
# =========================================================================

class Parakeet:
    """Direct Parakeet ASR context with word- and token-level timestamps.

    For most use cases prefer :class:`Session` which auto-dispatches to
    Parakeet when the GGUF metadata indicates it.
    """

    def __init__(self, model_path: str, *, n_threads: int = 4,
                 use_flash: bool = True, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        fn = self._lib.crispasr_parakeet_init
        fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        fn.restype = ctypes.c_void_p
        self._handle = fn(model_path.encode("utf-8"), n_threads, 1 if use_flash else 0)
        if not self._handle:
            raise RuntimeError(f"Failed to load Parakeet model: {model_path}")

    def transcribe(self, pcm: "np.ndarray", language: Optional[str] = None):
        """Transcribe mono 16 kHz float32 PCM. Returns a dict with text,
        words [(text, t0_cs, t1_cs)], and tokens [(text, t0_cs, t1_cs, p)]."""
        pcm_arr = np.ascontiguousarray(pcm, dtype=np.float32)
        lib = self._lib
        fn = lib.crispasr_parakeet_transcribe
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                       ctypes.c_int, ctypes.c_char_p]
        fn.restype = ctypes.c_void_p
        lang = language.encode("utf-8") if language else None
        res = fn(self._handle,
                 pcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                 int(pcm_arr.size), lang)
        if not res:
            raise RuntimeError("crispasr_parakeet_transcribe returned null")
        try:
            # Text
            lib.crispasr_parakeet_result_text.argtypes = [ctypes.c_void_p]
            lib.crispasr_parakeet_result_text.restype = ctypes.c_char_p
            raw = lib.crispasr_parakeet_result_text(res)
            text = raw.decode("utf-8") if raw else ""
            # Words
            lib.crispasr_parakeet_result_n_words.argtypes = [ctypes.c_void_p]
            lib.crispasr_parakeet_result_n_words.restype = ctypes.c_int
            lib.crispasr_parakeet_result_word_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_word_text.restype = ctypes.c_char_p
            lib.crispasr_parakeet_result_word_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_word_t0.restype = ctypes.c_int64
            lib.crispasr_parakeet_result_word_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_word_t1.restype = ctypes.c_int64
            nw = lib.crispasr_parakeet_result_n_words(res)
            words = []
            for i in range(nw):
                wt = lib.crispasr_parakeet_result_word_text(res, i)
                words.append((
                    wt.decode("utf-8") if wt else "",
                    lib.crispasr_parakeet_result_word_t0(res, i),
                    lib.crispasr_parakeet_result_word_t1(res, i),
                ))
            # Tokens
            lib.crispasr_parakeet_result_n_tokens.argtypes = [ctypes.c_void_p]
            lib.crispasr_parakeet_result_n_tokens.restype = ctypes.c_int
            lib.crispasr_parakeet_result_token_text.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_token_text.restype = ctypes.c_char_p
            lib.crispasr_parakeet_result_token_t0.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_token_t0.restype = ctypes.c_int64
            lib.crispasr_parakeet_result_token_t1.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_token_t1.restype = ctypes.c_int64
            lib.crispasr_parakeet_result_token_p.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.crispasr_parakeet_result_token_p.restype = ctypes.c_float
            nt = lib.crispasr_parakeet_result_n_tokens(res)
            tokens = []
            for i in range(nt):
                tt = lib.crispasr_parakeet_result_token_text(res, i)
                tokens.append((
                    tt.decode("utf-8") if tt else "",
                    lib.crispasr_parakeet_result_token_t0(res, i),
                    lib.crispasr_parakeet_result_token_t1(res, i),
                    float(lib.crispasr_parakeet_result_token_p(res, i)),
                ))
            return {"text": text, "words": words, "tokens": tokens}
        finally:
            lib.crispasr_parakeet_result_free.argtypes = [ctypes.c_void_p]
            lib.crispasr_parakeet_result_free.restype = None
            lib.crispasr_parakeet_result_free(res)

    def close(self):
        if self._handle:
            self._lib.crispasr_parakeet_free.argtypes = [ctypes.c_void_p]
            self._lib.crispasr_parakeet_free.restype = None
            self._lib.crispasr_parakeet_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# =========================================================================
# Standalone helpers — full C-ABI parity
# =========================================================================

def lcs_dedup_prefix_count(
    prev_tail_tokens: List[int],
    curr_tokens: List[int],
    *,
    min_lcs_length: int = 1,
    lib_path: Optional[str] = None,
) -> int:
    """Chunk-boundary LCS dedup: returns the number of leading tokens
    of ``curr_tokens`` to drop to remove overlap with ``prev_tail_tokens``."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_lcs_dedup_prefix_count
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    prev_arr = (ctypes.c_int32 * len(prev_tail_tokens))(*prev_tail_tokens)
    curr_arr = (ctypes.c_int32 * len(curr_tokens))(*curr_tokens)
    return fn(prev_arr, len(prev_tail_tokens), curr_arr, len(curr_tokens), min_lcs_length)


def kokoro_lang_is_german(lang: str, *, lib_path: Optional[str] = None) -> bool:
    """Whether ``lang`` is German (Kokoro phoneme selection)."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_kokoro_lang_is_german_abi"):
        return False
    fn = lib.crispasr_kokoro_lang_is_german_abi
    fn.argtypes = [ctypes.c_char_p]
    fn.restype = ctypes.c_bool
    return fn(lang.encode("utf-8"))


def kokoro_lang_has_native_voice(lang: str, *, lib_path: Optional[str] = None) -> bool:
    """Whether ``lang`` has a native Kokoro voice (vs. cross-lingual fallback)."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_kokoro_lang_has_native_voice_abi"):
        return False
    fn = lib.crispasr_kokoro_lang_has_native_voice_abi
    fn.argtypes = [ctypes.c_char_p]
    fn.restype = ctypes.c_bool
    return fn(lang.encode("utf-8"))


def vad_slices(
    pcm: "np.ndarray",
    model_path: str,
    *,
    sample_rate: int = 16000,
    threshold: float = 0.0,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 30,
    max_chunk_duration_s: float = 30.0,
    n_threads: int = 4,
    lib_path: Optional[str] = None,
) -> List[VadSpan]:
    """Run the unified VAD dispatcher returning speech spans in seconds.

    Can use Silero, FireRedVAD, MarbleNet, or Whisper-VAD-EncDec depending
    on the concrete model at ``model_path``. threshold <= 0 leaves per-model
    default intact.
    """
    lib = ctypes.CDLL(lib_path or _find_lib())
    fn = lib.crispasr_vad_slices
    fn.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_float, ctypes.c_int,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ]
    fn.restype = ctypes.c_int
    lib.crispasr_vad_free.argtypes = [ctypes.POINTER(ctypes.c_float)]
    lib.crispasr_vad_free.restype = None

    pcm_arr = np.ascontiguousarray(pcm, dtype=np.float32)
    out_spans = ctypes.POINTER(ctypes.c_float)()
    n = fn(
        model_path.encode("utf-8"),
        pcm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(pcm_arr.size), sample_rate, threshold,
        min_speech_ms, min_silence_ms, speech_pad_ms,
        max_chunk_duration_s, n_threads,
        ctypes.byref(out_spans),
    )
    if n < 0:
        raise RuntimeError(f"crispasr_vad_slices failed (rc={n})")
    spans = []
    for i in range(n):
        spans.append(VadSpan(start=float(out_spans[2 * i]),
                             end=float(out_spans[2 * i + 1])))
    if n > 0:
        lib.crispasr_vad_free(out_spans)
    return spans


# ---------------------------------------------------------------------------
# Watermark — AI-generated audio marking
# ---------------------------------------------------------------------------

def watermark_load_model(gguf_path: str) -> None:
    """Load an AudioSeal GGUF for neural watermarking.

    Once loaded, :func:`watermark_embed` and :func:`watermark_detect`
    dispatch to AudioSeal automatically. Without loading, they use the
    built-in spread-spectrum watermark.
    """
    lib = _get_lib()
    fn = lib.crispasr_watermark_load_model
    fn.argtypes = [ctypes.c_char_p]
    fn.restype = ctypes.c_int
    rc = fn(gguf_path.encode())
    if rc != 0:
        raise RuntimeError(f"crispasr_watermark_load_model failed (rc={rc})")


def watermark_embed(pcm: "numpy.ndarray", alpha: float = -1.0) -> None:
    """Embed an AI-generated watermark into float32 PCM in-place.

    ``alpha`` controls spread-spectrum strength; ignored when AudioSeal is
    loaded. The default (``<= 0``) selects the band-limited strength that makes
    the mark reliably *detectable*, which is the property EU AI Act Art. 50(2)
    requires — this is the call :func:`Session.synthesize_raw` callers use to
    discharge marking themselves, so it has to produce a findable mark.

    This defaulted to ``0.005`` — the strength the C ABI documents as "too
    faint to reliably detect on real speech". An explicit positive alpha is
    passed through verbatim and bypasses the robust default, so every caller
    that relied on the default was emitting audio it could not detect a
    watermark in. Only pass a literal alpha to A/B watermark strength.
    """
    import numpy as np
    if pcm.dtype != np.float32:
        raise TypeError("pcm must be float32")
    lib = _get_lib()
    fn = lib.crispasr_watermark_embed
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float]
    fn.restype = None
    fn(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
       ctypes.c_int(len(pcm)), ctypes.c_float(alpha))


def watermark_detect(pcm: "numpy.ndarray") -> float:
    """Detect AI-generated watermark. Returns confidence in [0, 1]."""
    import numpy as np
    if pcm.dtype != np.float32:
        raise TypeError("pcm must be float32")
    lib = _get_lib()
    fn = lib.crispasr_watermark_detect
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    fn.restype = ctypes.c_float
    return float(fn(pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(len(pcm))))


# ---------------------------------------------------------------------------
# Chat / LLM — the crispasr_chat.h surface
# ---------------------------------------------------------------------------
#
# EU AI Act: this surface generates synthetic TEXT, which the runtime does NOT
# mark for you the way every audio path watermarks. Show
# `ChatSession.ai_disclosure_text()` at or before the first turn of anything
# that talks to a person, and read docs/eu-ai-act.md §6.6 before shipping.
#
# Every entry point of that header is bound here, crispasr_chat_memory_estimate
# included.

# The one error code crispasr_chat.h promises as a contract. Every other
# non-zero value is a diagnostic aid — read the message, don't switch on it.
CRISPASR_CHAT_ERR_ABORTED = 40


class ChatAborted(RuntimeError):
    """A registered abort predicate stopped the generation.

    Subclasses :class:`RuntimeError`, which is what every other failure on
    this binding raises, so ``except RuntimeError`` still catches a cancel
    while ``except ChatAborted`` tells a cancel apart from a decode fault.

    Whatever text already reached ``on_token`` is valid, and the session has
    been flushed back to its just-opened state — reuse it directly, without a
    :meth:`ChatSession.reset`.
    """


@dataclass
class ChatMessage:
    """One turn in a conversation.

    ``role`` is one of "system", "user", "assistant", "tool" (the OpenAI chat
    schema); the model's chat template maps those onto whatever it expects.
    Every method that takes messages also accepts plain dicts with the same
    two keys.
    """
    role: str
    content: str


# ABI structs — must match include/crispasr_chat.h.
class _ChatErrorAbi(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_int32),
        ("message", ctypes.c_char * 256),
    ]


class _ChatMessageAbi(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("content", ctypes.c_char_p),
    ]


class _ChatOpenParamsAbi(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("n_ctx", ctypes.c_int32),
        ("n_batch", ctypes.c_int32),
        ("n_ubatch", ctypes.c_int32),
        ("n_gpu_layers", ctypes.c_int32),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("chat_template", ctypes.c_char_p),
    ]


class _ChatGenerateParamsAbi(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_int32),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("min_p", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("repeat_last_n", ctypes.c_int32),
        ("seed", ctypes.c_uint32),
        ("stop", ctypes.POINTER(ctypes.c_char_p)),
        ("n_stop", ctypes.c_size_t),
        ("prefill_only", ctypes.c_bool),
    ]


def _chat_lib(lib_path: Optional[str] = None):
    """Load libcrispasr and declare every crispasr_chat_* signature."""
    lib = ctypes.CDLL(lib_path or _find_lib())
    if not hasattr(lib, "crispasr_chat_open"):
        raise RuntimeError("chat API not present in this libcrispasr build")

    err_p = ctypes.POINTER(_ChatErrorAbi)
    msg_p = ctypes.POINTER(_ChatMessageAbi)
    open_p = ctypes.POINTER(_ChatOpenParamsAbi)
    gen_p = ctypes.POINTER(_ChatGenerateParamsAbi)

    lib.crispasr_chat_open_params_default.argtypes = [open_p]
    lib.crispasr_chat_open_params_default.restype = None
    lib.crispasr_chat_generate_params_default.argtypes = [gen_p]
    lib.crispasr_chat_generate_params_default.restype = None

    lib.crispasr_chat_open.argtypes = [ctypes.c_char_p, open_p, err_p]
    lib.crispasr_chat_open.restype = ctypes.c_void_p
    lib.crispasr_chat_close.argtypes = [ctypes.c_void_p]
    lib.crispasr_chat_close.restype = None
    lib.crispasr_chat_reset.argtypes = [ctypes.c_void_p, err_p]
    lib.crispasr_chat_reset.restype = ctypes.c_int32

    # Returned as POINTER(c_char), not c_char_p: ctypes turns a c_char_p
    # restype into a bytes object and loses the pointer we have to hand back
    # to crispasr_chat_string_free.
    lib.crispasr_chat_generate.argtypes = [ctypes.c_void_p, msg_p, ctypes.c_size_t, gen_p, err_p]
    lib.crispasr_chat_generate.restype = ctypes.POINTER(ctypes.c_char)
    lib.crispasr_chat_string_free.argtypes = [ctypes.POINTER(ctypes.c_char)]
    lib.crispasr_chat_string_free.restype = None

    lib.crispasr_chat_generate_stream.argtypes = [
        ctypes.c_void_p, msg_p, ctypes.c_size_t, gen_p,
        ChatSession._ON_TOKEN_CB_TYPE, ctypes.c_void_p, err_p,
    ]
    lib.crispasr_chat_generate_stream.restype = ctypes.c_int32
    lib.crispasr_chat_set_abort_callback.argtypes = [
        ctypes.c_void_p, ChatSession._ABORT_CB_TYPE, ctypes.c_void_p,
    ]
    lib.crispasr_chat_set_abort_callback.restype = None

    lib.crispasr_chat_count_tokens.argtypes = [ctypes.c_void_p, msg_p, ctypes.c_size_t, err_p]
    lib.crispasr_chat_count_tokens.restype = ctypes.c_int32
    lib.crispasr_chat_memory_estimate.argtypes = [ctypes.c_char_p, open_p, err_p]
    lib.crispasr_chat_memory_estimate.restype = ctypes.c_size_t
    lib.crispasr_chat_n_ctx.argtypes = [ctypes.c_void_p]
    lib.crispasr_chat_n_ctx.restype = ctypes.c_int32
    lib.crispasr_chat_template_name.argtypes = [ctypes.c_void_p]
    lib.crispasr_chat_template_name.restype = ctypes.c_char_p
    lib.crispasr_chat_ai_disclosure_text.argtypes = []
    lib.crispasr_chat_ai_disclosure_text.restype = ctypes.c_char_p
    return lib


def _chat_raise(err: _ChatErrorAbi, fallback: str, code_hint: int = 0):
    """Raise the right exception for a filled crispasr_chat_error.

    The one-shot path signals failure by returning NULL, so there `err` is the
    only carrier; the streaming path also returns the code, passed as
    `code_hint`.
    """
    code = err.code if err.code != 0 else code_hint
    message = err.message.decode("utf-8", "replace") if err.message else ""
    if not message:
        message = fallback
    if code == CRISPASR_CHAT_ERR_ABORTED:
        raise ChatAborted(message)
    raise RuntimeError(message)


def _chat_cstr(value: str, field: str) -> bytes:
    """UTF-8 encode a string for the C ABI, rejecting an interior NUL.

    C reads a NUL as the end of the string, so a NUL inside a model path, a
    message, a stop sequence or a chat template would silently drop everything
    after it instead of being passed through — for a path, that means opening
    a different file from the one the caller named. Rust's ``CString::new``
    rejects the same input; this raises :class:`ValueError` naming the field.
    """
    data = value.encode("utf-8")
    if b"\x00" in data:
        raise ValueError(f"{field} contains an interior NUL byte, which C cannot carry")
    return data


def _chat_messages(messages) -> Tuple[ctypes.Array, list]:
    """Build the C message array. The returned list keeps the encoded bytes
    alive for as long as the caller holds it — the array only holds pointers."""
    encoded = []
    for i, m in enumerate(messages):
        if isinstance(m, ChatMessage):
            role, content = m.role, m.content
        else:
            role, content = m["role"], m["content"]
        encoded.append((_chat_cstr(role, f"messages[{i}].role"),
                        _chat_cstr(content, f"messages[{i}].content")))
    arr = (_ChatMessageAbi * len(encoded))()
    for i, (role_b, content_b) in enumerate(encoded):
        arr[i].role = role_b
        arr[i].content = content_b
    return arr, encoded


def _chat_open_params(
    lib, *,
    n_threads: Optional[int] = None,
    n_threads_batch: Optional[int] = None,
    n_ctx: Optional[int] = None,
    n_batch: Optional[int] = None,
    n_ubatch: Optional[int] = None,
    n_gpu_layers: Optional[int] = None,
    use_mmap: Optional[bool] = None,
    use_mlock: Optional[bool] = None,
    chat_template: Optional[str] = None,
) -> Tuple[_ChatOpenParamsAbi, list]:
    """ABI defaults with the fields the caller named applied over them, so no
    default is duplicated here. `None` means "leave the ABI's own value"."""
    params = _ChatOpenParamsAbi()
    lib.crispasr_chat_open_params_default(ctypes.byref(params))
    if n_threads is not None:
        params.n_threads = int(n_threads)
    if n_threads_batch is not None:
        params.n_threads_batch = int(n_threads_batch)
    if n_ctx is not None:
        params.n_ctx = int(n_ctx)
    if n_batch is not None:
        params.n_batch = int(n_batch)
    if n_ubatch is not None:
        params.n_ubatch = int(n_ubatch)
    if n_gpu_layers is not None:
        params.n_gpu_layers = int(n_gpu_layers)
    if use_mmap is not None:
        params.use_mmap = bool(use_mmap)
    if use_mlock is not None:
        params.use_mlock = bool(use_mlock)
    keep = []
    if chat_template is not None:
        tmpl = _chat_cstr(chat_template, "chat_template")
        keep.append(tmpl)
        params.chat_template = tmpl
    return params, keep


def _chat_generate_params(
    lib, *,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    prefill_only: Optional[bool] = None,
) -> Tuple[_ChatGenerateParamsAbi, list]:
    """As :func:`_chat_open_params`, for the per-call sampler settings."""
    params = _ChatGenerateParamsAbi()
    lib.crispasr_chat_generate_params_default(ctypes.byref(params))
    if max_tokens is not None:
        params.max_tokens = int(max_tokens)
    if temperature is not None:
        params.temperature = float(temperature)
    if top_k is not None:
        params.top_k = int(top_k)
    if top_p is not None:
        params.top_p = float(top_p)
    if min_p is not None:
        params.min_p = float(min_p)
    if repeat_penalty is not None:
        params.repeat_penalty = float(repeat_penalty)
    if repeat_last_n is not None:
        params.repeat_last_n = int(repeat_last_n)
    if seed is not None:
        params.seed = int(seed)
    if prefill_only is not None:
        params.prefill_only = bool(prefill_only)
    keep = []
    if stop:
        items = [_chat_cstr(s, f"stop[{i}]") for i, s in enumerate(stop)]
        arr = (ctypes.c_char_p * len(items))(*items)
        keep.append(items)
        keep.append(arr)
        params.stop = ctypes.cast(arr, ctypes.POINTER(ctypes.c_char_p))
        params.n_stop = len(items)
    return params, keep


class _ChatCallbackState:
    """Carries an exception a Python callback raised across the native call.

    ctypes prints and swallows an exception that escapes a callback, and
    letting one unwind through C is undefined behaviour either way, so the
    trampolines below stash it here and the calling method re-raises once the
    native call has returned.

    Once ``error`` is set neither callback calls into the caller's code again:
    a predicate that raised cannot be trusted to answer, and once the token
    callback has raised there is nobody left to hand chunks to. The abort
    trampoline then answers "stop" on its own, which is what turns either
    failure into a cancellation of the run.
    """
    __slots__ = ("error",)

    def __init__(self):
        self.error = None


class ChatSession:
    """Text → text chat / LLM session over a GGUF model.

    Usage::

        with crispasr.ChatSession("gemma-3-1b-it-Q4_K_M.gguf") as chat:
            msgs = [{"role": "user", "content": "Name three primes."}]
            print(chat.count_tokens(msgs), "of", chat.n_ctx, "prompt tokens")
            print(chat.generate(msgs))

    One call at a time per session, which crispasr_chat.h requires and this
    class enforces: a :meth:`generate` or :meth:`generate_stream` entered while
    another thread is inside one on the same session raises
    :class:`RuntimeError` instead of waiting. It is a diagnostic, not a queue —
    the intended pattern is one session per worker thread, and a caller who
    lands here has two threads sharing a session, which no amount of waiting
    turns into concurrency. :meth:`close` is the exception: it waits, since
    freeing the session under a running call is worse than a pause.

    Pass the WHOLE conversation in ``messages`` on every call: the session
    compares the templated prompt against the tokens it already holds and
    decodes only the new suffix. Passing just the latest turn is not wrong, it
    simply shares no prefix and re-prefills from scratch.

    Generation releases the GIL for its duration — ctypes drops the GIL around
    every call into a :class:`ctypes.CDLL`, which is how this library is
    loaded — so other Python threads run while the model decodes.
    ``on_token`` and ``should_continue`` re-acquire it for their own duration and
    both run on the calling thread.
    """

    # void(const char* utf8_chunk, void* user) and bool(void* user). Both are
    # built per call and held in the calling frame, which outlives the native
    # call it made: ctypes keeps no reference of its own, and a collected
    # trampoline is a dangling function pointer inside a running generation.
    # An instance attribute cannot hold them — it is one slot, so a second
    # call on the same session overwrites it and frees a pointer C is still
    # holding, and close() on another thread does the same.
    _ON_TOKEN_CB_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)
    _ABORT_CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

    def __init__(
        self, model_path: str, *,
        lib_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        chat_template: Optional[str] = None,
    ):
        """Open a GGUF chat model. Every parameter left as ``None`` keeps the
        value ``crispasr_chat_open_params_default`` supplies.

        ``chat_template`` overrides the template baked into the GGUF; ``None``
        reads ``tokenizer.chat_template`` from the model and falls back to
        "chatml".
        """
        # First, so close() from __del__ can take them even if the open below
        # fails.
        self._init_state()
        self._lib = _chat_lib(lib_path)
        params, _keep = _chat_open_params(
            self._lib, n_threads=n_threads, n_threads_batch=n_threads_batch,
            n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
            n_gpu_layers=n_gpu_layers, use_mmap=use_mmap, use_mlock=use_mlock,
            chat_template=chat_template,
        )
        err = _ChatErrorAbi()
        handle = self._lib.crispasr_chat_open(
            _chat_cstr(model_path, "model_path"), ctypes.byref(params), ctypes.byref(err))
        if not handle:
            _chat_raise(err, f"failed to open chat model {model_path!r}")
        self._handle = handle

    @property
    def n_ctx(self) -> int:
        """The session's context window, in tokens."""
        with self._use("n_ctx", closed_raises=False) as handle:
            return int(self._lib.crispasr_chat_n_ctx(handle))

    @property
    def template_name(self) -> str:
        """The chat template the session resolved against — "chatml",
        "llama3", "gemma", ..."""
        # C returns a pointer into the session's own string, valid only until
        # close — but the c_char_p restype makes ctypes copy the bytes into a
        # Python object as the call returns, which is inside the hold. `name`
        # is that copy, so decoding it outside costs nothing and races nothing.
        # A restype of POINTER(c_char) here would be a use-after-free.
        with self._use("template_name", closed_raises=False) as handle:
            name = self._lib.crispasr_chat_template_name(handle)
        return name.decode("utf-8") if name else ""

    def reset(self) -> None:
        """Clear the KV cache so the next generate re-prefills from scratch.
        Call it when starting a new conversation in a reused session — not
        after a :class:`ChatAborted`, which already flushed."""
        err = _ChatErrorAbi()
        with self._use("reset") as handle:
            rc = self._lib.crispasr_chat_reset(handle, ctypes.byref(err))
        if rc != 0:
            _chat_raise(err, "crispasr_chat_reset failed", rc)

    def count_tokens(self, messages) -> int:
        """Prompt tokens a FRESH session prefills for ``messages``.

        Counts the whole prompt — the template's control tokens, the leading
        BOS, and the trailing generation prompt — so it compares straight
        against :attr:`n_ctx`. An empty ``messages`` counts the template's
        own opening, which is whatever that template emits for no messages —
        template-dependent, and possibly nothing at all: several chat
        templates write only from inside their loop over the messages, and
        those return ``0``. Do not read a positive overhead into it. On a
        session part-way
        through a conversation it is an upper bound, since only the unheld
        suffix is re-decoded. A pure query: it touches neither the KV cache
        nor the history.
        """
        msgs, _keep = _chat_messages(messages)
        err = _ChatErrorAbi()
        with self._use("count_tokens") as handle:
            n = self._lib.crispasr_chat_count_tokens(
                handle, msgs, len(msgs), ctypes.byref(err))
        if n < 0:
            # A negative return is the failure sentinel, not an error code —
            # `err` is the only carrier here, so there is no hint to pass.
            _chat_raise(err, "crispasr_chat_count_tokens failed")
        return int(n)

    def generate(
        self, messages, *,
        should_continue: Optional[Callable[[], bool]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        prefill_only: Optional[bool] = None,
    ) -> str:
        """Apply the chat template to ``messages``, generate, return the reply.

        Sampler settings left as ``None`` keep the ABI's own defaults.
        ``max_tokens=0`` is not "generate nothing": the ABI reads any
        non-positive value as unset and applies its own default of 256 — pass
        ``prefill_only=True`` to suppress generation instead.
        ``should_continue`` is described on :meth:`generate_stream`; on abort
        this raises :class:`ChatAborted` and the partial text is discarded,
        since the one-shot path has nowhere to hand it back. Called on a
        session another thread is already generating on, it raises
        :class:`RuntimeError` without waiting — see the class docstring.
        """
        params, _keep = _chat_generate_params(
            self._lib, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, min_p=min_p,
            repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n,
            seed=seed, stop=stop, prefill_only=prefill_only)
        msgs, _msg_keep = _chat_messages(messages)
        err = _ChatErrorAbi()
        state = _ChatCallbackState()

        with self._use("generate") as handle:
            self._enter_call("generate")
            try:
                abort_cb = self._register_abort(handle, should_continue, state)
                try:
                    out = self._lib.crispasr_chat_generate(
                        handle, msgs, len(msgs), ctypes.byref(params), ctypes.byref(err))
                finally:
                    self._clear_abort(handle, should_continue)
                    # C no longer holds the pointer, so the trampoline may go.
                    del abort_cb
            finally:
                self._call_lock.release()
        # A callback that raised outranks whatever the native call reported.
        if state.error is not None:
            raise state.error
        if not out:
            _chat_raise(err, "crispasr_chat_generate failed")
        try:
            return ctypes.cast(out, ctypes.c_char_p).value.decode("utf-8", "replace")
        finally:
            self._lib.crispasr_chat_string_free(out)

    def generate_stream(
        self, messages, on_token: Callable[[str], None], *,
        should_continue: Optional[Callable[[], bool]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        repeat_last_n: Optional[int] = None,
        seed: Optional[int] = None,
        stop: Optional[List[str]] = None,
        prefill_only: Optional[bool] = None,
    ) -> None:
        """Generate, calling ``on_token(chunk)`` once per detokenised chunk.

        Concatenating the chunks yields exactly what :meth:`generate` returns
        for the same messages and params — except when a stop sequence ends
        the generation. The C side hands each piece to the callback before it
        scans for a match, so the chunk the match lands in has already been
        delivered, while the one-shot return value is truncated before the
        match. With ``stop`` set, the streamed text is therefore the one-shot
        text plus that last chunk, and a caller who wants the truncated form
        has to cut it back themselves.

        Every chunk is whole characters. A model that spells a character the
        tokeniser does not hold emits it one byte per token, so those bytes
        are buffered here and delivered once the character is complete: such
        a run of tokens produces ONE call rather than one per token, and the
        number of calls is therefore at most the number of tokens, not equal
        to it. If the generation stops part-way through a character the
        leftover bytes arrive as replacement characters in a final call,
        rather than being dropped.

        ``on_token`` runs on the calling thread for the duration of this call
        only, with the session mutex held — so like ``should_continue`` it must
        not call back into this session, which deadlocks. Called on a session
        another thread is already generating on, this raises
        :class:`RuntimeError` without waiting — see the class docstring.
        ``max_tokens=0`` behaves as on :meth:`generate`: it selects the ABI's
        default of 256, not "no tokens".

        **Abort polarity: ``should_continue()`` returns True to LET THE
        GENERATION CONTINUE** and False to abort it — a "may I keep going?"
        predicate. That is the polarity of ``crispasr_chat_abort_callback``
        in crispasr_chat.h, which in turn matches the encoder-begin callback
        on the ASR surface; this binding passes your answer to C as it is,
        without inverting it. It is called on the generating thread before
        each prompt batch and before each sampled token, and on the CPU
        backend additionally from inside a running compute graph, so keep it
        cheap and non-blocking. It must not call back into this session: the
        session mutex is held for the whole generation and re-entering
        deadlocks. On abort this raises :class:`ChatAborted`, having already
        delivered the partial text through ``on_token``, and the session is
        left reusable without a :meth:`reset`.

        An exception from ``on_token`` or ``should_continue`` never unwinds
        through C. It is captured, further chunks are dropped, and the
        exception is re-raised from this method once the native call has
        returned. If an abort predicate is registered the generation is also
        cancelled: from the next check onwards the predicate is no longer
        consulted and the answer is "stop". With no predicate registered
        there is no way to ask the ABI to stop, so the call runs to
        completion first.
        """
        params, _keep = _chat_generate_params(
            self._lib, max_tokens=max_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, min_p=min_p,
            repeat_penalty=repeat_penalty, repeat_last_n=repeat_last_n,
            seed=seed, stop=stop, prefill_only=prefill_only)
        msgs, _msg_keep = _chat_messages(messages)
        err = _ChatErrorAbi()
        state = _ChatCallbackState()

        # The C side hands over raw bytes, and a model that spells a character
        # the tokeniser does not hold emits it one byte per token — so a chunk
        # can end part-way through a character. The incremental decoder keeps
        # the unfinished tail for the next chunk and only substitutes a
        # replacement character for bytes no continuation could complete.
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        def _token_trampoline(chunk, _user):
            if state.error is not None:
                return
            try:
                text = decoder.decode(chunk if chunk else b"")
                if text:
                    on_token(text)
            except BaseException as e:  # re-raised after the native call returns
                state.error = e

        # A local, and only a local: this frame outlives the native call it is
        # handed to, which is exactly the span C may call through it.
        token_cb = ChatSession._ON_TOKEN_CB_TYPE(_token_trampoline)
        with self._use("generate_stream") as handle:
            self._enter_call("generate_stream")
            try:
                abort_cb = self._register_abort(handle, should_continue, state)
                try:
                    rc = self._lib.crispasr_chat_generate_stream(
                        handle, msgs, len(msgs), ctypes.byref(params),
                        token_cb, None, ctypes.byref(err))
                finally:
                    self._clear_abort(handle, should_continue)
                    # C no longer holds either pointer, so both trampolines may go.
                    del abort_cb
            finally:
                self._call_lock.release()
        # Bytes still buffered are a character the generation stopped in the
        # middle of; hand them over with replacement rather than lose output.
        # Aborted or not, they belong to the caller.
        if state.error is None:
            tail = decoder.decode(b"", final=True)
            if tail:
                try:
                    on_token(tail)
                except BaseException as e:
                    state.error = e
        if state.error is not None:
            raise state.error
        if rc != 0:
            _chat_raise(err, "crispasr_chat_generate_stream failed", rc)

    def _init_state(self) -> None:
        """The session state that exists before — and independently of — the
        native handle.

        ``_call_lock`` enforces one generation at a time. ``_lifetime`` is a
        separate, briefly-held lock counting the calls currently holding the
        handle, so :meth:`close` can wait for all of them, including the ones
        that take no part in ``_call_lock``.
        """
        self._call_lock = threading.Lock()
        self._lifetime = threading.Condition()
        self._in_use = 0
        self._handle = None

    @contextlib.contextmanager
    def _use(self, method: str, *, closed_raises: bool = True):
        """Hold the native handle for the span of one call.

        Every operation that hands ``self._handle`` to C goes through this,
        the properties and :meth:`count_tokens` included.

        ``crispasr_chat_close`` counts the calls that are already inside C and
        waits for them, so this is not what keeps a running generation's
        session alive — C does that. What C cannot see is the window between
        this method reading ``self._handle`` and the native call actually
        entering C: a close landing there finds nothing inside, frees, and the
        pending call goes through with a dangling pointer. Counting on the
        Python side covers that window, because the count is taken before the
        handle leaves this frame and released after the call returns.

        The lock is held only to read the handle and move the counter, never
        across the native call, so this neither serialises calls nor blocks a
        read-only accessor behind a running generation.

        With ``closed_raises=False`` a closed session yields NULL instead of
        raising, for the three read-only accessors whose C entry points answer
        a null session with the "nothing here" value.
        """
        with self._lifetime:
            handle = self._handle
            if not handle and closed_raises:
                raise RuntimeError(f"ChatSession.{method}: session is closed")
            if handle:
                self._in_use += 1
        try:
            yield handle
        finally:
            if handle:
                with self._lifetime:
                    self._in_use -= 1
                    if not self._in_use:
                        self._lifetime.notify_all()

    def _enter_call(self, method: str) -> None:
        """Claim the session for one native call, or say who already has it.

        Held for the span in which C holds pointers to this call's
        trampolines, so a second call cannot register over them and
        :meth:`close` cannot free the session under them. The acquire does not
        block: crispasr_chat.h allows one call at a time per session, so a
        thread arriving here is misusing the session and gets told which rule
        it broke, rather than a wait that looks like working concurrency.
        """
        if self._call_lock.acquire(blocking=False):
            return
        raise RuntimeError(
            f"ChatSession.{method}: this session is already running a call on "
            "another thread. crispasr_chat.h allows one call at a time per "
            "session — use one session per worker thread.")

    def _register_abort(self, handle, should_continue, state: _ChatCallbackState):
        """Register the abort predicate for one call, returning its trampoline.

        The caller holds the returned object until :meth:`_clear_abort` has
        run. It is a return value and not an instance attribute because an
        attribute is one slot: a second call assigning to it would drop this
        trampoline's last reference while C still held the pointer.

        Its answer goes to C as it is: True continues, False aborts, the same
        way round as ``crispasr_chat_abort_callback`` in crispasr_chat.h. The
        one thing this adds is the failure paths, which answer False without
        consulting the predicate at all — a predicate that raised cannot be
        asked again, and once the token callback has raised there is nobody
        left reading the output.
        """
        if should_continue is None:
            return None

        def _abort_trampoline(_user):
            if state.error is not None:
                return False
            try:
                return bool(should_continue())
            except BaseException as e:  # re-raised after the native call returns
                state.error = e
                return False

        abort_cb = ChatSession._ABORT_CB_TYPE(_abort_trampoline)
        self._lib.crispasr_chat_set_abort_callback(handle, abort_cb, None)
        return abort_cb

    def _clear_abort(self, handle, should_continue) -> None:
        if should_continue is None:
            return
        # A NULL instance of the callback type, not None: a function-pointer
        # argtype rejects None on newer CPython.
        self._lib.crispasr_chat_set_abort_callback(
            handle, ChatSession._ABORT_CB_TYPE(), None)

    @staticmethod
    def memory_estimate(
        model_path: str, *,
        lib_path: Optional[str] = None,
        n_threads: Optional[int] = None,
        n_threads_batch: Optional[int] = None,
        n_ctx: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_ubatch: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        chat_template: Optional[str] = None,
    ) -> int:
        """Conservative working set in bytes (weights + KV cache +
        activations) for a model on disk, reading its metadata but never its
        tensor data — a pre-flight guard for low-memory devices. The
        parameters matter mostly for ``n_ctx``, which sizes the KV term
        linearly; leave it ``None`` and the model's own trained context is
        used.

        The number is deliberately high, not approximate. The KV term bills
        both the K and the V cache at the full attention width ``n_embd``,
        but a grouped-query model gives each layer a K/V width that is a
        fraction of that: on Gemma 3 1B the KV term comes out 4.50× llama.cpp's
        real cache (117.00 MiB against 26.00 MiB at ``n_ctx`` 1024), which is
        1.33× on the whole estimate at ``n_ctx`` 4096. Over-reporting is the
        safe direction for a "will this fit?" guard: it can turn away a model
        that would just have fitted, and never admits one that would not.

        Raises :class:`RuntimeError` if the estimate could not be made — a
        model that could not be read is a failure, not a zero estimate.
        """
        lib = _chat_lib(lib_path)
        params, _keep = _chat_open_params(
            lib, n_threads=n_threads, n_threads_batch=n_threads_batch,
            n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
            n_gpu_layers=n_gpu_layers, use_mmap=use_mmap, use_mlock=use_mlock,
            chat_template=chat_template,
        )
        err = _ChatErrorAbi()
        n = lib.crispasr_chat_memory_estimate(
            _chat_cstr(model_path, "model_path"), ctypes.byref(params), ctypes.byref(err))
        if n == 0:
            _chat_raise(err, f"could not estimate memory for chat model {model_path!r}")
        return int(n)

    @staticmethod
    def ai_disclosure_text(*, lib_path: Optional[str] = None) -> str:
        """The canonical "you are talking to an AI" wording (EU AI Act Art.
        50(1)). Show it visibly, at or before the first turn."""
        lib = _chat_lib(lib_path)
        text = lib.crispasr_chat_ai_disclosure_text()
        return text.decode("utf-8") if text else ""

    def close(self) -> None:
        """Free the session and its KV cache. Idempotent.

        Unlike :meth:`generate`, this WAITS rather than raising: closing is the
        one thing a second thread legitimately wants to do to a busy session,
        and a shutdown path cannot usefully be told "try again" — nor may
        ``__exit__`` and ``__del__`` raise. A generation holds the session for
        as long as it decodes, so expect the wait to be that long; cancel first
        with ``should_continue`` if you need it to be shorter.

        The handle is retired before the wait, so a call arriving after this
        point is refused with :class:`RuntimeError` instead of joining the
        queue.

        The wait covers EVERY call holding the handle, not only the generate
        pair. ``crispasr_chat_close`` does its own waiting for the calls that
        have already reached C, but it cannot see one that has read the handle
        here and not yet entered; :meth:`count_tokens`, :meth:`reset` and the
        properties are all in that state for a moment, so they are counted and
        waited on here.

        Called from inside ``on_token`` or ``should_continue`` it deadlocks,
        like every other re-entry into a session mid-call.
        """
        with self._lifetime:
            handle, self._handle = self._handle, None
            if not handle:
                return
            while self._in_use:
                self._lifetime.wait()
            self._lib.crispasr_chat_close(handle)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
