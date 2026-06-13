"""CrispASR Python wrapper via ctypes.

Provides speech-to-text transcription using ggml inference.
Wraps the whisper.h C API from crispasr / CrispASR.
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
    """Locate the crispasr / whisper shared library.

    The wheel is pure-Python and does not bundle the native library —
    matches crispasr's binding pattern. The user is expected to have
    `libcrispasr.{so,dylib,dll}` on their system, either installed by
    a package manager (Homebrew, apt) or built from source.

    Probe order:
      1. $CRISPASR_LIB_PATH (explicit override — full path to the .so/.dylib/.dll)
      2. sys.prefix/lib (pip install --user, virtualenv, conda)
      3. Standard install prefixes (Homebrew arm64/x64, /usr/local, /usr)
      4. Repo-relative `build/` paths (for `pip install -e .` from a clone)
      5. The bare filename (lets the loader use $LD_LIBRARY_PATH /
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
        return override

    search = [
        Path(sys.prefix) / "lib",
        Path("/opt/homebrew/lib"),  # macOS arm64 Homebrew
        Path("/usr/local/lib"),     # macOS x64 Homebrew, /usr/local installs
        Path("/usr/lib"),           # apt, dnf
        Path("/usr/lib/x86_64-linux-gnu"),  # Debian/Ubuntu multiarch
        Path("/usr/lib/aarch64-linux-gnu"),
        # Repo-relative — last so `pip install crispasr` doesn't accidentally
        # pick up an old build/ from cwd.
        Path(__file__).parent,
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
                return str(p)
    # Fall back to bare name; ctypes will use the system loader path.
    return candidates[0]


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


# =========================================================================
# Diarization (shared C-ABI, 0.4.5+)
# =========================================================================

class DiarizeMethod:
    """Diarization method identifiers matching the C-ABI enum."""
    ENERGY = 0      # stereo only
    XCORR = 1       # stereo only
    VAD_TURNS = 2   # mono-friendly, timing-based
    PYANNOTE = 3    # mono-friendly, GGUF pyannote seg model


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


def registry_lookup(backend: str, *, lib_path: Optional[str] = None) -> Optional[RegistryEntry]:
    """Look up the canonical GGUF for a backend. Returns ``None`` on miss."""
    return _registry_call("crispasr_registry_lookup_abi", backend, lib_path)


def registry_lookup_by_filename(filename: str, *, lib_path: Optional[str] = None) -> Optional[RegistryEntry]:
    """Look up the canonical GGUF by filename (exact, then fuzzy substring)."""
    return _registry_call("crispasr_registry_lookup_by_filename_abi", filename, lib_path)


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
    lib_path: Optional[str] = None,
) -> bool:
    """Assign a speaker index to each of ``segs``, mutating in place.

    Four methods — see :class:`DiarizeMethod`. ``left`` is mono PCM for
    mono-only methods, otherwise the left channel of a stereo pair.
    All PCM is 16 kHz float32. Returns ``True`` on success; only
    ``PYANNOTE`` can fail (model load failure).
    """
    if not segs or left is None or len(left) == 0:
        return True

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

    # ABI structs must match crispasr_c_api.cpp.
    class _SegAbi(ctypes.Structure):
        _fields_ = [
            ("t0_cs", ctypes.c_int64),
            ("t1_cs", ctypes.c_int64),
            ("speaker", ctypes.c_int32),
            ("_pad", ctypes.c_int32),
        ]

    class _OptsAbi(ctypes.Structure):
        _fields_ = [
            ("method", ctypes.c_int32),
            ("n_threads", ctypes.c_int32),
            ("slice_t0_cs", ctypes.c_int64),
            ("pyannote_model_path", ctypes.c_char_p),
        ]

    seg_array = (_SegAbi * len(segs))()
    for i, s in enumerate(segs):
        seg_array[i].t0_cs = int(round(s.t0 * 100))
        seg_array[i].t1_cs = int(round(s.t1 * 100))
        seg_array[i].speaker = s.speaker
        seg_array[i]._pad = 0

    opts = _OptsAbi(
        method=int(method),
        n_threads=int(n_threads),
        slice_t0_cs=int(round(slice_t0 * 100)),
        pyannote_model_path=(pyannote_model_path.encode("utf-8")
                             if pyannote_model_path else None),
    )

    rc = lib.crispasr_diarize_segments_abi(
        left_ptr, right_ptr, int(len(left_np)), 1 if is_stereo else 0,
        ctypes.byref(seg_array), len(segs), ctypes.byref(opts),
    )
    if rc == 0:
        for i, s in enumerate(segs):
            s.speaker = int(seg_array[i].speaker)
    return rc == 0


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

    def __init__(self, model_path: str, lib_path: Optional[str] = None,
                 n_threads: int = 4, backend: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
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
        lib.crispasr_session_available_backends.argtypes = [ctypes.c_char_p, ctypes.c_int]
        lib.crispasr_session_available_backends.restype = ctypes.c_int
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
        # 0.5.13: per-word top-N alternative candidates.
        if hasattr(lib, "crispasr_session_result_word_n_alts"):
            lib.crispasr_session_result_word_n_alts.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            lib.crispasr_session_result_word_n_alts.restype = ctypes.c_int
        if hasattr(lib, "crispasr_session_result_word_alt_p"):
            lib.crispasr_session_result_word_alt_p.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.crispasr_session_result_word_alt_p.restype = ctypes.c_float
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
        lib.crispasr_session_available_backends(buf, 256)
        csv = buf.value.decode("utf-8")
        return [s.strip() for s in csv.split(",") if s.strip()]

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
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

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
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

    # ---------------------------------------------------------------------
    # TTS synthesis (vibevoice, qwen3-tts, kokoro, orpheus, chatterbox, outetts, indextts, voxcpm2, csm, dia, zonos-tts, bark, speecht5, parler-tts, pocket-tts, kugelaudio, tada, lfm2-audio)
    # ---------------------------------------------------------------------

    def set_codec_path(self, path: str) -> None:
        """Load a separate codec GGUF.

        Required for qwen3-tts (12 Hz tokenizer) and orpheus (SNAC
        codec); no-op for other backends.
        """
        if not hasattr(self._lib, "crispasr_session_set_codec_path"):
            raise RuntimeError("TTS API not present in this libcrispasr build")
        self._lib.crispasr_session_set_codec_path.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.crispasr_session_set_codec_path.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_codec_path(self._handle, path.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_codec_path failed (rc={rc}) for backend {self.backend!r}")

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
        """Set the natural-language voice description (qwen3-tts VoiceDesign).

        VoiceDesign generates speech in a voice **described by a
        natural-language instruction** — no reference WAV, no preset
        speaker. The instruct text is wrapped as
        ``"<|im_start|>user\\n{instruct}<|im_end|>\\n"`` and prepended
        to the talker prefill; the codec bridge omits the speaker
        frame entirely.

        Required for qwen3-tts VoiceDesign before
        :meth:`synthesize`. Re-callable; latest call wins. Raises if
        the active backend isn't VoiceDesign.

        Detect VoiceDesign via :meth:`is_voice_design`.
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
        (canary, cohere, parakeet, moonshine). Other backends silently no-op.

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
        """Set the diffusion / CFM step count for diffusion-based TTS backends.

        Soft no-op (rc=-2) when the active backend has no diffusion stage.
        """
        if not hasattr(self._lib, "crispasr_session_set_tts_steps"):
            return
        self._lib.crispasr_session_set_tts_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.crispasr_session_set_tts_steps.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_tts_steps(self._handle, int(steps))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_tts_steps failed (rc={rc})")

    def set_top_p(self, top_p: float) -> None:
        """Set the top-p nucleus-sampling threshold. Honoured by chatterbox."""
        if not hasattr(self._lib, "crispasr_session_set_top_p"):
            return
        self._lib.crispasr_session_set_top_p.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self._lib.crispasr_session_set_top_p.restype = ctypes.c_int
        rc = self._lib.crispasr_session_set_top_p(self._handle, float(top_p))
        if rc != 0 and rc != -2:
            raise RuntimeError(f"set_top_p failed (rc={rc})")

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
        mimo-asr, moss-audio, lfm2-audio, mini-omni2. For moss-audio this
        enables audio understanding beyond ASR (e.g. "Describe the sounds
        in this clip." or "What language is spoken?").
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
        ``csm``, ``dia``, ``fastpitch``, ``speecht5``, ``melotts``, ``piper``,
        ``parler-tts``, ``outetts``, ``cosyvoice3-tts``, ``pocket-tts``,
        ``f5-tts``, ``bark``, ``kugelaudio``, ``tada``, ``lfm2-audio``.
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

    def speech_to_speech(self, input_pcm: "np.ndarray", language: str = None) -> tuple:
        """Speech-to-speech: audio in → audio out via a single model pass.

        Supported on backends with S2S capability (``lfm2-audio``,
        ``mini-omni2``).  Input is 16 kHz mono float32 PCM.  Returns a
        tuple ``(output_pcm, transcript)`` where *output_pcm* is a
        float32 numpy array at the backend's TTS sample rate (typically
        24 kHz) and *transcript* is the intermediate ASR text (may be
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
    """File-based speaker profile database for speaker identification."""

    def __init__(self, dir_path: str, lib_path: str = None):
        self._lib = ctypes.CDLL(lib_path or _find_lib())
        self._lib.crispasr_speaker_db_load.argtypes = [ctypes.c_char_p]
        self._lib.crispasr_speaker_db_load.restype = ctypes.c_void_p
        self._lib.crispasr_speaker_db_free.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_db_free.restype = None
        self._lib.crispasr_speaker_db_count.argtypes = [ctypes.c_void_p]
        self._lib.crispasr_speaker_db_count.restype = ctypes.c_int32
        self._lib.crispasr_speaker_db_match.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
            ctypes.c_float, ctypes.c_char_p, ctypes.c_int32,
        ]
        self._lib.crispasr_speaker_db_match.restype = ctypes.c_float
        self._lib.crispasr_speaker_db_enroll.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
        ]
        self._lib.crispasr_speaker_db_enroll.restype = ctypes.c_int32
        self._db = self._lib.crispasr_speaker_db_load(dir_path.encode())
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
        """Enroll a speaker with the given name and embedding."""
        import numpy as np
        emb = np.ascontiguousarray(embedding, dtype=np.float32)
        rc = self._lib.crispasr_speaker_db_enroll(
            self._dir.encode(), name.encode(),
            emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(emb),
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
    if rc != 0:
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


def watermark_embed(pcm: "numpy.ndarray", alpha: float = 0.005) -> None:
    """Embed an AI-generated watermark into float32 PCM in-place."""
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
