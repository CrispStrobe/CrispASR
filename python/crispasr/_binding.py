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
    """Find the crispasr / whisper shared library.

    As of CrispASR 0.4.0 the build produces both `libcrispasr.*`
    (preferred — all 10 backends linked) and the historical
    `libwhisper.*` (alias). We probe candidates in order so any build
    layout is picked up transparently.
    """
    system = platform.system()
    if system == "Darwin":
        candidates = ["libcrispasr.dylib", "libwhisper.dylib"]
    elif system == "Windows":
        candidates = ["crispasr.dll", "whisper.dll"]
    else:
        candidates = ["libcrispasr.so", "libwhisper.so"]

    search = [
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
    return candidates[0]


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

        # 0.4.2 — VAD + tdrz param setters on whisper_full_params.
        for _sym, _argtypes in [
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
        lib.crispasr_session_result_free.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_result_free.restype = None
        lib.crispasr_session_close.argtypes = [ctypes.c_void_p]
        lib.crispasr_session_close.restype = None

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
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
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
        """Transcribe with Silero VAD segmentation + whisper.cpp-style stitching.

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
                for j in range(wn):
                    wt = self._lib.crispasr_session_result_word_text(res, i, j)
                    words.append(SessionWord(
                        text=wt.decode("utf-8") if wt else "",
                        start=self._lib.crispasr_session_result_word_t0(res, i, j) / 100.0,
                        end=self._lib.crispasr_session_result_word_t1(res, i, j) / 100.0,
                    ))
                out.append(SessionSegment(text=text.strip(), start=t0, end=t1, words=words))
            return out
        finally:
            self._lib.crispasr_session_result_free(res)

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
