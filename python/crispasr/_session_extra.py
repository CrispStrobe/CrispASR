"""Additional shared-session operations. All returned native views are copied."""
import ctypes as C
from dataclasses import dataclass
from functools import wraps

import numpy as np

P = C.c_void_p
F = C.POINTER(C.c_float)
I = C.POINTER(C.c_int)
S = C.c_char_p


def function(lib, name, args, result):
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise RuntimeError(f"{name} is unavailable; rebuild libcrispasr") from None
    fn.argtypes, fn.restype = args, result
    return fn


def mono(pcm):
    arr = np.asarray(pcm)
    if arr.ndim != 1:
        raise ValueError("expected a one-dimensional mono PCM array")
    if arr.size > 2147483647:
        raise ValueError("PCM exceeds the C API sample limit")
    return np.ascontiguousarray(arr, dtype=np.float32)


def pointer(arr):
    return arr.ctypes.data_as(F)


def copy_view(ptr, shape, dtype=np.float32):
    if any(n < 0 for n in shape):
        raise RuntimeError("native result has a negative dimension")
    if not all(shape):
        return np.empty(shape, dtype=dtype)
    if not ptr:
        raise RuntimeError("native result has a count but no data")
    return np.ctypeslib.as_array(ptr, shape=(int(np.prod(shape)),)).copy().reshape(shape)


@dataclass
class PianoResult:
    """Notes (onset_ms, offset_ms, MIDI pitch, velocity), parallel GM programs."""
    notes: np.ndarray
    programs: np.ndarray
    sample_rate: int


@dataclass
class ChordResult:
    """Spans (start_ms, end_ms, label, confidence) and corresponding names."""
    spans: np.ndarray
    names: list
    vocabulary_size: int


@dataclass
class BeatResult:
    """Events (time_s, is_downbeat), tempo in BPM and native input rate."""
    events: np.ndarray
    tempo_bpm: float
    sample_rate: int


@dataclass
class TabResult:
    """Log probabilities [frame,string,class]; these are not decoded fingering."""
    emissions: np.ndarray
    silent_class: int
    frame_period: float
    open_midi: tuple


def session_operation(fn):
    """Reject use after close and session re-entry from synchronous callbacks."""
    @wraps(fn)
    def checked(self, *args, **kwargs):
        if not getattr(self, "_handle", None):
            raise RuntimeError("session is closed")
        if getattr(self, "_inside_callback", False):
            raise RuntimeError("do not re-enter the session from its callback")
        return fn(self, *args, **kwargs)
    return checked


class SessionExtras:
    def _extra(self, suffix, args=(), result=C.c_int):
        if not getattr(self, "_handle", None):
            raise RuntimeError("session is closed")
        return function(self._lib, "crispasr_session_" + suffix, [P, *args], result)

    def _query(self, suffix, result=C.c_int):
        return self._extra(suffix, result=result)(self._handle)

    def _run_pcm(self, task, pcm, sample_rate=None):
        arr = mono(pcm)
        args, values = [F, C.c_int], [pointer(arr), arr.size]
        if sample_rate is not None:
            if int(sample_rate) <= 0:
                raise ValueError("sample_rate must be positive")
            args.append(C.c_int)
            values.append(int(sample_rate))
        rc = self._extra(task, args)(self._handle, *values)
        if rc < 0:
            raise RuntimeError(f"{task} failed (rc={rc})")
        return rc

    def _view(self, suffix, width):
        n = C.c_int()
        ptr = self._extra(suffix, [I], F)(self._handle, C.byref(n))
        return copy_view(ptr, (n.value, width))

    def piano(self, pcm) -> PianoResult:
        """Transcribe mono PCM at piano_sample_rate(); empty notes are valid."""
        self._run_pcm("piano", pcm)
        notes = self._view("piano_notes", 4)
        n = C.c_int()
        ptr = self._extra("piano_note_programs", [I], C.POINTER(C.c_int))(
            self._handle, C.byref(n))
        if n.value != len(notes):
            raise RuntimeError("native piano notes/programs length mismatch")
        return PianoResult(notes, copy_view(ptr, (n.value,), np.int32), self.piano_sample_rate())

    def chords(self, pcm, sample_rate=22050) -> ChordResult:
        """Recognize chords; native code resamples the supplied sample rate."""
        self._run_pcm("chords", pcm, sample_rate)
        spans = self._view("chords_spans", 4)
        name = self._extra("chords_span_name", [C.c_int], S)
        names = [(name(self._handle, i) or b"").decode("utf-8") for i in range(len(spans))]
        return ChordResult(spans, names, self.chords_vocab_size())

    def beats(self, pcm, sample_rate=None) -> BeatResult:
        """Track beats at the native rate; mismatched rates are rejected."""
        rate = self.beats_sample_rate()
        if sample_rate is None:
            sample_rate = rate
        self._run_pcm("beats", pcm, sample_rate)
        return BeatResult(self._view("beats_events", 2), self.beats_tempo_bpm(), rate)

    def tab(self, pcm, sample_rate=22050) -> TabResult:
        """Return tablature log probabilities; callers supply fingering decoding."""
        self._run_pcm("tab", pcm, sample_rate)
        n, strings, classes = C.c_int(), C.c_int(), C.c_int()
        ptr = self._extra("tab_emissions", [I, I, I], F)(
            self._handle, C.byref(n), C.byref(strings), C.byref(classes))
        emissions = copy_view(ptr, (n.value, strings.value, classes.value))
        get_midi = self._extra("tab_string_open_midi", [C.c_int])
        return TabResult(emissions, self.tab_silent_class(), self.tab_frame_period(),
                         tuple(get_midi(self._handle, i) for i in range(strings.value)))

    def convert(self, content, f0_hz, speaker_id=0, *, noise_zp=None, noise_sine=None):
        """Convert frame-major ContentVec features to owned mono PCM.

        Content must be [frames, convert_content_dim()]. The caller supplies the
        content encoder. Noise defaults to native random draws. Replay buffers
        must have shapes [convert_inter_channels(), frames] and
        [frames, convert_upsample_product()] respectively. Both are optional.
        """
        content = np.ascontiguousarray(content, dtype=np.float32)
        f0 = mono(f0_hz)
        dim, speakers = self.convert_content_dim(), self.convert_n_speakers()
        if content.ndim != 2 or content.shape != (len(f0), dim) or not len(f0):
            raise ValueError(f"content must have shape (len(f0_hz), {dim}) with nonempty frames")
        if not 0 <= int(speaker_id) < speakers:
            raise ValueError("speaker_id is outside the model's speaker bank")
        def noise_array(value, shape):
            if value is None:
                return None
            arr = np.ascontiguousarray(value, dtype=np.float32)
            if arr.shape != shape or any(n <= 0 for n in shape):
                raise ValueError(f"replay noise must have shape {shape}")
            return arr
        zp = noise_array(noise_zp, (self.convert_inter_channels(), len(f0))) if noise_zp is not None else None
        sine = noise_array(noise_sine, (len(f0), self.convert_upsample_product())) if noise_sine is not None else None
        rc = self._extra("convert", [F, C.c_int, F, C.c_int, F, F])(
            self._handle, pointer(content), len(f0), pointer(f0), int(speaker_id),
            None if zp is None else pointer(zp), None if sine is None else pointer(sine))
        if rc <= 0:
            raise RuntimeError(f"convert failed (rc={rc})")
        return self._view("convert_audio", 1).reshape(-1)

    def set_voice_samples(self, pcm, sample_rate, ref_text=None):
        """Install mono reference PCM; native consent/marking rules apply."""
        arr = mono(pcm)
        if not len(arr) or int(sample_rate) <= 0:
            raise ValueError("voice samples and a positive sample rate are required")
        rc = self._extra("set_voice_samples", [F, C.c_int32, C.c_int32, S])(
            self._handle, pointer(arr), len(arr), int(sample_rate),
            None if ref_text is None else ref_text.encode("utf-8"))
        if rc != 0:
            raise RuntimeError(f"set_voice_samples failed (rc={rc})")

    def tada_set_makeref_models(self, encoder=None, aligner=None):
        """Configure TADA reference companions; None clears an override."""
        rc = self._extra("tada_set_makeref_models", [S, S])(
            self._handle, None if encoder is None else str(encoder).encode(),
            None if aligner is None else str(aligner).encode())
        if rc != 0:
            raise RuntimeError(f"tada_set_makeref_models failed (rc={rc})")

    def set_tts_cfg_scale(self, scale):
        """Set the backend's TTS CFG scale (distinct from cfg_weight)."""
        rc = self._extra("set_tts_cfg_scale", [C.c_float])(self._handle, float(scale))
        if rc != 0:
            raise RuntimeError(f"set_tts_cfg_scale failed (rc={rc})")

    def last_synth_error(self):
        return (self._query("last_synth_error", S) or b"").decode("utf-8")

    def disclaimer_text(self):
        return (function(self._lib, "crispasr_session_disclaimer_text", [], S)() or b"").decode("utf-8")

    def get_disclaimer_pcm(self):
        """Get owned disclosure audio BEFORE installing a cloning voice."""
        n = C.c_int()
        free = function(self._lib, "crispasr_pcm_free", [F], None)
        ptr = self._extra("get_disclaimer_pcm", [I], F)(self._handle, C.byref(n))
        try:
            if not ptr or n.value <= 0:
                raise RuntimeError(self.last_synth_error() or "no disclaimer audio")
            return copy_view(ptr, (n.value,))
        finally:
            if ptr:
                free(ptr)

    def synthesize_streaming(self, text, callback):
        """Call callback(owned_pcm, is_final) synchronously per sentence.

        This is sentence streaming, not incremental vocoder output. Exceptions
        are re-raised after the native call returns; callbacks must not re-enter
        or close this session. A failed native audio chunk raises an error even
        when the native streaming dispatcher reports success.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")
        cb_type = C.CFUNCTYPE(None, F, C.c_int, C.c_int, P)
        errors = []

        def deliver(ptr, n, final, _):
            if errors:
                return
            try:
                if not ptr or n <= 0:
                    raise RuntimeError(self.last_synth_error() or "streaming synthesis returned an empty chunk")
                self._inside_callback = True
                try:
                    callback(copy_view(ptr, (n,)), bool(final))
                finally:
                    self._inside_callback = False
            except BaseException as exc:
                errors.append(exc)
        cb = cb_type(deliver)
        rc = self._extra("synthesize_streaming", [S, cb_type, P])(
            self._handle, text.encode("utf-8"), cb, None)
        if errors:
            raise errors[0]
        if rc != 0:
            raise RuntimeError(f"synthesize_streaming failed (rc={rc})")

    def _set_callback(self, kind, callback, cb_type, adapt):
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        holders = getattr(self, "_extra_callback_holders", {})
        self._extra_callback_holders = holders

        def deliver(*args):
            if getattr(self, "_extra_callback_error", None) is not None:
                return
            try:
                self._inside_callback = True
                try:
                    callback(*adapt(args))
                finally:
                    self._inside_callback = False
            except BaseException as exc:
                self._extra_callback_error = exc
        cb = cb_type(deliver) if callback is not None else cb_type()
        self._extra("set_" + kind + "_callback", [cb_type, P], None)(self._handle, cb, None)
        holders[kind] = cb

    def raise_callback_error(self):
        """Re-raise a saved callback exception after native inference returns."""
        exc = getattr(self, "_extra_callback_error", None)
        self._extra_callback_error = None
        if exc is not None:
            raise exc

    def set_progress_callback(self, callback):
        """callback(processed_samples, total_samples), or None to clear.

        Only native chunked paths report progress. Runs synchronously; callback
        exceptions are transported back to the inference caller.
        """
        cb_type = C.CFUNCTYPE(None, C.c_int, C.c_int, P)
        self._set_callback("progress", callback, cb_type, lambda a: (a[0], a[1]))

    def set_segment_callback(self, callback):
        """callback(text, start_seconds, end_seconds, index), or None to clear.

        Runs synchronously during inference. Do not re-enter/close the session.
        """
        cb_type = C.CFUNCTYPE(None, S, C.c_int64, C.c_int64, C.c_int, P)
        self._set_callback("segment", callback, cb_type,
                           lambda a: ((a[0] or b"").decode("utf-8"), a[1]/100, a[2]/100, a[3]))

    def set_token_callback(self, callback):
        """callback(raw_utf8_bytes, index), preserving split UTF-8 BPE pieces."""
        cb_type = C.CFUNCTYPE(None, S, C.c_int, P)
        self._set_callback("token", callback, cb_type, lambda a: (a[0] or b"", a[1]))

    def reset_progress(self):
        function(self._lib, "crispasr_reset_progress", [], None)()

    def get_streamed_segment_count(self):
        return function(self._lib, "crispasr_get_streamed_segment_count", [], C.c_int)()

    def get_streamed_token_count(self):
        return function(self._lib, "crispasr_get_streamed_token_count", [], C.c_int)()

    def reset_streamed_segments(self):
        function(self._lib, "crispasr_reset_streamed_segments", [], None)()

    def reset_streamed_tokens(self):
        function(self._lib, "crispasr_reset_streamed_tokens", [], None)()

    def drain_streamed_segments(self):
        """Drain the globally selected last-streaming session, not necessarily self."""
        free = function(self._lib, "crispasr_session_result_free", [P], None)
        result = function(self._lib, "crispasr_drain_streamed_segments", [], P)()
        if not result:
            return []
        try:
            return self._read_result(result)
        finally:
            free(result)

    def drain_streamed_tokens(self):
        """Drain raw token bytes from the globally selected last-streaming session."""
        n = C.c_int()
        ptr = function(self._lib, "crispasr_drain_streamed_tokens", [I], P)(C.byref(n))
        if n.value < 0 or (n.value and not ptr):
            raise RuntimeError("invalid native token buffer")
        out = []
        for _ in range(n.value):
            token = C.string_at(ptr)
            out.append(token)
            ptr += len(token) + 1
        return out


# Explicit symbol table keeps the header census aware of generated query methods.
_QUERIES = {
    "input_channels": ("crispasr_session_input_channels", C.c_int),
    "output_channels": ("crispasr_session_output_channels", C.c_int),
    "pitch_n_frames": ("crispasr_session_pitch_n_frames", C.c_int),
    "pitch_sample_rate": ("crispasr_session_pitch_sample_rate", C.c_int),
    "convert_inter_channels": ("crispasr_session_convert_inter_channels", C.c_int),
    "convert_upsample_product": ("crispasr_session_convert_upsample_product", C.c_int),
    "convert_content_dim": ("crispasr_session_convert_content_dim", C.c_int),
    "convert_n_speakers": ("crispasr_session_convert_n_speakers", C.c_int),
    "convert_sample_rate": ("crispasr_session_convert_sample_rate", C.c_int),
    "tab_n_frames": ("crispasr_session_tab_n_frames", C.c_int),
    "tab_silent_class": ("crispasr_session_tab_silent_class", C.c_int),
    "tab_frame_period": ("crispasr_session_tab_frame_period", C.c_float),
    "chords_n_spans": ("crispasr_session_chords_n_spans", C.c_int),
    "chords_vocab_size": ("crispasr_session_chords_vocab_size", C.c_int),
    "beats_n_events": ("crispasr_session_beats_n_events", C.c_int),
    "beats_tempo_bpm": ("crispasr_session_beats_tempo_bpm", C.c_float),
    "beats_sample_rate": ("crispasr_session_beats_sample_rate", C.c_int),
    "piano_n_notes": ("crispasr_session_piano_n_notes", C.c_int),
    "piano_sample_rate": ("crispasr_session_piano_sample_rate", C.c_int),
    "separate_sample_rate": ("crispasr_session_separate_sample_rate", C.c_int),
}


def _make_query(symbol, restype):
    def query(self):
        return self._query(symbol.removeprefix("crispasr_session_"), restype)
    query.__doc__ = f"Query {symbol}; returns the native value without unit conversion."
    return query


for _name, (_symbol, _type) in _QUERIES.items():
    setattr(SessionExtras, _name, _make_query(_symbol, _type))


# Apply the same lifecycle contract to generated queries and explicit methods.
for _name, _value in list(vars(SessionExtras).items()):
    if not _name.startswith("_") and _name != "raise_callback_error" and callable(_value):
        setattr(SessionExtras, _name, session_operation(_value))
