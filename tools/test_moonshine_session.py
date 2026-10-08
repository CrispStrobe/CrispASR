#!/usr/bin/env python3
"""Replay mono 16 kHz WAV through the native C API (requires numpy).

Usage: python tools/test_moonshine_session.py LIBRARY MODEL WAV
Checks batch recognition, packet-size independence, idempotent flush, and
rejection of audio after flush. Text accuracy needs a human reference.
"""

import ctypes as C, wave, numpy as np, time, json, pathlib, sys

if len(sys.argv) != 4:
    raise SystemExit("usage: test_moonshine_session.py LIBRARY MODEL WAV")
lib = C.CDLL(sys.argv[1])
P = C.c_void_p
I = C.c_int
F = C.POINTER(C.c_float)
S = C.c_char_p
spec = {
    "crispasr_session_open": (P, [S, I]),
    "crispasr_session_open_explicit": (P, [S, S, I]),
    "crispasr_session_close": (None, [P]),
    "crispasr_session_transcribe_lang": (P, [P, F, I, S]),
    "crispasr_session_result_n_segments": (I, [P]),
    "crispasr_session_result_segment_text": (S, [P, I]),
    "crispasr_session_result_free": (None, [P]),
    "crispasr_session_stream_kind": (I, [P]),
    "crispasr_session_stream_open": (P, [P, I, I, I, I, S, I]),
    "crispasr_stream_feed": (I, [P, F, I]),
    "crispasr_stream_flush": (I, [P]),
    "crispasr_stream_get_text": (
        I,
        [P, P, I, C.POINTER(C.c_double), C.POINTER(C.c_double), C.POINTER(C.c_int64)],
    ),
    "crispasr_stream_close": (None, [P]),
}
for n, (r, a) in spec.items():
    fn = getattr(lib, n)
    fn.restype = r
    fn.argtypes = a
w = wave.open(sys.argv[3])
assert w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
audio = (
    np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    / 32768
)
path = sys.argv[2]
s = lib.crispasr_session_open(path.encode(), 3)
assert s, "auto open failed"
kind = lib.crispasr_session_stream_kind(s)
print("kind", kind, flush=True)
t = time.monotonic()
r = lib.crispasr_session_transcribe_lang(s, audio.ctypes.data_as(F), len(audio), b"de")
assert r, "batch failed"
batch = " ".join(
    lib.crispasr_session_result_segment_text(r, i).decode()
    for i in range(lib.crispasr_session_result_n_segments(r))
)
lib.crispasr_session_result_free(r)
print("batch", time.monotonic() - t, batch, flush=True)
if kind == 2:
    results = []
    for packet in [1777, 5120]:
        stream = lib.crispasr_session_stream_open(s, 3, 640, 20000, 0, b"de", 0)
        assert stream
        t = time.monotonic()
        buf = C.create_string_buffer(16384)
        count = C.c_int64()
        a = C.c_double()
        b = C.c_double()
        drafts = []
        for off in range(0, len(audio), packet):
            x = audio[off : off + packet]
            assert lib.crispasr_stream_feed(stream, x.ctypes.data_as(F), len(x)) >= 0
            lib.crispasr_stream_get_text(
                stream, buf, len(buf), C.byref(a), C.byref(b), C.byref(count)
            )
            text = buf.value.decode()
            if text and (not drafts or text != drafts[-1]):
                drafts.append(text)
        assert lib.crispasr_stream_flush(stream) >= 0
        assert lib.crispasr_stream_flush(stream) >= 0
        lib.crispasr_stream_get_text(
            stream, buf, len(buf), C.byref(a), C.byref(b), C.byref(count)
        )
        result = buf.value.decode()
        results.append(result)
        assert lib.crispasr_stream_feed(stream, audio.ctypes.data_as(F), 1) < 0
        print("stream", packet, time.monotonic() - t, len(drafts), result, flush=True)
        lib.crispasr_stream_close(stream)
    assert results[0] == results[1], results
    print("PARTITION PARITY PASS", flush=True)
lib.crispasr_session_close(s)
