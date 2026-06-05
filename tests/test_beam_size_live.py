#!/usr/bin/env python3
"""Live regression tests for session beam_size wiring (PLAN #90).

Verifies that crispasr_session_set_beam_size > 1 produces non-empty
transcription from the three backends newly wired in commit 0c24178e:
  - qwen3-asr
  - granite / granite-4.1
  - voxtral

Each test class is skipped when the required model file or binary is
absent, so this file stays green in any partial build environment.

Two test classes:

  TestBeamSizeSymbol   — static: crispasr_session_set_beam_size is
                         exported from the shared library and returns
                         -1 on a null handle.  No model, no network.

  TestBeamSizeLive     — live: opens a session for each backend with
                         beam_size=2, transcribes samples/jfk.wav,
                         asserts non-empty output.  Model files are
                         resolved via CRISPASR_{BACKEND}_MODEL env
                         vars or well-known local paths.

Run:
  python tests/test_beam_size_live.py
  pytest tests/test_beam_size_live.py -v
  CRISPASR_QWEN3_MODEL=/path/to/qwen3-asr.gguf pytest tests/test_beam_size_live.py -v
"""

import ctypes
import os
import subprocess
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]

# Prefer the Python binding so tests exercise the full session stack.
sys.path.insert(0, str(REPO / "python"))

BIN = os.environ.get(
    "CRISPASR_BIN",
    str(REPO / "build-ninja-compile" / "bin" / "crispasr"),
)
SAMPLE = str(REPO / "samples" / "jfk.wav")

_MODEL_DIR = Path("/Volumes/backups/ai/crispasr")

# Per-backend model resolution: env var > well-known path > None (skip).
def _model(env_var: str, filename: str) -> str:
    from_env = os.environ.get(env_var, "")
    if from_env and Path(from_env).exists():
        return from_env
    candidate = _MODEL_DIR / filename
    if candidate.exists():
        return str(candidate)
    return ""


QWEN3_MODEL = _model("CRISPASR_QWEN3_MODEL", "qwen3-asr-0.6b-q4_k.gguf")
GRANITE_MODEL = _model("CRISPASR_GRANITE_MODEL", "granite-speech-4.1-2b-q4_k.gguf")
VOXTRAL_MODEL = _model("CRISPASR_VOXTRAL_MODEL", "voxtral-mini-3b-2507-q4_k.gguf")

# Shared library path (same resolution as test_session_setters.py).
LIB_PATH = os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for _candidate in [
        REPO / "build-ninja-compile" / "src" / "libwhisper.dylib",
        REPO / "build-ninja-compile" / "src" / "libwhisper.so",
        REPO / "build" / "src" / "libwhisper.dylib",
        REPO / "build" / "src" / "libwhisper.so",
    ]:
        if _candidate.exists():
            LIB_PATH = str(_candidate)
            break


# ---------------------------------------------------------------------------
# Static: symbol export check
# ---------------------------------------------------------------------------


@unittest.skipUnless(LIB_PATH, "libcrispasr not built — set CRISPASR_LIB or build first")
class TestBeamSizeSymbol(unittest.TestCase):
    """crispasr_session_set_beam_size must be exported and return -1 on null."""

    @classmethod
    def setUpClass(cls):
        cls.lib = ctypes.CDLL(LIB_PATH)

    def test_symbol_exported(self):
        self.assertTrue(
            hasattr(self.lib, "crispasr_session_set_beam_size"),
            "crispasr_session_set_beam_size not found in shared library",
        )

    def test_null_handle_returns_neg1(self):
        fn = self.lib.crispasr_session_set_beam_size
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        self.assertEqual(fn(None, 2), -1)

    def test_beam_size_1_returns_neg1_on_null(self):
        fn = self.lib.crispasr_session_set_beam_size
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        self.assertEqual(fn(None, 1), -1)


# ---------------------------------------------------------------------------
# Live: beam_size=2 produces non-empty output for each newly-wired backend
# ---------------------------------------------------------------------------


@unittest.skipUnless(os.path.exists(BIN), f"crispasr binary not found at {BIN}")
@unittest.skipUnless(os.path.exists(SAMPLE), f"sample not found: {SAMPLE}")
class TestBeamSizeLiveCLI(unittest.TestCase):
    """CLI-level smoke test: --beam-size 2 does not crash and produces output."""

    def _run(self, backend: str, model: str, extra_args=()) -> str:
        cmd = [BIN, "--backend", backend, "-m", model,
               "--beam-size", "2", SAMPLE] + list(extra_args)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, f"crispasr failed:\n{r.stderr}")
        return (r.stdout + r.stderr).strip()

    @unittest.skipUnless(QWEN3_MODEL, "qwen3-asr model not found — set CRISPASR_QWEN3_MODEL")
    def test_qwen3_asr_beam2(self):
        out = self._run("qwen3", QWEN3_MODEL)
        self.assertTrue(len(out) > 0, "qwen3-asr beam=2 produced no output")

    @unittest.skipUnless(GRANITE_MODEL, "granite model not found — set CRISPASR_GRANITE_MODEL")
    def test_granite_beam2(self):
        out = self._run("granite-4.1", GRANITE_MODEL)
        self.assertTrue(len(out) > 0, "granite beam=2 produced no output")

    @unittest.skipUnless(VOXTRAL_MODEL, "voxtral model not found — set CRISPASR_VOXTRAL_MODEL")
    def test_voxtral_beam2(self):
        out = self._run("voxtral", VOXTRAL_MODEL)
        self.assertTrue(len(out) > 0, "voxtral beam=2 produced no output")


class TestBeamSizeLivePython(unittest.TestCase):
    """Python-binding smoke test: set_beam_size(2) does not crash and returns text."""

    def setUp(self):
        try:
            from crispasr import Session  # noqa: F401
            self._Session = Session
        except ImportError as exc:
            self.skipTest(f"crispasr Python package not importable: {exc}")
        if not os.path.exists(SAMPLE):
            self.skipTest(f"sample not found: {SAMPLE}")

    def _transcribe_with_beam(self, model: str, backend: str) -> str:
        import wave, array
        with wave.open(SAMPLE) as wf:
            raw = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
        samples = array.array("h", raw)
        pcm = [s / 32768.0 for s in samples]

        s = self._Session(model, backend=backend)
        try:
            s.set_beam_size(2)
            segs = s.transcribe(pcm, sample_rate=sr)
            return " ".join(seg.text for seg in segs).strip()
        finally:
            s.close()

    @unittest.skipUnless(QWEN3_MODEL, "qwen3-asr model not found")
    def test_qwen3_asr_beam2_python(self):
        text = self._transcribe_with_beam(QWEN3_MODEL, "qwen3")
        self.assertTrue(len(text) > 0, f"qwen3-asr beam=2 returned empty text (got {text!r})")

    @unittest.skipUnless(GRANITE_MODEL, "granite model not found")
    def test_granite_beam2_python(self):
        text = self._transcribe_with_beam(GRANITE_MODEL, "granite-4.1")
        self.assertTrue(len(text) > 0, f"granite beam=2 returned empty text (got {text!r})")

    @unittest.skipUnless(VOXTRAL_MODEL, "voxtral model not found")
    def test_voxtral_beam2_python(self):
        text = self._transcribe_with_beam(VOXTRAL_MODEL, "voxtral")
        self.assertTrue(len(text) > 0, f"voxtral beam=2 returned empty text (got {text!r})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
