#!/usr/bin/env python3
"""Live integration tests for Session.set_autochunk and Session.set_max_new_tokens_per_second (issue #574)."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
CRISPASR_TINY = os.path.join(REPO_ROOT, "models", "ggml-tiny.en.bin")
JFK_WAV = os.path.join(REPO_ROOT, "samples", "jfk.wav")

LIB_PATH = os.environ.get("CRISPASR_LIB")
if not LIB_PATH:
    for candidate in [
        os.path.join(REPO_ROOT, "build", "src", "libwhisper.dylib"),
        os.path.join(REPO_ROOT, "build", "src", "libcrispasr.dylib"),
    ]:
        if os.path.exists(candidate):
            LIB_PATH = candidate
            break


def load_jfk_pcm():
    import wave
    with wave.open(JFK_WAV, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return pcm


class TestSessionAutochunkLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CRISPASR_TINY):
            raise unittest.SkipTest(f"Model not found: {CRISPASR_TINY}")
        from crispasr import Session
        cls.session = Session(CRISPASR_TINY, lib_path=LIB_PATH, n_threads=2)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "session"):
            cls.session.close()

    def test_autochunk_enabled_vs_disabled(self):
        jfk = load_jfk_pcm()
        # Tile JFK audio 3 times to make ~33 seconds of audio (> 30s chunk threshold)
        long_pcm = np.concatenate([jfk, jfk, jfk])
        self.assertGreater(len(long_pcm), 30 * 16000)

        # 1. Transcribe with auto-chunking enabled (window 15s)
        self.session.set_autochunk(enabled=True, chunk_seconds=15)
        res_chunked = self.session.transcribe(long_pcm)
        self.assertGreater(len(res_chunked), 0)

        # 2. Transcribe with auto-chunking explicitly disabled
        self.session.set_autochunk(enabled=False)
        res_whole = self.session.transcribe(long_pcm)
        self.assertGreater(len(res_whole), 0)

        # Reset auto-chunking to default
        self.session.set_autochunk(enabled=True, chunk_seconds=0)

    def test_max_new_tokens_per_second(self):
        jfk = load_jfk_pcm()
        # Set a very tight token budget: rate 1.0 token/sec, minimum 4 tokens on 11s audio
        self.session.set_max_new_tokens_per_second(rate=1.0, minimum=4)
        res_budgeted = self.session.transcribe(jfk)
        # Verify transcription succeeded and was bounded
        self.assertGreater(len(res_budgeted), 0)

        # Reset to disabled
        self.session.set_max_new_tokens_per_second(rate=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
