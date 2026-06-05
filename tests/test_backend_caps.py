#!/usr/bin/env python3
"""Capability-matrix regression tests (PLAN #74b).

Two test classes:

  TestCapabilityJSON   — static: crispasr --list-backends-json must declare
                         translate / src-tgt-language / voice-cloning for the
                         known set of backends.  No model, no network.

  TestTranslateLive    — live: runs --translate -l de on samples/jfk.wav and
                         asserts non-empty output.  Skipped when the binary or
                         model file is absent.

Run:
  python tests/test_backend_caps.py
  pytest tests/test_backend_caps.py -v
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BIN = os.environ.get(
    "CRISPASR_BIN",
    str(REPO / "build-ninja-compile" / "bin" / "crispasr"),
)
SAMPLE = str(REPO / "samples" / "jfk.wav")

# ---------------------------------------------------------------------------
# Ground-truth caps for backends we care about in this sweep.
# A backend listed here must declare ALL of the named caps; unlisted backends
# are ignored so new additions don't break existing tests.
# ---------------------------------------------------------------------------

_TRANSLATE_BACKENDS = {
    "whisper",
    "canary",
    "granite",
    "granite-4.1",
    "granite-4.1-plus",
    "voxtral",
    "qwen3",
    "qwen3-1.7b",
    "mega-asr",
    "m2m100",
    "m2m100-wmt21",
    "madlad",
    "gemma4-e2b",
}

_SRC_TGT_BACKENDS = {
    "canary",
    "granite",
    "granite-4.1",
    "granite-4.1-plus",
    "voxtral",
    "qwen3",
    "qwen3-1.7b",
    "mega-asr",
    "m2m100",
    "m2m100-wmt21",
    "madlad",
    "gemma4-e2b",
}

_VOICE_CLONING_BACKENDS = {
    "chatterbox",
    "chatterbox-turbo",
    "kartoffelbox-turbo",
    "lahgtna-chatterbox",
    "vibevoice-1.5b",
    "indextts",
    "voxcpm2-tts",
    "qwen3-tts",
    "qwen3-tts-1.7b-base",
}

# Backends that must NOT declare voice-cloning (preset-speaker, not reference-WAV cloning).
_NO_VOICE_CLONING_BACKENDS = {
    "qwen3-tts-customvoice",
    "qwen3-tts-1.7b-customvoice",
    "qwen3-tts-1.7b-voicedesign",
    "vibevoice",
}


@unittest.skipUnless(os.path.exists(BIN), f"crispasr binary not found at {BIN} — set CRISPASR_BIN or build first")
class TestCapabilityJSON(unittest.TestCase):
    """crispasr --list-backends-json must declare translate / voice-cloning caps correctly."""

    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            [BIN, "--list-backends-json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        cls.data = json.loads(result.stdout)
        cls.by_name = {b["name"]: set(b.get("caps", [])) for b in cls.data["backends"]}

    def test_translate_backends_declare_translate(self):
        for name in _TRANSLATE_BACKENDS:
            if name not in self.by_name:
                continue
            self.assertIn("translate", self.by_name[name], f"{name} missing 'translate' cap")

    def test_src_tgt_backends_declare_src_tgt_language(self):
        for name in _SRC_TGT_BACKENDS:
            if name not in self.by_name:
                continue
            self.assertIn("src-tgt-language", self.by_name[name], f"{name} missing 'src-tgt-language' cap")

    def test_voice_cloning_backends_declare_voice_cloning(self):
        for name in _VOICE_CLONING_BACKENDS:
            if name not in self.by_name:
                continue
            self.assertIn("voice-cloning", self.by_name[name], f"{name} missing 'voice-cloning' cap")

    def test_no_voice_cloning_backends_omit_voice_cloning(self):
        for name in _NO_VOICE_CLONING_BACKENDS:
            if name not in self.by_name:
                continue
            self.assertNotIn("voice-cloning", self.by_name[name],
                             f"{name} must NOT declare 'voice-cloning' (preset-speaker, not reference-WAV)")

    def test_whisper_does_not_declare_src_tgt_language(self):
        if "whisper" not in self.by_name:
            return
        self.assertNotIn("src-tgt-language", self.by_name["whisper"],
                         "whisper uses --language for target; src-tgt-language is for separate -sl/-tl flags")


@unittest.skipUnless(os.path.exists(BIN), f"crispasr binary not found — set CRISPASR_BIN or build first")
@unittest.skipUnless(os.path.exists(SAMPLE), f"sample file not found: {SAMPLE}")
class TestTranslateLive(unittest.TestCase):
    """Live translation smoke-test — skipped when model files are absent."""

    _WHISPER_MODEL = os.environ.get(
        "CRISPASR_WHISPER_MODEL",
        "/Volumes/backups/ai/crispasr/ggml-tiny.bin",
    )

    @unittest.skipUnless(
        os.path.exists(os.environ.get("CRISPASR_WHISPER_MODEL", "/Volumes/backups/ai/crispasr/ggml-tiny.bin")),
        "whisper-tiny model not found — set CRISPASR_WHISPER_MODEL",
    )
    def test_whisper_translate_to_german(self):
        result = subprocess.run(
            [BIN, "--backend", "whisper", "-m", self._WHISPER_MODEL,
             "--translate", "-l", "de", SAMPLE],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"crispasr failed:\n{result.stderr}")
        combined = result.stdout + result.stderr
        self.assertTrue(len(combined.strip()) > 0, "translate produced no output")


if __name__ == "__main__":
    unittest.main(verbosity=2)
