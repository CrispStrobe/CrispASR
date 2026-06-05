"""Integration tests for qwen3-tts CustomVoice --instruct style control (issue #91).

These tests require a CustomVoice GGUF + codec GGUF to be present on disk. They
are skipped automatically when the models are not available, so they integrate
cleanly into any CI run without needing model downloads.

The fast vs slow speech test verifies that the style instruction actually changes
the output — specifically that "speak very fast" produces fewer codec frames than
"speak very slowly", which is the most reliable measurable proxy for speech rate
in a codec-token model.

Run locally (requires model files):
    QWT_CV_MODEL=/path/to/qwen3-tts-1.7b-customvoice-q4_k.gguf \
    QWT_CODEC_MODEL=/path/to/qwen3-tts-tokenizer-12hz-q4_k.gguf \
    python -m unittest tests/test_qwen3_tts_cv_style.py -v

Or just:
    python -m unittest tests/test_qwen3_tts_cv_style.py -v
(will skip with a clear message if models are absent)
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Model discovery — check env vars first, then common local paths.
# ---------------------------------------------------------------------------
_SEARCH_DIRS = [
    Path("/mnt/storage/models"),
    Path("/mnt/storage"),
    Path.home() / "models",
    Path("."),
]

_CV_PATTERNS = [
    "qwen3-tts-1.7b-customvoice-q4_k.gguf",
    "qwen3-tts-1.7b-customvoice*.gguf",
    "qwen3-tts-customvoice-1.7b*.gguf",
]
_CODEC_PATTERNS = [
    "qwen3-tts-tokenizer-12hz-q4_k.gguf",
    "qwen3-tts-tokenizer*.gguf",
    "qwen3-tts-codec*.gguf",
]


def _find_gguf(patterns: list[str], env_var: str) -> Path | None:
    if v := os.environ.get(env_var):
        p = Path(v)
        return p if p.exists() else None
    for d in _SEARCH_DIRS:
        for pat in patterns:
            for hit in d.glob(pat):
                return hit
    return None


_CV_MODEL = _find_gguf(_CV_PATTERNS, "QWT_CV_MODEL")
_CODEC_MODEL = _find_gguf(_CODEC_PATTERNS, "QWT_CODEC_MODEL")
_SAMPLE = Path("samples/jfk.wav")

# The CLI binary.
_CLI = Path("build/bin/crispasr")
if not _CLI.exists():
    _CLI = Path("build-shared/bin/crispasr")


def _models_available() -> bool:
    return bool(_CV_MODEL and _CV_MODEL.exists() and
                _CODEC_MODEL and _CODEC_MODEL.exists() and
                _CLI.exists() and _SAMPLE.exists())


_SKIP_REASON = (
    f"CustomVoice GGUF not found (set QWT_CV_MODEL / QWT_CODEC_MODEL or place "
    f"qwen3-tts-1.7b-customvoice-q4_k.gguf + codec under /mnt/storage/models). "
    f"cv={_CV_MODEL}, codec={_CODEC_MODEL}, cli={_CLI}"
)


def _run_cv(speaker: str, instruct: str | None, text: str, extra_flags: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run crispasr with qwen3-tts CustomVoice and return the completed process."""
    cmd = [
        str(_CLI),
        "-m", str(_CV_MODEL),
        "--codec-model", str(_CODEC_MODEL),
        "--backend", "qwen3-tts",
        "--voice", speaker,
        "-t", text,
    ]
    if instruct:
        cmd += ["--instruct", instruct]
    if extra_flags:
        cmd += extra_flags
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300)


class TestCVStyleNullGuard(unittest.TestCase):
    """API-contract tests that run without a model (always executed)."""

    def test_cv_style_instruct_api_in_header(self):
        """Verify qwen3_tts_set_cv_style_instruct is declared in qwen3_tts.h."""
        header = Path("src/qwen3_tts.h")
        self.assertTrue(header.exists(), "src/qwen3_tts.h not found")
        content = header.read_text()
        self.assertIn("qwen3_tts_set_cv_style_instruct", content,
                      "qwen3_tts_set_cv_style_instruct missing from qwen3_tts.h")

    def test_cli_passes_instruct_to_cv(self):
        """Verify the CLI backend file wires --instruct for CustomVoice."""
        backend_cpp = Path("examples/cli/crispasr_backend_qwen3_tts.cpp")
        self.assertTrue(backend_cpp.exists(), "CLI backend source not found")
        content = backend_cpp.read_text()
        self.assertIn("qwen3_tts_set_cv_style_instruct", content,
                      "CLI backend does not call qwen3_tts_set_cv_style_instruct; "
                      "fix from issue #91 may not be wired up")


@unittest.skipUnless(_models_available(), _SKIP_REASON)
class TestCVStyleIntegration(unittest.TestCase):
    """End-to-end tests that require the CustomVoice GGUF + codec."""

    _TEXT = "Ask not what your country can do for you."

    def test_synthesis_without_instruct_succeeds(self):
        """Baseline: synthesis with a speaker but no style instruct completes."""
        r = _run_cv("EN-US", None, self._TEXT)
        self.assertEqual(r.returncode, 0,
                         f"CV synthesis failed (no instruct).\nstderr:\n{r.stderr}")

    def test_synthesis_with_instruct_succeeds(self):
        """Synthesis with a style instruct must not error out (issue #91 fix)."""
        r = _run_cv("EN-US", "spoke with a calm and measured pace", self._TEXT)
        self.assertEqual(r.returncode, 0,
                         f"CV synthesis with --instruct failed (issue #91).\nstderr:\n{r.stderr}")

    def test_fast_speech_fewer_frames_than_slow(self):
        """'spoke very fast' should produce fewer codec frames than 'spoke very slowly'.

        This is the primary measurable proxy for speech rate: faster speech
        → shorter duration → fewer codec tokens at 12 Hz. The test reads
        the 'crispasr: n_codes' line from stderr to extract the frame count.

        If the model ignores the instruct, both runs produce roughly the same
        frame count and the test fails — which is exactly what would have
        happened before the issue #91 fix.
        """
        def codec_frames(instruct: str) -> int:
            r = _run_cv("EN-US", instruct, self._TEXT)
            self.assertEqual(r.returncode, 0,
                             f"CV synthesis failed for instruct='{instruct}'.\nstderr:\n{r.stderr}")
            for line in (r.stdout + r.stderr).splitlines():
                if "n_codes" in line or "codec_frames" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if "n_codes" in p or "codec_frames" in p:
                            # expect "n_codes=NNN" or "codec_frames NNN"
                            if "=" in p:
                                try:
                                    return int(p.split("=")[-1])
                                except ValueError:
                                    pass
                            elif i + 1 < len(parts):
                                try:
                                    return int(parts[i + 1])
                                except ValueError:
                                    pass
            self.fail(
                f"Could not parse codec frame count from output for instruct='{instruct}'.\n"
                f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
            )

        fast_frames = codec_frames("spoke very fast and with high energy")
        slow_frames = codec_frames("spoke very slowly and deliberately")
        self.assertLess(
            fast_frames, slow_frames,
            f"Expected fast speech ({fast_frames} frames) < slow speech ({slow_frames} frames). "
            f"The model may be ignoring --instruct for CustomVoice (issue #91 regression)."
        )


if __name__ == "__main__":
    unittest.main()
