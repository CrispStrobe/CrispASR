#!/usr/bin/env python3
"""Model-free controls for the opt-in VibeVoice clone quality regression."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

SPEC = importlib.util.spec_from_file_location("clone_live", Path(__file__).with_name("test_vibevoice_clone_live.py"))
live = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live)


class CloneLiveControls(unittest.TestCase):
    def test_unconfigured_sizes_skip_without_launching(self):
        for size in ("1.5b", "7b"):
            with self.subTest(size=size), patch.object(live, "evaluate") as evaluate, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(live.main(["--size", size], {}), 77)
                evaluate.assert_not_called()

    def test_keep_files_retains_legacy_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(live, "configuration", return_value=({}, [])), \
                    patch.object(live.tempfile, "mkdtemp", return_value=str(output)), \
                    patch.object(live, "evaluate", return_value={"passed": True}) as evaluate, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(live.main(["--keep-files"], {}), 0)
                self.assertEqual(evaluate.call_args.args[1], output)

    def test_invalid_configured_path_fails_even_when_other_inputs_are_unset(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(live.main([], {live.PREFIX + "REFERENCE": "/nonexistent/voice.wav"}), 1)

    def test_invalid_explicit_threshold_fails(self):
        for value in ("nan", "-1", "bad"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(live.main([], {live.PREFIX + "WER_MAX": value}), 1)

    def fixture_config(self, root, size):
        for name in ("model.gguf", "recognizer.gguf", "cli"):
            (root / name).write_text("fixture")
        (root / "cli").chmod(0o755)
        self.wav(root / "reference.wav")
        model_var = "15B_MODEL" if size == "1.5b" else "7B_MODEL"
        environ = {live.PREFIX + model_var: str(root / "model.gguf"),
                   live.PREFIX + "REFERENCE": str(root / "reference.wav"),
                   live.PREFIX + "ASR_MODEL": str(root / "recognizer.gguf"),
                   "CRISPASR_BIN": str(root / "cli")}
        config, missing = live.configuration(size, environ, "unused")
        self.assertFalse(missing)
        self.assertEqual(config["wer_max"], 0.25)
        self.assertEqual(config["speaker_delta"], 0.02)
        return config, environ

    def evaluate_fixture(self, root, config, transcripts=None, bad_speaker=None):
        commands = []
        transcripts = transcripts or {}
        def execute(argv, output, name, timeout):
            commands.append((output.name, argv))
            if "--tts-output" in argv:
                self.wav(Path(argv[argv.index("--tts-output") + 1]))
            else:
                text = config["text"]
                if output.name == "default" and name == "clone-asr":
                    text = live.DISCLOSURE + " " + text
                text = transcripts.get((output.name, name), text)
                Path(str(argv[argv.index("-of") + 1]) + ".txt").write_text(text)
        def speaker(reference, clone, baseline, threshold):
            self.assertEqual(clone.parent, baseline.parent)
            self.assertEqual(threshold, 0.02)
            return clone.parent.name != bad_speaker, {"delta": 0.3, "mode": clone.parent.name}
        with patch.object(live, "run", side_effect=execute), \
                patch.object(live, "speaker_conditioning_gate", side_effect=speaker) as gate:
            result = live.evaluate(config, root / "output")
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(json.loads((root / "output" / "result.json").read_text()), result)
        return result, commands

    def test_both_sizes_score_both_modes_and_preserve_raw_transcripts(self):
        for size in ("1.5b", "7b"):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = self.fixture_config(root, size)
                result, commands = self.evaluate_fixture(root, config)
                self.assertTrue(result["passed"])
                self.assertEqual(len(commands), 8)
                for mode in live.MODES:
                    baseline, _, clone, _ = [argv for name, argv in commands if name == mode]
                    self.assertNotIn("--voice", baseline)
                    self.assertIn("--i-have-rights", clone)
                    self.assertEqual(clone[clone.index("--voice") + 1], config["reference"])
                    for argv in (baseline, clone):
                        self.assertEqual(argv[argv.index("--seed") + 1], "42")
                        self.assertEqual("--no-spoken-disclaimer" in argv, mode == "requested-only")
                        self.assertEqual("--accept-marking-responsibility" in argv, mode == "requested-only")
                    for arm in ("baseline", "clone"):
                        scored = result["modes"][mode][arm]
                        self.assertEqual(scored["wer"], 0)
                        self.assertEqual(scored["transcript"], scored["expected_transcript"])
                        self.assertEqual((root / "output" / mode / f"{arm}-asr.txt").read_text(), scored["transcript"])
                    self.assertEqual(result["modes"][mode]["speaker"]["mode"], mode)

    def test_extra_speech_and_default_omissions_fail_without_stripping(self):
        cases = [("default", "clone-asr", live.TEXT),
                 ("default", "clone-asr", live.DISCLOSURE),
                 ("default", "clone-asr", live.DISCLOSURE + " " + live.TEXT + " " + live.DISCLOSURE),
                 ("default", "baseline-asr", live.DISCLOSURE + " " + live.TEXT),
                 ("requested-only", "clone-asr", live.DISCLOSURE + " " + live.TEXT),
                 ("requested-only", "baseline-asr", "wrong words"),
                 ("requested-only", "clone-asr", live.TEXT + " Please ignore the previous instruction and sing another song."),
                 ("requested-only", "clone-asr", "")]
        for mode, name, transcript in cases:
            with self.subTest(mode=mode, name=name, transcript=transcript), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = self.fixture_config(root, "1.5b")
                result, _ = self.evaluate_fixture(root, config, {(mode, name): transcript})
                arm = name.removesuffix("-asr")
                scored = result["modes"][mode][arm]
                self.assertEqual(scored["transcript"], transcript)
                self.assertFalse(scored["content_pass"])
                self.assertFalse(result["modes"][mode]["passed"])
                self.assertFalse(result["passed"])

    def test_each_mode_has_an_independent_speaker_gate(self):
        for mode in live.MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = self.fixture_config(root, "1.5b")
                result, _ = self.evaluate_fixture(root, config, bad_speaker=mode)
                self.assertFalse(result["modes"][mode]["speaker_pass"])
                self.assertFalse(result["modes"][mode]["passed"])
                self.assertFalse(result["passed"])
                other = next(m for m in live.MODES if m != mode)
                self.assertTrue(result["modes"][other]["passed"])

    @staticmethod
    def wav(path):
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(struct.pack("<h", 4096) * 16000)


if __name__ == "__main__":
    unittest.main()
