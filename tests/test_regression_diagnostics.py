"""Model-free controls for regression failure evidence retention."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_one", ROOT / "tests/regression/run_one.py")
run_one = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_one)


class VibeVoiceBoundaryTests(unittest.TestCase):
    def test_voice_diff_rejects_scale_and_shape_errors_and_missing_stages(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location("voice_diff", ROOT / "tools/diff_vibevoice_tts.py")
        diff = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(diff)
        for defect in ("none", "scale", "shape", "missing"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                ref, native = Path(temporary) / "ref", Path(temporary) / "native"
                ref.mkdir()
                native.mkdir()
                for ref_name, native_name, _, _ in diff.VOICE_STAGES:
                    np.asarray([1, 2, 3], dtype="<f4").tofile(ref / f"{ref_name}.bin")
                    if defect != "missing":
                        values = [2, 4, 6] if defect == "scale" else [1, 2] if defect == "shape" else [1, 2, 3]
                        np.asarray(values, dtype="<f4").tofile(native / f"{native_name}.bin")
                with patch.object(sys, "argv", ["diff", str(ref), str(native), "--voice-only"]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(diff.main(), 0 if defect == "none" else 1)


class SpeakerConditioningTests(unittest.TestCase):
    def test_requested_speaker_gate_cannot_be_advisory(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            commands = []
            def execute(argv, **kwargs):
                commands.append(argv)
                if "--tts-output" in argv:
                    Path(argv[argv.index("--tts-output") + 1]).write_bytes(b"RIFF" + b"0" * 2048)
                return subprocess.CompletedProcess(argv, 0, "The quick brown fox", "")
            manifest = {"backends": [{"name": "asr", "backend_id": "whisper",
                                      "gguf": {"repo": "r", "file": "asr", "revision": "p"}}],
                        "tts_backends": [{"name": "clone", "gguf": {"repo": "r", "file": "tts", "revision": "p"},
                                          "voice": {"repo": "r", "file": "ref.wav", "revision": "p"},
                                          "tts_phrase": "The quick brown fox", "roundtrip_asr_backend": "asr",
                                          "wer_max": 0, "advisory": True,
                                          "tts_extra_args": ["--backend", "original"],
                                          "speaker_conditioning": {"baseline_backend": "generic"}}]}
            with patch.object(run_one, "hf_download", side_effect=lambda repo, file, revision, directory: work / file), \
                    patch.object(run_one, "_run_captured", side_effect=execute), \
                    patch.object(run_one, "speaker_conditioning_gate", return_value=(False, {"delta": 0})), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_one.tts_roundtrip_for("clone", manifest, work, Path("cli")), 1)
            self.assertNotIn("--voice", commands[-1])
            self.assertEqual(commands[-1].count("--backend"), 1)
            self.assertEqual(commands[-1][commands[-1].index("--backend") + 1], "generic")

    def test_ignored_voice_fails_even_when_output_is_non_silent(self):
        self._speaker_control([1.0, 0.0], [1.0, 0.0], False)

    def test_cloning_must_move_toward_reference(self):
        self._speaker_control([1.0, 0.0], [0.0, 1.0], True)

    def test_silent_reference_fails_closed(self):
        self._speaker_control([1.0, 0.0], [0.0, 1.0], False, silent=True)

    def _speaker_control(self, cloned, baseline, expected, silent=False):
        import types
        import numpy as np
        values = iter(np.asarray(v, dtype=np.float32) for v in ([1.0, 0.0], cloned, baseline))
        fake = types.SimpleNamespace(
            VoiceEncoder=lambda device: types.SimpleNamespace(embed_utterance=lambda audio: next(values)),
            preprocess_wav=lambda path: np.full(8000, 0.0 if silent else 0.1, dtype=np.float32))
        with patch.dict(sys.modules, {"resemblyzer": fake}):
            passed, _ = run_one.speaker_conditioning_gate(Path("r"), Path("c"), Path("b"))
        self.assertEqual(passed, expected)


class RegressionDiagnosticsTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "fake CLI uses a POSIX shebang and chmod")
    def test_failed_tts_roundtrip_keeps_commands_and_wav_after_work_cleanup(self):
        self._roundtrip_control(successful=False, keep_wav=False)

    @unittest.skipUnless(os.name == "posix", "fake CLI uses a POSIX shebang and chmod")
    def test_successful_roundtrip_keeps_wav_only_when_explicitly_requested(self):
        for keep_wav in (False, True):
            with self.subTest(keep_wav=keep_wav):
                self._roundtrip_control(successful=True, keep_wav=keep_wav)

    def _roundtrip_control(self, *, successful, keep_wav):
        with tempfile.TemporaryDirectory(prefix="regression-diagnostics-test-") as temp:
            root = Path(temp)
            work = root / "work"
            diag = root / "diagnostics"
            argv_log = root / "actual-argv.jsonl"
            fake_cli = root / "fake-crispasr"
            fake_cli.write_text("""#!/usr/bin/env python3
import json, os, pathlib, sys
with open(os.environ['FAKE_ARGV_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv) + '\\n')
if '--tts-output' in sys.argv:
    out = pathlib.Path(sys.argv[sys.argv.index('--tts-output') + 1])
    out.write_bytes(b'RIFF' + b'\\0' * 2048)
    print('synth complete')
else:
    print(os.environ['FAKE_TRANSCRIPT'])
""")
            fake_cli.chmod(0o755)
            manifest = {
                "fixtures": {"repo": "unused", "revision": "unused"},
                "backends": [{
                    "name": "asr", "backend_id": "parakeet",
                    "gguf": {"repo": "asr-repo", "file": "asr.gguf", "revision": "pin"},
                }],
                "tts_backends": [{
                    "name": "csm-1b", "gguf": {"repo": "tts-repo", "file": "tts.gguf", "revision": "pin"},
                    "no_voice": True, "roundtrip_asr_backend": "asr",
                    "tts_phrase": "The quick brown fox", "tts_seed": 1234, "wer_max": 0.15,
                }],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))

            def fake_download(repo, filename, revision, dest):
                path = dest / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake model bytes")
                return path

            env = {
                "REGRESSION_MANIFEST": str(manifest_path),
                "CRISPASR_BIN": str(fake_cli),
                "WORK_DIR": str(work),
                "REGRESSION_DIAGNOSTICS_DIR": str(diag),
                "FAKE_ARGV_LOG": str(argv_log),
                "KEEP_WORK": "0",
                "REGRESSION_KEEP_TTS_WAV": "1" if keep_wav else "0",
                "FAKE_TRANSCRIPT": "The quick brown fox" if successful else "the grip brown fox",
            }
            with patch.dict(os.environ, env), patch.object(sys, "argv", ["run_one.py", "csm-1b"]), \
                    patch.object(run_one, "hf_download", side_effect=fake_download), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_one.main(), 0 if successful else 1)

            self.assertFalse(work.exists())
            actual_argv = [json.loads(line) for line in argv_log.read_text().splitlines()]
            synth = json.loads((diag / "tts-synth.json").read_text())
            asr = json.loads((diag / "roundtrip-asr.json").read_text())
            self.assertEqual(synth["argv"], actual_argv[0])
            self.assertEqual(asr["argv"], actual_argv[1])
            self.assertEqual(synth["argv"][1:3], ["-m", str(work / "tts.gguf")])
            self.assertEqual(synth["argv"][-3:], ["--seed", "1234", "--no-prints"])
            self.assertEqual(asr["argv"][1:3], ["--backend", "parakeet"])
            self.assertEqual(synth["returncode"], 0)
            self.assertEqual(asr["returncode"], 0)
            expected = ["roundtrip-asr.json", "tts-synth.json"]
            if not successful or keep_wav:
                self.assertEqual((diag / "tts_csm-1b.wav").read_bytes(), b"RIFF" + b"\0" * 2048)
                expected.append("tts_csm-1b.wav")
            self.assertEqual(sorted(p.name for p in diag.iterdir()), expected)

    @unittest.skipUnless(os.name == "posix", "fake CLI uses a POSIX shebang and chmod")
    def test_quantized_and_f16_transcripts_keep_separate_evidence(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict(os.environ, {"REGRESSION_DIAGNOSTICS_DIR": temp}), \
                patch.object(run_one.subprocess, "run", side_effect=[
                    subprocess.CompletedProcess([], 0, stdout="quantized words\n", stderr="q4"),
                    subprocess.CompletedProcess([], 0, stdout="reference words\n", stderr="f16")]):
            run_one.run_transcript(Path("cli"), Path("q4.gguf"), Path("audio.wav"))
            run_one.run_transcript(Path("cli"), Path("f16.gguf"), Path("audio.wav"), label="transcript-f16")
            q4 = json.loads((Path(temp) / "transcript.json").read_text())
            f16 = json.loads((Path(temp) / "transcript-f16.json").read_text())
            self.assertIn("quantized words", q4["stdout"]["text"])
            self.assertIn("reference words", f16["stdout"]["text"])

    def test_command_log_is_bounded_and_retains_failure_status(self):
        with tempfile.TemporaryDirectory(prefix="regression-diagnostics-test-") as temp:
            root = Path(temp)
            fake_cli = root / "fail-cli"
            fake_cli.write_text("#!/usr/bin/env python3\nimport sys\nsys.stderr.write('x' * 600000)\nsys.exit(7)\n")
            fake_cli.chmod(0o755)
            diag = root / "diagnostics"
            with patch.dict(os.environ, {"REGRESSION_DIAGNOSTICS_DIR": str(diag)}):
                proc = run_one._run_captured([str(fake_cli), "--probe"],
                                             check=False, timeout=5, label="failure")
            self.assertEqual(proc.returncode, 7)
            evidence = json.loads((diag / "failure.json").read_text())
            self.assertEqual(evidence["argv"], [str(fake_cli), "--probe"])
            self.assertEqual(evidence["returncode"], 7)
            self.assertTrue(evidence["stderr"]["truncated"])
            self.assertEqual(evidence["stderr"]["bytes"], 600000)
            self.assertLessEqual(len(evidence["stderr"]["text"].encode()), run_one._LOG_LIMIT)

    def test_malformed_utf8_replacement_stays_under_encoded_byte_cap(self):
        evidence = run_one._bounded_output(b"\xff" * (run_one._LOG_LIMIT // 2))
        self.assertEqual(evidence["bytes"], run_one._LOG_LIMIT // 2)
        self.assertTrue(evidence["truncated"])
        self.assertTrue(evidence["text"].endswith("\ufffd"))
        self.assertLessEqual(len(evidence["text"].encode("utf-8")), run_one._LOG_LIMIT)

    def test_diagnostics_inside_work_dir_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="regression-diagnostics-test-") as temp:
            root = Path(temp)
            work = root / "work"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"backends": [{"name": "fake", "skip_diff": True}]}))
            env = {
                "REGRESSION_MANIFEST": str(manifest),
                "CRISPASR_BIN": str(Path(__file__)),
                "WORK_DIR": str(work),
                "REGRESSION_DIAGNOSTICS_DIR": str(work / "evidence"),
                "KEEP_WORK": "0",
            }
            errors = io.StringIO()
            with patch.dict(os.environ, env), patch.object(sys, "argv", ["run_one.py", "fake"]), \
                    contextlib.redirect_stderr(errors):
                with self.assertRaises(SystemExit):
                    run_one.main()
            self.assertIn("must be outside WORK_DIR", errors.getvalue())
            self.assertFalse(work.exists())

    def test_timeout_keeps_partial_stdout_and_stderr(self):
        with tempfile.TemporaryDirectory(prefix="regression-diagnostics-test-") as temp:
            diag = Path(temp) / "diagnostics"
            argv = ["fake-asr", "-f", "generated.wav"]
            timeout = subprocess.TimeoutExpired(argv, 180, output=b"partial transcript", stderr=b"decoder stalled")
            with patch.dict(os.environ, {"REGRESSION_DIAGNOSTICS_DIR": str(diag)}), \
                    patch.object(run_one.subprocess, "run", side_effect=timeout):
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_one._run_captured(argv, check=False, timeout=180, label="roundtrip-asr")
            evidence = json.loads((diag / "roundtrip-asr.json").read_text())
            self.assertEqual(evidence["argv"], argv)
            self.assertTrue(evidence["timed_out"])
            self.assertIsNone(evidence["returncode"])
            self.assertEqual(evidence["stdout"]["text"], "partial transcript")
            self.assertEqual(evidence["stderr"]["text"], "decoder stalled")


if __name__ == "__main__":
    unittest.main()
