#!/usr/bin/env python3
"""False-success controls for the paired model benchmark (no models needed)."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_mel_ab", ROOT / "tools/bench_mel_ab.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "cli"
        self.binary.write_bytes(b"synthetic executable identity")
        (self.root / "model.gguf").write_bytes(b"synthetic model identity")
        (self.root / "audio.wav").write_bytes(b"synthetic audio identity")
        self.case = {"name": "fixture", "model": "model.gguf", "audio": "audio.wav",
                     "language": "en", "expected_transcript": "Every run must match."}
        self.calls = []

    def runner(self, command, **kwargs):
        self.calls.append(kwargs["env"].get("CRISPASR_MEL_SERIAL"))
        prefix = command[command.index("-of") + 1]
        Path(prefix + ".txt").write_text("Every run\nmust match.\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "transcribed 2.0s audio in 0.5s (4.0x realtime)", "")

    def collect(self, **kwargs):
        return BENCH.collect(self.binary, [self.case], self.root / "result", base_dir=self.root,
                             warmups=1, runs=3, runner=kwargs.pop("runner", self.runner), **kwargs)

    def failed_receipt(self):
        receipt = json.loads((self.root / "result/receipt.json").read_text())
        self.assertFalse(receipt["complete"])
        self.assertFalse(receipt["quality_pass"])
        self.assertIn("error", receipt)

    def test_every_run_checked_and_inherited_serial_is_overridden(self):
        with patch.dict(os.environ, {"CRISPASR_MEL_SERIAL": "1"}):
            receipt = self.collect()
        self.assertEqual(self.calls, ["1", None, "1", None, None, "1", "1", None])
        self.assertTrue(receipt["complete"])
        self.assertTrue(receipt["quality_pass"])
        self.assertEqual(len(receipt["cases"][0]["events"]), 8)
        self.assertIn("model load", receipt["timing_scope"])
        self.assertEqual(receipt["cases"][0]["median_reported_inference_s"], {"A": 0.5, "B": 0.5})

    def test_presence_only_serial_control_selects_distinct_arms(self):
        # core/mel.cpp enables parallelism only when getenv(MEL_SERIAL) is
        # nullptr. Simulate that production predicate, rather than accepting
        # "0" as false in the test runner and silently measuring serial twice.
        selected_paths = []

        def presence_control(command, **kwargs):
            selected_paths.append("serial" if "CRISPASR_MEL_SERIAL" in kwargs["env"] else "parallel")
            return self.runner(command, **kwargs)

        with patch.dict(os.environ, {"CRISPASR_MEL_SERIAL": "1"}):
            receipt = BENCH.collect(self.binary, [self.case], self.root / "result", base_dir=self.root,
                                    warmups=0, runs=1, runner=presence_control)
        self.assertEqual(selected_paths, ["serial", "parallel"])
        self.assertEqual([event["control"] for event in receipt["cases"][0]["events"]], ["1", "unset"])

    def test_a_later_transcript_mismatch_fails(self):
        def corrupt(command, **kwargs):
            result = self.runner(command, **kwargs)
            if len(self.calls) == 6:
                prefix = command[command.index("-of") + 1]
                Path(prefix + ".txt").write_text("A later run changed a word.")
            return result
        with self.assertRaises(RuntimeError):
            self.collect(runner=corrupt)
        self.assertEqual(len(self.calls), 6)
        self.failed_receipt()

    def test_missing_transcript_is_not_replaced_by_stdout_logging(self):
        def missing(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "Everything looks successful!", "")
        with self.assertRaises(RuntimeError):
            self.collect(runner=missing)
        self.failed_receipt()

    def test_empty_transcript_fails(self):
        def empty(command, **kwargs):
            result = self.runner(command, **kwargs)
            Path(command[command.index("-of") + 1] + ".txt").write_text("  \n")
            return result
        with self.assertRaises(RuntimeError):
            self.collect(runner=empty)
        self.failed_receipt()

    def test_nonzero_exit_fails_even_with_matching_transcript(self):
        def failed(command, **kwargs):
            result = self.runner(command, **kwargs)
            result.returncode = 7
            return result
        with self.assertRaises(RuntimeError):
            self.collect(runner=failed)
        self.failed_receipt()

    def test_missing_model_fails_before_any_process(self):
        self.case["model"] = "absent.gguf"
        with self.assertRaises(FileNotFoundError):
            self.collect()
        self.assertEqual(self.calls, [])
        self.failed_receipt()

    def test_hf_model_overrides_fail_before_any_process(self):
        # These CLI options can replace the explicit -m path after parsing;
        # accepting them would make the receipt hash the wrong model.
        for index, flag in enumerate(("--hf-repo", "-hfr", "--hf-file", "-hff")):
            for joined in (False, True):
                with self.subTest(flag=flag, joined=joined):
                    case = dict(self.case, extra_args=[flag + "=override"] if joined else [flag, "override"])
                    output = self.root / f"rejected-{index}-{joined}"
                    with self.assertRaisesRegex(ValueError, "extra_args cannot override"):
                        BENCH.collect(self.binary, [case], output, base_dir=self.root,
                                      warmups=0, runs=1, runner=self.runner)
                    self.assertEqual(self.calls, [])
                    receipt = json.loads((output / "receipt.json").read_text())
                    self.assertFalse(receipt["complete"])
                    self.assertFalse(receipt["quality_pass"])
                    self.assertIn("ValueError", receipt["error"])

    def test_empty_suite_and_missing_truth_are_errors(self):
        with self.assertRaises(ValueError):
            BENCH.validate_cases([], self.root)
        del self.case["expected_transcript"]
        with self.assertRaises(ValueError):
            self.collect()
        self.failed_receipt()

    def test_timeout_leaves_failed_receipt(self):
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        with self.assertRaises(subprocess.TimeoutExpired):
            self.collect(runner=timeout)
        self.failed_receipt()

    def test_existing_receipt_is_not_overwritten(self):
        self.collect()
        before = (self.root / "result/receipt.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.collect()
        self.assertEqual((self.root / "result/receipt.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
