"""Controls against false acceptance in the actual-model qualification runner."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "mini_bs_validation", Path(__file__).resolve().parents[1] / "tools" / "validate_mini_bs_roformer.py")
VALIDATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATION)


class ValidationControls(unittest.TestCase):
    def test_atomic_checkpoint_replaces_stale_state_without_temp_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            sentinel = Path(directory) / "progress.json.tmp"
            target.write_text('{"run_id":"old","acceptance":true}')
            sentinel.write_text("another writer")
            current = {"run_id": "new", "complete": False, "acceptance": False, "status": "initializing"}
            VALIDATION.atomic_json_write(target, current)
            self.assertEqual(json.loads(target.read_text()), current)
            self.assertEqual(sentinel.read_text(), "another writer")
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {target.name, sentinel.name})

    def test_atomic_checkpoint_rejects_nan_without_losing_prior_data(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text('{"complete":false}')
            with self.assertRaises(ValueError):
                VALIDATION.atomic_json_write(target, {"score": float("nan")})
            self.assertEqual(json.loads(target.read_text()), {"complete": False})
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_atomic_checkpoint_cleans_up_when_publication_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "progress.json"
            target.write_text('{"run_id":"prior"}')
            with mock.patch.object(VALIDATION.os, "replace", side_effect=OSError("publication failed")):
                with self.assertRaises(OSError):
                    VALIDATION.atomic_json_write(target, {"run_id": "new"})
            self.assertEqual(json.loads(target.read_text()), {"run_id": "prior"})
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_existing_quant_reuse_requires_identical_tool_output(self):
        with tempfile.TemporaryDirectory() as directory:
            original, generated = (Path(directory) / name for name in ("existing.gguf", "generated.gguf"))
            original.write_bytes(b"quantized artifact")
            generated.write_bytes(original.read_bytes())
            self.assertEqual(VALIDATION.verify_existing_quant(generated, original), VALIDATION.sha256(original))
            self.assertEqual(original.read_bytes(), b"quantized artifact")
            # The caller records this hash before deleting the temporary file.
            self.assertTrue(generated.exists())

    def test_existing_quant_mismatch_retains_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            original, generated = (Path(directory) / name for name in ("existing.gguf", "generated.gguf"))
            original.write_bytes(b"accepted artifact")
            generated.write_bytes(b"different tool output")
            with self.assertRaises(ValueError):
                VALIDATION.verify_existing_quant(generated, original)
            self.assertEqual(original.read_bytes(), b"accepted artifact")
            self.assertEqual(generated.read_bytes(), b"different tool output")

    def test_existing_quant_cannot_alias_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "existing.gguf"
            original.write_bytes(b"accepted artifact")
            with self.assertRaises(ValueError):
                VALIDATION.verify_existing_quant(original, original)
            self.assertEqual(original.read_bytes(), b"accepted artifact")

    def log(self, score=1.0, row=1.0):
        return "\n".join(f"[PASS] {stage} cos_global={score} cos_min={row} norm_ratio=1 n_nonfinite=0"
                         for stage in VALIDATION.REQUIRED_STAGES)

    def test_complete_stage_set_passes(self):
        self.assertEqual(len(VALIDATION.parse_diff(self.log(), "f16")), 25)

    def test_outputs_only_must_be_explicit_and_include_all_stems(self):
        log = "\n".join(line for line in self.log().splitlines()
                        if line.split()[1] in VALIDATION.OUTPUT_STAGES)
        self.assertEqual(len(VALIDATION.parse_diff(log, "f16", outputs_only=True)), 5)
        with self.assertRaises(ValueError):
            VALIDATION.parse_diff(log, "f16")
        with self.assertRaises(ValueError):
            VALIDATION.parse_diff("\n".join(log.splitlines()[:-1]), "f16", outputs_only=True)

    def test_thresholds_are_strict(self):
        for precision in ("q8_0", "q4_k"):
            with self.subTest(precision=precision), self.assertRaises(ValueError):
                VALIDATION.parse_diff(self.log(VALIDATION.THRESHOLDS[precision]), precision)

    def test_f16_catches_local_layout_error(self):
        with self.assertRaises(ValueError):
            VALIDATION.parse_diff(self.log(1.0, 0.8), "f16")

    def test_quant_reports_fragile_rows_without_false_failure(self):
        report = VALIDATION.parse_diff(self.log(0.995, 0.7), "q8_0")
        self.assertEqual(report["mask_raw"]["cos_min"], 0.7)

    def test_f16_stft_pair_minimum_is_reported_but_not_a_gate(self):
        log = self.log().replace("stft_output cos_global=1.0 cos_min=1.0",
                                 "stft_output cos_global=1.0 cos_min=-0.3")
        report = VALIDATION.parse_diff(log, "f16")
        self.assertEqual(report["stft_output"]["cos_min"], -0.3)

    def test_missing_and_duplicate_stages_fail(self):
        lines = self.log().splitlines()
        for log in ("\n".join(lines[:-1]), self.log() + "\n" + lines[0]):
            with self.subTest(log=log[-100:]), self.assertRaises(ValueError):
                VALIDATION.parse_diff(log, "q4_k")

    def test_skips_nonfinite_and_bogus_pass_fail(self):
        for change in (self.log().replace("[PASS]", "[SKIP]", 1),
                       self.log().replace("cos_global=1.0", "cos_global=nan", 1),
                       self.log().replace("n_nonfinite=0", "n_nonfinite=2", 1),
                       self.log().replace("norm_ratio=1", "norm_ratio=2", 1),
                       self.log().replace("cos_global=1.0", "unrelated=1.0", 1)):
            with self.subTest(log=change[:100]), self.assertRaises(ValueError):
                VALIDATION.parse_diff(change, "q8_0")

    def test_scaled_waveform_cannot_pass_on_cosine(self):
        reference = np.ones((4, 2, 32), dtype=np.float32)
        report = VALIDATION.waveform_metrics(reference * 2, reference, 0.98)
        self.assertTrue(all(row["cos_global"] > 0.999 for row in report.values()))
        self.assertFalse(any(row["acceptance"] for row in report.values()))

    def test_silent_replacement_fails_and_is_json_serializable(self):
        reference = np.ones((4, 2, 32), dtype=np.float32)
        silent = np.zeros_like(reference)
        for actual, expected in ((silent, reference), (reference, silent)):
            report = VALIDATION.waveform_metrics(actual, expected, 0.94)
            self.assertFalse(any(row["acceptance"] for row in report.values()))
            json.dumps(report, allow_nan=False)

    def test_stem_shape_or_nan_is_fatal(self):
        reference = np.ones((4, 2, 32), dtype=np.float32)
        for actual in (reference[:, :, :-1], np.full_like(reference, np.nan)):
            with self.subTest(shape=actual.shape), self.assertRaises(ValueError):
                VALIDATION.waveform_metrics(actual, reference, 0.94)

    def test_crispembed_recipe_pins_arithmetic_operands(self):
        command = VALIDATION.quantizer_command("crispembed-quantize", "crispembed", "in", "out", "q4_k")
        self.assertIn("*.probe=f32", command)
        self.assertIn("*.register_tokens=f32", command)
        self.assertIn("freq_domain_model.mask_model.mask_estimator.*.weight=f16", command)
        self.assertIn("freq_domain_model.time_pooling.*.weight=q8_0", command)
        self.assertEqual(VALIDATION.quantizer_command("crispasr-quantize", "crispasr", "in", "out", "q4_k"),
                         ["crispasr-quantize", "in", "out", "q4_k"])

    def test_imatrix_argument_reaches_either_real_tool(self):
        for kind in ("crispasr", "crispembed"):
            command = VALIDATION.quantizer_command("quantize", kind, "in", "out", "q4_k", "calibration.gguf")
            self.assertEqual(command[-2:], ["--imatrix", "calibration.gguf"])


if __name__ == "__main__":
    unittest.main()
