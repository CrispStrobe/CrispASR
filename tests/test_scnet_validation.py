"""Negative controls for the actual SCNet qualification/benchmark checker."""

import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "scnet_validation", Path(__file__).resolve().parents[1] / "tools" / "validate_scnet.py")
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)


def string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_bytes(name, value):
    if isinstance(value, str):
        return string(name) + struct.pack("<I", 8) + string(value)
    if isinstance(value, float):
        return string(name) + struct.pack("<If", 6, value)
    return string(name) + struct.pack("<IQ", 10, value)


def write_gguf(path, tensors, metadata=None):
    """Independent compact GGUF emitter, no native or gguf package dependency."""
    metadata = metadata or {}
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        header.extend(metadata_bytes(key, value))
    payload = bytearray()
    for name, shape, kind, data in tensors:
        header.extend(string(name) + struct.pack("<I", len(shape)))
        header.extend(struct.pack("<" + "Q" * len(shape), *shape))
        header.extend(struct.pack("<IQ", kind, len(payload)))
        payload.extend(data)
        payload.extend(b"\0" * (-len(payload) % 32))
    header.extend(b"\0" * (-len(header) % 32))
    path.write_bytes(header + payload)
    return path


def captured_tensors(input_audio=None):
    values = []
    for index, name in enumerate(V.REQUIRED_STAGES):
        data = np.array([0.2, -0.7, 0.1, index + 1.0], dtype="<f4")
        shape = [4]
        if name == "input_audio" and input_audio is not None:
            data, shape = np.ascontiguousarray(input_audio.T, dtype="<f4"), [len(input_audio), 2]
        values.append((name, shape, 0, data.tobytes()))
    return values


class SCNetValidationControls(unittest.TestCase):
    def log(self, precision="f16", cosine=1.0):
        rows = [f"[PASS] {name} cos_global={cosine} norm_ratio=1 relative_l2=0 max_abs=0 nonfinite=0 n=4"
                for name in V.REQUIRED_STAGES]
        rows.append(f"scnet diff: 35 required stages, 0 failed, strict cosine > {V.THRESHOLDS[precision]:.3f}")
        return "\n".join(rows)

    def test_complete_35_stage_log_and_strict_thresholds(self):
        self.assertEqual(len(V.REQUIRED_STAGES), 35)
        for precision in V.PRECISIONS:
            self.assertEqual(len(V.parse_diff(self.log(precision), precision)), 35)
            with self.assertRaises(ValueError):
                V.parse_diff(self.log(precision, V.THRESHOLDS[precision]), precision)

    def test_missing_duplicate_skipped_bad_summary_and_fake_pass_fail(self):
        original = self.log()
        bad = ["\n".join(original.splitlines()[1:]), original + "\n" + original.splitlines()[0],
               original.replace("[PASS]", "[SKIP]", 1), original.replace("[PASS]", "[FAIL]", 1),
               original.replace("35 required", "34 required"),
               original.replace("cos_global=1.0", "cos_global=nan", 1),
               original.replace("relative_l2=0", "relative_l2=inf", 1),
               original.replace("nonfinite=0", "nonfinite=1", 1),
               original.replace("norm_ratio=1", "norm_ratio=2", 1),
               original.replace("n=4", "n=0", 1),
               original.replace("max_abs=0", "other=0", 1)]
        for log in bad:
            with self.subTest(log=log[:90]), self.assertRaises(ValueError):
                V.parse_diff(log, "f16")

    def test_independent_capture_comparison_rejects_missing_shape_and_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rows = captured_tensors()
            reference = V.GGUFFile(write_gguf(work / "reference.gguf", rows))
            actual = V.GGUFFile(write_gguf(work / "actual.gguf", rows))
            self.assertTrue(V.all_pass(V.compare_captures(actual, reference, "f16")))
            missing = V.GGUFFile(write_gguf(work / "missing.gguf", rows[:-1]))
            with self.assertRaises(ValueError):
                V.compare_captures(missing, reference, "f16")
            wrong = rows.copy()
            name, _, kind, data = wrong[0]
            wrong[0] = (name, [2, 2], kind, data)
            mismatch = V.GGUFFile(write_gguf(work / "wrong-shape.gguf", wrong))
            with self.assertRaises(ValueError):
                V.compare_captures(mismatch, reference, "f16")
            scaled = [(name, shape, kind, (np.frombuffer(data, dtype="<f4") * 2).tobytes())
                      for name, shape, kind, data in rows]
            observed = V.GGUFFile(write_gguf(work / "scaled.gguf", scaled))
            report = V.compare_captures(observed, reference, "q4_k")
            self.assertFalse(V.all_pass(report))
            self.assertTrue(all(row["cos_global"] > .999 for row in report.values()))

    def test_rank_padding_preserves_geometry_but_not_axis_swaps(self):
        self.assertEqual(V.shape4([8, 2]), V.shape4([8, 2, 1, 1]))
        self.assertNotEqual(V.shape4([8, 2]), V.shape4([2, 8]))

    def test_nonfinite_and_silent_replacements_cannot_pass(self):
        actual = np.ones((4, 2, 16), np.float32)
        silent = np.zeros_like(actual)
        for x, y in ((actual, silent), (silent, actual)):
            report = V.waveform_metrics(x, y, "q4_k")
            self.assertFalse(V.all_pass(report))
            json.dumps(report, allow_nan=False)
        for bad in (actual[:, :, :-1], actual * np.nan):
            with self.assertRaises(ValueError):
                V.waveform_metrics(bad, actual, "f16")

    def test_f16_five_percent_gain_and_relative_error_fail(self):
        expected = np.arange(1, 65, dtype=np.float32).reshape(4, 2, 8)
        report = V.waveform_metrics(expected * 1.05, expected, "f16")
        self.assertFalse(V.all_pass(report))
        self.assertTrue(all(row["cos_global"] > .999999 for row in report.values()))
        with self.assertRaises(ValueError):
            V.parse_diff(self.log().replace("norm_ratio=1", "norm_ratio=1.05", 1), "f16")
        with self.assertRaises(ValueError):
            V.parse_diff(self.log("q8_0").replace("relative_l2=0", "relative_l2=0.21", 1), "q8_0")

    def test_local_row_failure_cannot_hide_in_global_f16_cosine(self):
        reference = np.ones((2000, 8), dtype=np.float32)
        observed = reference.copy()
        observed[0, :4] = -1
        global_report = V.tensor_metrics(observed, reference, "f16")
        rows = V.row_metrics(observed, reference, 8)
        self.assertGreater(global_report["cos_global"], .999)
        self.assertEqual(rows["rows_failing_f16_cosine"], 1)
        self.assertEqual(rows["cos_min"], 0)
        zero = V.row_metrics(np.zeros((2, 8)), np.zeros((2, 8)), 8)
        self.assertEqual(zero["zero_row_pairs"], 2)
        self.assertEqual(zero["cos_min"], 1)
        one_changed = np.zeros((2, 8))
        one_changed[0, 0] = 1e-30
        mismatch = V.row_metrics(one_changed, np.zeros((2, 8)), 8)
        self.assertEqual(mismatch["zero_row_mismatches"], 1)
        self.assertEqual(mismatch["cos_min"], 0)

    def model_metadata(self, precision):
        return {"general.architecture": "scnet", "general.file_type": V.FILE_TYPES[precision],
                "scnet.tensor_layout_version": 1, "general.source.revision": V.SOURCE_REVISION,
                "scnet.checkpoint.sha256": "a" * 64}

    def model(self, path, precision, small_target=False, norm=0.25):
        main_kind = {"f16": 1, "q8_0": 8, "q4_k": 12}[precision]
        rows = []
        for name, shape, kind in (("matrix.weight", [256, 32], 1 if small_target else main_kind),
                                  ("small.weight", [256, 2], main_kind), ("norm.weight", [32], 0)):
            block, width = V.TYPE_GEOMETRY[kind]
            data = b"\0" * (int(np.prod(shape)) // block * width)
            if name == "norm.weight":
                data = np.full(32, norm, dtype="<f4").tobytes()
            rows.append((name, shape, kind, data))
        return V.GGUFFile(write_gguf(path, rows, self.model_metadata(precision)))

    def test_real_target_type_coverage_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            base = self.model(work / "f16.gguf", "f16")
            for precision in ("q8_0", "q4_k"):
                model = self.model(work / (precision + ".gguf"), precision)
                report = V.check_inventory(model, precision, base, "a" * 64)
                self.assertEqual(report["target_type_matrix_fraction"], 1.0)
                weak = self.model(work / (precision + "-fake.gguf"), precision, small_target=True)
                with self.assertRaises(ValueError):
                    V.check_inventory(weak, precision, base, "a" * 64)
                with self.assertRaises(ValueError):
                    V.check_inventory(model, precision, base, "b" * 64)
                with self.assertRaises(ValueError):
                    V.check_inventory(model, precision, base, "a" * 64, {"SCNet.py": "c" * 64})
                changed_norm = self.model(work / (precision + "-norm.gguf"), precision, norm=0.5)
                with self.assertRaises(ValueError):
                    V.check_inventory(changed_norm, precision, base, "a" * 64)

    def test_malformed_gguf_does_not_masquerade_as_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            good = write_gguf(work / "good.gguf", captured_tensors())
            for size in (0, 4, 12, len(good.read_bytes()) - 32):
                broken = work / "truncated.gguf"
                broken.write_bytes(good.read_bytes()[:size])
                with self.subTest(size=size), self.assertRaises(ValueError):
                    V.GGUFFile(broken)
            duplicated = write_gguf(work / "duplicate.gguf", captured_tensors() + captured_tensors()[:1])
            with self.assertRaises(ValueError):
                V.GGUFFile(duplicated)
            unsupported = write_gguf(work / "bad-type.gguf", [("tensor", [32], 2, b"\0" * 128)])
            with self.assertRaises(ValueError):
                V.GGUFFile(unsupported)

    def test_reference_requires_exact_input_and_default_full_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            audio = np.arange(32, dtype=np.float32).reshape(16, 2)
            metadata = {"scnet.reference.source_revision": V.SOURCE_REVISION,
                        "scnet.reference.checkpoint_sha256": "a" * 64,
                        "scnet.reference.input_pcm_sha256": V.pcm_sha256(audio.T),
                        "scnet.reference.mode": "raw", "scnet.reference.segment_seconds": 20,
                        "scnet.reference.shifts": 1, "scnet.reference.shift_seed": 0,
                        "scnet.reference.overlap": 0.5}
            reference = V.GGUFFile(write_gguf(work / "reference.gguf", captured_tensors(audio), metadata))
            self.assertFalse(V.validate_reference(reference, audio, "a" * 64, diagnostic=True)["full_profile"])
            with self.assertRaises(ValueError):
                V.validate_reference(reference, audio, "a" * 64)
            with self.assertRaises(ValueError):
                V.validate_reference(reference, audio + 1, "a" * 64, diagnostic=True)
            with self.assertRaises(ValueError):
                V.validate_reference(reference, audio, "b" * 64, diagnostic=True)
            with self.assertRaises(ValueError):
                V.validate_reference(reference, audio, "a" * 64, diagnostic=True,
                                     source_hashes={"SCNet.py": "c" * 64})
            # The native/reference APIs cap segments at 119 seconds so the
            # possible half-second shift remains within the 120-second bound.
            upper = dict(metadata, **{"scnet.reference.segment_seconds": 119})
            reference = V.GGUFFile(write_gguf(work / "segment119.gguf", captured_tensors(audio), upper))
            self.assertEqual(V.validate_reference(reference, audio, "a" * 64, diagnostic=True)
                             ["segment_seconds"], 119)
            excessive = dict(metadata, **{"scnet.reference.segment_seconds": 120})
            reference = V.GGUFFile(write_gguf(work / "segment120.gguf", captured_tensors(audio), excessive))
            with self.assertRaisesRegex(ValueError, "invalid reference inference parameters"):
                V.validate_reference(reference, audio, "a" * 64, diagnostic=True)

    def calls(self):
        return [{"iteration": i, "implementation": name, "seconds": 0.5 if name == "native" else 1.0,
                 "quality_acceptance": True}
                for i in range(3) for name in (("original", "native") if i % 2 == 0 else ("native", "original"))]

    def test_complete_ordered_timing_pairs_and_quality_required(self):
        calls = self.calls()
        self.assertEqual(V.summarize_pairs(calls, 3)["speedup"], 2.0)
        self.assertTrue(V.summarize_pairs(calls, 3)["performance_acceptance"])
        for bad in (calls[:-1], calls + calls[-1:], list(reversed(calls))):
            with self.assertRaises(ValueError):
                V.summarize_pairs(bad, 3)
        for value in (0, -1, float("nan"), float("inf")):
            bad = self.calls()
            bad[0]["seconds"] = value
            with self.assertRaises(ValueError):
                V.summarize_pairs(bad, 3)
        bad = self.calls()
        bad[0]["quality_acceptance"] = False
        with self.assertRaises(ValueError):
            V.summarize_pairs(bad, 3)

    def test_checkpoint_is_atomic_and_nan_cannot_replace_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            prior = {"complete": False, "calls": self.calls()[:1]}
            V.atomic_json_write(path, prior)
            with self.assertRaises(ValueError):
                V.atomic_json_write(path, {"seconds": float("nan")})
            self.assertEqual(json.loads(path.read_text()), prior)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
