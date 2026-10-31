#!/usr/bin/env python3
"""Negative controls for the opt-in model qualification driver; no models/GPU."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("post_release_model_gate",
                                             ROOT / "tools/ci-heavy/post_release_model_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


class ModelGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = "a" * 40
        (self.root / "fixture").write_bytes(b"immutable test bytes")
        pins = {"path": "fixture", "sha256": gate.digest(self.root / "fixture")}
        self.manifest = {
            "version": 1, "source_sha": self.source, "fixture_root": ".",
            "files": {name: dict(pins) for name in ("model", "ref", "audio", "adapter", "embedding", "array", "text", "lock")},
            "cases": [],
        }
        for backend, scope in gate.SCOPES.items():
            case = {
                "id": backend, "backend": backend,
                "model_revision": "b" * 40, "reference_revision": "c" * 40,
                "original_code_revision": "d" * 40,
                "model": "model", "reference": "ref", "audio": "audio", "adapter": "adapter", "reference_lock": "lock",
                "argv": ["{python}", "{file:adapter}", "--out", "{out}"],
                "arrays": {role: {"actual": role + ".npy", "reference": "array",
                                   "max_abs": 1e-4, "relative_l2_max": 1e-4, "cosine_min": 0.999}
                           for role in scope["arrays"]},
                "texts": {role: {"actual": role + ".txt", "reference": "text"} for role in scope["texts"]},
            }
            if backend == "meanvc2":
                case["speaker_embedding"] = "embedding"
                case["argv"] += ["--embedding", "{file:embedding}"]
            self.manifest["cases"].append(case)

    def check_manifest(self, manifest=None):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest if manifest is None else manifest), encoding="utf-8")
        return gate.preflight(path, self.source)

    def test_complete_pinned_inputs_validate(self):
        manifest, files = self.check_manifest()
        self.assertEqual(len(manifest["cases"]), 3)
        self.assertEqual(set(files), set(self.manifest["files"]))

    def test_changed_input_fails_hash(self):
        (self.root / "fixture").write_bytes(b"changed bytes")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.check_manifest()

    def test_wrong_source_revision_fails(self):
        self.manifest["source_sha"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "checked-out"):
            self.check_manifest()

    def test_missing_backend_fails(self):
        self.manifest["cases"].pop()
        with self.assertRaisesRegex(ValueError, "ArkTTS, Haas, and MeanVC2"):
            self.check_manifest()

    def test_missing_arrays_and_aliases_fail(self):
        for key, role in (("arrays", "decode_k"), ("texts", "synthesized_text")):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["cases"][0][key].pop(role)
                with self.assertRaisesRegex(ValueError, "missing"):
                    self.check_manifest(manifest)

    def test_unpinned_or_unused_adapter_fails(self):
        self.manifest["cases"][0]["argv"].insert(0, "unreviewed-program")
        with self.assertRaisesRegex(ValueError, "pinned adapter"):
            self.check_manifest()

    def test_meanvc2_must_receive_real_embedding(self):
        self.manifest["cases"][-1]["argv"] = ["{python}", "{file:adapter}", "--out", "{out}"]
        with self.assertRaisesRegex(ValueError, "receive the pinned speaker embedding"):
            self.check_manifest()

    def test_missing_nonfinite_or_loosened_intermediate_bounds_fail(self):
        for role, mutation in (("decode_k", {"cosine_min": float("nan")}),
                               ("decode_k", {"cosine_min": 0}),
                               ("decode_k", {"comparison": "cosine"}),
                               ("waveform", {"comparison": "cosine"})):
            with self.subTest(role=role, mutation=mutation):
                manifest = copy.deepcopy(self.manifest)
                manifest["cases"][0]["arrays"][role].update(mutation)
                with self.assertRaises(ValueError):
                    self.check_manifest(manifest)

    def test_path_escape_fails(self):
        self.manifest["cases"][0]["arrays"]["decode_k"]["actual"] = "../previous-run.npy"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.check_manifest()

    def test_stage_success_requires_all_named_stages(self):
        names = sorted(gate.SCOPES["arktts"]["stages"])
        text = "\n".join("[PASS] " + name + " cos_min=1.000000" for name in names)
        self.assertEqual(gate.verify_diff(text, "arktts"), names)
        for bad in ("summary: 0 pass, 0 fail", text.replace("[PASS]", "[SKIP]", 1), text + "\n[FAIL] extra"):
            with self.subTest(text=bad), self.assertRaises(ValueError):
                gate.verify_diff(bad, "arktts")

    def test_cuda_acceptance_requires_actual_kernel_instances(self):
        header = '"Time (%)","Total Time (ns)","Instances","Name"\n'
        self.assertEqual(gate.cuda_kernel_instances("report header\n" + header + '100,2000,7,"kernel_name"\n'), 7)
        for text in ("GPU is present", header, header + '100,0,0,"kernel_name"\n'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                gate.cuda_kernel_instances(text)

    def test_array_gates_reject_shape_nan_and_numerical_regression(self):
        expected = self.root / "expected.npy"
        actual = self.root / "actual.npy"
        np.save(expected, np.ones((3, 4), dtype=np.float32))
        spec = {"max_abs": 1e-4, "relative_l2_max": 1e-4, "cosine_min": 0.999}
        np.save(actual, np.ones((3, 4), dtype=np.float32))
        self.assertEqual(gate.compare_array(actual, expected, spec)["max_abs"], 0)
        for data in (np.ones((4, 3)), np.full((3, 4), np.nan), np.full((3, 4), 2.0)):
            with self.subTest(shape=data.shape):
                np.save(actual, data)
                with self.assertRaises(ValueError):
                    gate.compare_array(actual, expected, spec)
        np.save(actual, np.zeros(256))
        np.save(expected, np.zeros(256))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            gate.compare_array(actual, expected, {"comparison": "cosine", "cosine_min": 0.9})

    def test_waveform_gate_rejects_silent_and_wrong_duration(self):
        expected = self.root / "expected.npy"
        actual = self.root / "actual.npy"
        np.save(expected, np.full(100, 0.1))
        spec = {"comparison": "waveform", "duration_ratio_min": 0.9, "duration_ratio_max": 1.1,
                "rms_ratio_min": 0.9, "rms_ratio_max": 1.1, "peak_abs_max": 1.0}
        np.save(actual, np.full(100, 0.1))
        self.assertEqual(gate.compare_array(actual, expected, spec)["duration_ratio"], 1)
        for data in (np.zeros(100), np.full(50, 0.1)):
            np.save(actual, data)
            with self.assertRaises(ValueError):
                gate.compare_array(actual, expected, spec)

    def test_nonzero_process_exit_is_retained_and_fails(self):
        with self.assertRaisesRegex(RuntimeError, "exited 7"):
            gate.run_command([sys.executable, "-c", "print('failure evidence'); raise SystemExit(7)"],
                             self.root, "negative")
        self.assertIn("failure evidence", (self.root / "negative.log").read_text())

    def test_stale_capture_directory_fails_before_inference(self):
        build = self.root / "build"
        (build / "bin").mkdir(parents=True)
        (build / "bin" / "crispasr-diff").write_bytes(b"not executable; must not be invoked")
        out = self.root / "out"
        (out / "arktts").mkdir(parents=True)
        with self.assertRaises(FileExistsError):
            gate.run_cases(self.manifest, {}, build, out)


if __name__ == "__main__":
    unittest.main()
