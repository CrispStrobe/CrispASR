"""Small model-free fixtures for the original CSM RoPE component comparator."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ci-heavy"))
from csm_rope_component_check import CONFIG_SHA256, SOURCE_REVISION, compare, main


def fixture(root):
    reference, candidate = root / "reference", root / "candidate"
    manifest = {"complete": True, "source_revision": SOURCE_REVISION,
                "config_sha256": CONFIG_SHA256, "cases": {}}
    for case in ("backbone", "depth"):
        (reference / case).mkdir(parents=True)
        (candidate / case).mkdir(parents=True)
        q = np.array([[[0.2, -0.4, 0.7, 0.9]], [[0.3, 0.5, -0.2, 0.8]]], dtype="<f4")
        k = np.array([[[0.8, 0.1, -0.3, 0.6]], [[-0.5, 0.9, 0.4, 0.2]]], dtype="<f4")
        arrays = {"positions": np.array([0, 32], dtype="<i4"),
                  "q": q, "k": k, "q_rotated": q.copy(), "k_rotated": k.copy(),
                  "q_unscaled": q * 2, "k_unscaled": k * 2,
                  "freq_factors": np.array([1, 32], dtype="<f4")}
        files = {}
        for key, array in arrays.items():
            raw = array.tobytes()
            (reference / case / f"{key}.bin").write_bytes(raw)
            files[key] = {"shape": list(array.shape), "dtype": "int32" if key == "positions" else "float32",
                          "raw_sha256": hashlib.sha256(raw).hexdigest()}
        manifest["cases"][case] = {"files": files}
        for key in ("q_rotated", "k_rotated", "freq_factors"):
            (candidate / case / f"{key}.bin").write_bytes(arrays[key].tobytes())
    (reference / "receipt.json").write_text(json.dumps(manifest))
    return reference, candidate


class CsmRopeComponentCheckTests(unittest.TestCase):
    def test_exact_candidate_passes_and_negative_controls_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, candidate = fixture(Path(tmp))
            receipt = compare(reference, candidate)
            self.assertTrue(receipt["pass"])
            for case in ("backbone", "depth"):
                self.assertTrue(all(item["pass"] for item in receipt["cases"][case]["candidate"].values()))
                self.assertGreater(receipt["cases"][case]["unscaled_negative_control"]["q"]["rel_l2"], 0.1)

    def test_cosine_one_with_wrong_norm_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, candidate = fixture(Path(tmp))
            path = candidate / "backbone" / "q_rotated.bin"
            q = np.frombuffer(path.read_bytes(), dtype="<f4").copy()
            path.write_bytes((q * np.float32(1.00001)).tobytes())
            result = compare(reference, candidate)
            metrics = result["cases"]["backbone"]["candidate"]["q_rotated"]
            self.assertFalse(result["pass"])
            self.assertAlmostEqual(metrics["cosine"], 1.0)
            self.assertGreater(abs(metrics["norm_ratio"] - 1), 1e-6)

    def test_wrong_frequency_factor_fails_its_locked_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, candidate = fixture(Path(tmp))
            path = candidate / "depth" / "freq_factors.bin"
            path.write_bytes(np.array([1, 32.001], dtype="<f4").tobytes())
            result = compare(reference, candidate)
            self.assertFalse(result["pass"])
            self.assertFalse(result["cases"]["depth"]["candidate"]["freq_factors"]["pass"])

    def test_zero_nan_and_truncated_candidate_rejected(self):
        for damage in ("zero", "nan", "truncated"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as tmp:
                reference, candidate = fixture(Path(tmp))
                path = candidate / "depth" / "k_rotated.bin"
                data = bytearray(path.read_bytes())
                if damage == "zero":
                    data[:] = bytes(len(data))
                elif damage == "nan":
                    data[:4] = np.float32(np.nan).tobytes()
                else:
                    del data[-4:]
                path.write_bytes(data)
                if damage == "zero":
                    with self.assertRaisesRegex(ValueError, "zero norm"):
                        compare(reference, candidate)
                else:
                    with self.assertRaises(ValueError):
                        compare(reference, candidate)

    def test_corrupt_reference_hash_and_cli_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, candidate = fixture(root)
            path = reference / "depth" / "q_rotated.bin"
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "reference hash mismatch"):
                compare(reference, candidate)
            receipt = root / "check.json"
            self.assertEqual(main([str(reference), str(candidate), str(receipt)]), 1)
            self.assertFalse(json.loads(receipt.read_text())["pass"])


if __name__ == "__main__":
    unittest.main()
