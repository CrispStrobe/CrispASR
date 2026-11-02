#!/usr/bin/env python3
"""SCNet checkpoint, matrix-layout, and precision-contract regression tests."""

import copy
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("convert_scnet", ROOT / "models/convert-scnet-to-gguf.py")
CONVERT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERT)
from reference_backends import scnet  # noqa: E402


class ConverterTests(unittest.TestCase):
    def test_complete_checkpoint_schemas(self):
        small, large = scnet.expected_shapes("scnet"), scnet.expected_shapes("scnet-large")
        self.assertEqual(set(small), set(large))
        # 204 encoder tensors, 24 decoder tensors, and 144 recurrent tensors.
        self.assertEqual(len(small), 372)
        self.assertEqual(sum(math.prod(x) for x in small.values()), 10_578_768)
        self.assertEqual(sum(math.prod(x) for x in large.values()), 42_181_232)
        self.assertEqual(small["decoder.2.1.convtrs.2.weight"], (32, 16, 16, 1))
        self.assertEqual(large["separation_net.dp_modules.5.lstm_layers.1.weight_hh_l0_reverse"], (2048, 512))

    def test_reject_missing_extra_and_wrong_shapes(self):
        shapes = scnet.expected_shapes()
        state = {name: SimpleNamespace(shape=shape) for name, shape in shapes.items()}
        scnet.validate_state(state)
        key = next(iter(state))
        missing = dict(state)
        del missing[key]
        with self.assertRaisesRegex(ValueError, "missing="):
            scnet.validate_state(missing)
        extra = dict(state, unexpected=SimpleNamespace(shape=(1,)))
        with self.assertRaisesRegex(ValueError, "unexpected="):
            scnet.validate_state(extra)
        wrong = dict(state)
        wrong[key] = SimpleNamespace(shape=tuple(reversed(shapes[key])))
        with self.assertRaisesRegex(ValueError, "expected original shape"):
            scnet.validate_state(wrong)

    def test_transposed_convolution_layout_by_output_value(self):
        rng = np.random.default_rng(17)
        weight = rng.normal(size=(3, 2, 4, 1)).astype(np.float32)
        inputs = rng.normal(size=(3, 5)).astype(np.float32)
        expected = np.zeros((2, 12), dtype=np.float32)
        for ci in range(3):
            for co in range(2):
                for frame in range(5):
                    for tap in range(4):
                        expected[co, 2 * frame + tap] += weight[ci, co, tap, 0] * inputs[ci, frame]
        matrix = CONVERT.convert_tensor("decoder.0.1.convtrs.0.weight", weight, "f32")
        columns = (matrix @ inputs).reshape(2, 4, 5)
        actual = np.zeros_like(expected)
        for frame in range(5):
            actual[:, 2 * frame:2 * frame + 4] += columns[:, :, frame]
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)

    def test_convolution_layout_by_patch_inner_product(self):
        weight = np.arange(2 * 3 * 2 * 3, dtype=np.float32).reshape(2, 3, 2, 3) / 17
        patch = np.sin(np.arange(3 * 2 * 3, dtype=np.float32)).reshape(3, 2, 3)
        matrix = CONVERT.convert_tensor("encoder.0.globalconv.weight", weight, "f32")
        actual = matrix @ patch.ravel()
        expected = np.array([(weight[index] * patch).sum() for index in range(2)])
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_lstm_gate_order_is_unchanged(self):
        value = np.arange(32, dtype=np.float32).reshape(8, 4)
        converted = CONVERT.convert_tensor("separation_net.dp_modules.0.lstm_layers.0.weight_ih_l0", value)
        self.assertEqual(converted.dtype, np.float16)
        np.testing.assert_array_equal(converted, value)

    def test_vector_precision_and_overflow(self):
        values = np.array([1 + 2**-20, -0.003], dtype=np.float32)
        for name in ("encoder.0.globalconv.bias", "separation_net.dp_modules.0.norm_layers.0.weight"):
            result = CONVERT.convert_tensor(name, values)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result, values)
        with self.assertRaisesRegex(ValueError, "F16 overflow"):
            CONVERT.convert_tensor("matrix.weight", np.array([[70000]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            CONVERT.convert_tensor("matrix.weight", np.array([[np.nan]], dtype=np.float32))

    def test_configuration_mismatch_fails(self):
        cfg = scnet.model_config()
        CONVERT.validate_config({"model": cfg}, "scnet")
        for key, bad in (("hop_size", 512), ("normalized", 1), ("dims", [4, 64, 128, 256])):
            changed = copy.deepcopy(cfg)
            changed[key] = bad
            with self.assertRaisesRegex(ValueError, "unsupported"):
                CONVERT.validate_config(changed, "scnet")
        with self.assertRaisesRegex(ValueError, "exactly"):
            CONVERT.validate_config(dict(cfg, unknown=1), "scnet")

    def test_checkpoint_hash_is_checked_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.th"
            path.write_bytes(b"example tensor-only checkpoint")
            digest = scnet.sha256(path)
            self.assertEqual(scnet.verify_checkpoint(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                scnet.verify_checkpoint(path, "0" * 64)
            with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
                scnet.verify_checkpoint(path, "not-a-checksum")


if __name__ == "__main__":
    unittest.main()
