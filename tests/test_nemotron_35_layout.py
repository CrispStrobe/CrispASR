"""Model-free storage-layout checks for the issue #577 header diagnostic."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "nemotron_layout", Path(__file__).resolve().parents[1] / "tools" / "inspect_nemotron_35_layout.py")
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


class LayoutTests(unittest.TestCase):
    def test_expert_count_uses_every_dimension_and_ne0_block_fit(self):
        header = {"kv": {"general.architecture": "nemotron_h_moe"}, "header_bytes": 123,
                  "n_tensors": 4, "dtype_hist": {"Q8_0": 3, "F32": 1}, "tensors": [
                      ("blk.1.ffn_up_exps.weight", [2688, 1856, 128], "Q8_0", 0, 0),
                      ("blk.1.ffn_down_exps.weight", [1856, 2688, 128], "Q8_0", 0, 0),
                      ("blk.0.ssm_out.weight", [4096, 2688], "Q8_0", 0, 0),
                      ("norm.weight", [2688], "F32", 0, 0),
                  ]}
        result = layout.summarize(header)
        self.assertEqual(result["expert_bank_tensors"], 2)
        self.assertEqual(result["expert_bank_elements"], 2 * 2688 * 1856 * 128)
        self.assertEqual(result["row_block_compatibility"]["32"]["tensors"], 3)
        self.assertEqual(result["row_block_compatibility"]["256"]["tensors"], 1)
        # Total element divisibility cannot substitute for ne[0] compatibility:
        # both expert banks have 256-divisible totals but incompatible rows.
        self.assertEqual(result["row_block_compatibility"]["256"]["elements"], 4096 * 2688)

    def test_float_encoder_does_not_report_a_quantized_matrix_fraction(self):
        result = layout.summarize({"kv": {"general.architecture": "pee"}, "header_bytes": 16,
                                   "n_tensors": 1, "dtype_hist": {"F16": 1},
                                   "tensors": [("encoder.weight", [1280, 1280], "F16", 0, 0)]})
        self.assertEqual(result["quantized_matrix_tensors"], 0)
        self.assertIsNone(result["row_block_compatibility"]["256"]["fraction"])


if __name__ == "__main__":
    unittest.main()
