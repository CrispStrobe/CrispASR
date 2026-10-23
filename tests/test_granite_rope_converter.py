"""Original Granite5 YaRN metadata must survive conversion; old models stay unscaled."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from granite_rope import rope_scaling_metadata


class GraniteRopeMetadata(unittest.TestCase):
    def test_old_default_writes_no_scaling_fields(self):
        for cfg in ({}, {"rope_scaling": None}, {"rope_scaling": {"rope_type": "default"}}):
            self.assertEqual(rope_scaling_metadata(cfg), {})

    def test_original5_values(self):
        result = rope_scaling_metadata({"max_position_embeddings": 16384, "rope_scaling": {
            "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}})
        self.assertEqual(result["type"], "yarn")
        self.assertEqual(result["original_context"], 4096)
        self.assertEqual(result["factor"], 4.0)
        self.assertEqual(result["beta_fast"], 32.0)
        self.assertEqual(result["beta_slow"], 1.0)
        self.assertAlmostEqual(result["attention_factor"], 1.138629436111989, places=14)

    def test_unsupported_or_invalid_scaling_does_not_disappear(self):
        for rope in ({"rope_type": "linear", "factor": 4},
                     {"rope_type": "yarn", "factor": 4, "truncate": False},
                     {"rope_type": "yarn", "factor": 0, "original_max_position_embeddings": 4096},
                     {"rope_type": "yarn", "factor": float("nan"), "original_max_position_embeddings": 4096},
                     {"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 4096,
                      "attention_factor": float("inf")}):
            with self.subTest(rope=rope), self.assertRaises(ValueError):
                rope_scaling_metadata({"rope_scaling": rope})


if __name__ == "__main__":
    unittest.main()
